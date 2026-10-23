"""Evaluate retrieval quality and routing on questions from medquad.csv.

Evaluation sets (fixed, so results stay comparable between versions):
  * dataset    - 200 questions sampled from medquad.csv (seed 42), in the dataset's wording
  * paraphrase - hand-written natural questions from eval_questions.csv
  * safety     - off-topic, prompt injection, greeting and emergency messages
                 from safety_questions.csv

Dev set (only used by --tune to choose MIN_RELEVANCE_SCORE, never for reported results):
  * dev_questions.csv (hand-written medical and off-topic questions)
    plus 100 other dataset questions (seed 7) that are not in the evaluation sample

A retrieved passage counts as correct when it contains at least MIN_SHARED_CHARS
consecutive characters of the expected answer. Matching on answer text (not on
row ids) means a passage holding only the question text does not count, and the
metric keeps working when the ingestion format changes.

Metrics:
  Hit@k           share of questions with a correct passage in the top k results
  MRR             mean of 1 / rank of the first correct passage (0 if none in top k)
  Answered        share of medical questions that reach the LLM
  False refusals  medical questions that were refused, and why

Usage (with the stack running):
  docker compose exec api python evaluate.py          # report results
  docker compose exec api python evaluate.py --tune   # choose the threshold on the dev set
"""

import argparse
import csv
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import ModuleType

DATASET_PATH = "medquad.csv"  # the evaluation always uses the bundled dataset
PARAPHRASE_PATH = "eval_questions.csv"
SAFETY_PATH = "safety_questions.csv"
DEV_PATH = "dev_questions.csv"
EVAL_SEED, EVAL_SAMPLE_SIZE = 42, 200
DEV_SEED, DEV_SAMPLE_SIZE = 7, 100
TOP_K = 5
MIN_SHARED_CHARS = 60

# Routes that count as the right outcome for each safety label.
SAFETY_EXPECTED = {
    "off_topic": {"off_topic", "injection"},   # refused
    "injection": {"injection", "off_topic"},   # blocked, never answered
    "greeting": {"greeting"},
    "emergency": {"emergency"},
}


@dataclass
class Result:
    query: str
    rank: int | None  # 1-based rank of the first correct passage, None if not in top k
    top_score: float
    route: str

    @property
    def answered(self) -> bool:
        return self.route == "medical"


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace so formatting differences don't matter."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_answers_by_question(path: str) -> dict[str, list[str]]:
    """Map each question to all of its answers (MedQuAD repeats some questions)."""
    answers: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            answers[row["question"].strip()].append(normalize(row["answer"]))
    return answers


def shares_answer_text(passage: str, answers: list[str], min_chars: int = MIN_SHARED_CHARS) -> bool:
    """True if the passage contains a run of at least `min_chars` characters from any answer."""
    passage = normalize(passage)
    for answer in answers:
        if len(answer) < min_chars:
            if answer and answer in passage:
                return True
            continue
        # Slide a window over the passage and look for it inside the answer.
        for start in range(0, len(passage) - min_chars + 1, 10):
            if passage[start:start + min_chars] in answer:
                return True
    return False


def evaluate(cases: list[tuple[str, list[str]]], rag: ModuleType) -> list[Result]:
    """Run retrieval and routing for each (query, expected answers) pair."""
    results = []
    for query, answers in cases:
        docs_with_scores = rag.vector_db.similarity_search_with_relevance_scores(query, k=TOP_K)
        rank = next(
            (i for i, (doc, _) in enumerate(docs_with_scores, start=1) if shares_answer_text(doc.page_content, answers)),
            None,
        )
        top_score = docs_with_scores[0][1] if docs_with_scores else 0.0
        route, _ = rag.route_question(query)
        results.append(Result(query, rank, top_score, route))
    return results


def print_summary(name: str, results: list[Result]) -> None:
    n = len(results)

    def hit_at(k: int) -> float:
        return sum(1 for r in results if r.rank is not None and r.rank <= k) / n

    mrr = sum(1 / r.rank for r in results if r.rank is not None) / n
    refused = Counter(r.route for r in results if not r.answered)
    answered = n - sum(refused.values())

    print(f"\n=== {name} ({n} questions) ===")
    print(f"Retrieval        Hit@1 {hit_at(1):6.1%}   Hit@3 {hit_at(3):6.1%}   Hit@5 {hit_at(5):6.1%}   MRR {mrr:.3f}")
    print(f"Answered         {answered / n:6.1%} ({answered}/{n})")
    reasons = ", ".join(f"{route}={count}" for route, count in refused.most_common()) or "none"
    print(f"False refusals   {(n - answered) / n:6.1%} ({n - answered}/{n})  reasons: {reasons}")


def evaluate_safety(rag: ModuleType) -> None:
    """Check that off-topic and injection messages are refused, and greetings and emergencies are recognised."""
    with open(SAFETY_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        route, _ = rag.route_question(row["query"])
        by_label[row["label"]].append((row["query"], route))

    print(f"\n=== Safety questions ({len(rows)} questions) ===")
    failures = []
    for label, expected in SAFETY_EXPECTED.items():
        cases = by_label[label]
        passed = sum(route in expected for _, route in cases)
        print(f"{label:<10}  {passed}/{len(cases)} handled correctly")
        failures += [(label, query, route) for query, route in cases if route not in expected]
    for label, query, route in failures:
        print(f"  FAIL  expected {label}, got {route}: {query}")


def tune_threshold(rag: ModuleType, answers_by_question: dict[str, list[str]], eval_questions: set[str]) -> None:
    """Sweep MIN_RELEVANCE_SCORE on the dev set and recommend a value.

    Picks the threshold that maximises (medical answered rate - off-topic accepted rate).
    If several thresholds tie, it takes the middle of that range for a safety margin.
    """
    with open(DEV_PATH, encoding="utf-8") as f:
        dev = [(row["query"], row["label"]) for row in csv.DictReader(f)]
    candidates = sorted(set(answers_by_question) - eval_questions)
    dev += [(q, "medical") for q in random.Random(DEV_SEED).sample(candidates, DEV_SAMPLE_SIZE)]

    # A question is answered when it passes the text checks and its best passage scores >= threshold.
    scored = []
    for query, label in dev:
        passes_checks = rag.precheck(query) is None
        top = rag.vector_db.similarity_search_with_relevance_scores(query, k=1)
        scored.append((label, passes_checks, top[0][1] if top else 0.0))

    def rates(threshold: float) -> tuple[float, float]:
        answered = [ok and score >= threshold for label, ok, score in scored if label == "medical"]
        accepted = [ok and score >= threshold for label, ok, score in scored if label == "off_topic"]
        return sum(answered) / len(answered), sum(accepted) / len(accepted)

    thresholds = [round(0.20 + 0.01 * i, 2) for i in range(61)]
    gains = {t: rates(t)[0] - rates(t)[1] for t in thresholds}
    best_gain = max(gains.values())
    best = [t for t in thresholds if gains[t] == best_gain]
    chosen = best[len(best) // 2]

    n_medical = sum(label == "medical" for label, _, _ in scored)
    print(f"\n=== Threshold sweep on dev set ({n_medical} medical, {len(scored) - n_medical} off-topic) ===")
    print(f"{'threshold':>9}  {'medical answered':>16}  {'off-topic accepted':>18}")
    for t in thresholds:
        if t in (round(x * 0.05, 2) for x in range(4, 17)) or t == chosen:
            answered, accepted = rates(t)
            mark = "  <- chosen" if t == chosen else ""
            print(f"{t:9.2f}  {answered:16.1%}  {accepted:18.1%}{mark}")
    print(f"\nBest range {best[0]:.2f}-{best[-1]:.2f}; recommended MIN_RELEVANCE_SCORE = {chosen:.2f}")


def print_details(results: list[Result]) -> None:
    print(f"\n{'rank':>4}  {'score':>5}  {'route':<10}  query")
    for r in results:
        rank = str(r.rank) if r.rank else "-"
        print(f"{rank:>4}  {r.top_score:5.2f}  {r.route:<10}  {r.query}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tune", action="store_true", help="choose MIN_RELEVANCE_SCORE on the dev set instead of reporting results")
    args = parser.parse_args()

    # Import the RAG module here so --help works without loading models.
    logging.disable(logging.WARNING)
    import config
    import rag

    answers_by_question = load_answers_by_question(DATASET_PATH)

    # The evaluation sample is fixed (seed and size never change) so results stay comparable.
    sample = random.Random(EVAL_SEED).sample(sorted(answers_by_question), EVAL_SAMPLE_SIZE)
    with open(PARAPHRASE_PATH, encoding="utf-8") as f:
        paraphrase_rows = list(csv.DictReader(f))

    if args.tune:
        held_out = set(sample) | {row["expected_question"] for row in paraphrase_rows}
        tune_threshold(rag, answers_by_question, held_out)
        return

    dataset_cases = [(q, answers_by_question[q]) for q in sample]
    paraphrase_cases = [(row["query"], answers_by_question[row["expected_question"]]) for row in paraphrase_rows]
    missing = [q for q, answers in paraphrase_cases if not answers]
    if missing:
        raise SystemExit(f"Expected question not found in {DATASET_PATH} for: {missing}")

    print(f"MIN_RELEVANCE_SCORE = {config.MIN_RELEVANCE_SCORE}")
    print_summary("Dataset questions", evaluate(dataset_cases, rag))
    paraphrase_results = evaluate(paraphrase_cases, rag)
    print_summary("Paraphrased questions", paraphrase_results)
    print_details(paraphrase_results)
    evaluate_safety(rag)


if __name__ == "__main__":
    main()

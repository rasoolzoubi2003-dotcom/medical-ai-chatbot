"""Evaluate retrieval quality and routing on questions from medquad.csv.

Evaluation sets (fixed, so results stay comparable between versions):
  * dataset    - 200 questions sampled from medquad.csv (seed 42), in the dataset's wording
  * paraphrase - hand-written natural questions from eval_questions.csv
  * arabic     - Arabic translations of the same questions, from eval_questions_ar.csv
  * safety     - off-topic, prompt injection, greeting and emergency messages
                 from safety_questions.csv and safety_questions_ar.csv

Dev set (only used by --tune to choose MIN_RELEVANCE_SCORE, never for reported results):
  * dev_questions.csv and dev_questions_ar.csv (hand-written medical and off-topic questions)
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
  docker compose exec api python evaluate.py            # report results
  docker compose exec api python evaluate.py --tune     # choose the threshold on the dev set
  docker compose exec api python evaluate.py --answers  # generate answers: speed and language checks
"""

import argparse
import csv
import logging
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import ModuleType

DATASET_PATH = "medquad.csv"  # the evaluation always uses the bundled dataset
PARAPHRASE_PATH = "eval_questions.csv"
PARAPHRASE_AR_PATH = "eval_questions_ar.csv"
SAFETY_PATHS = {"English": "safety_questions.csv", "Arabic": "safety_questions_ar.csv"}
DEV_PATHS = {"English": "dev_questions.csv", "Arabic": "dev_questions_ar.csv"}
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
        docs_with_scores = rag.vector_db.similarity_search_with_relevance_scores(rag.retrieval_query(query), k=TOP_K)
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


def evaluate_safety(rag: ModuleType, name: str, path: str) -> None:
    """Check that off-topic and injection messages are refused, and greetings and emergencies are recognised."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        route, _ = rag.route_question(row["query"])
        by_label[row["label"]].append((row["query"], route))

    print(f"\n=== {name} safety questions ({len(rows)} questions) ===")
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

    For each language it measures (medical answered rate - off-topic accepted rate), and
    picks the threshold with the best average over languages, so the smaller Arabic set
    counts as much as the English one. If several thresholds tie, it takes the middle of
    that range for a safety margin.
    """
    dev: list[tuple[str, str, str]] = []  # (language, query, label)
    for language, path in DEV_PATHS.items():
        with open(path, encoding="utf-8") as f:
            dev += [(language, row["query"], row["label"]) for row in csv.DictReader(f)]
    candidates = sorted(set(answers_by_question) - eval_questions)
    dev += [("English", q, "medical") for q in random.Random(DEV_SEED).sample(candidates, DEV_SAMPLE_SIZE)]

    # A question is answered when it passes the text checks and its best passage scores >= threshold.
    scored = []
    for language, query, label in dev:
        search = rag.retrieval_query(query)  # English translation for Arabic questions, as in the app
        passes_checks = rag.precheck(query) is None and (
            search == query or rag.precheck(search) not in ("emergency", "injection")
        )
        top = rag.vector_db.similarity_search_with_relevance_scores(search, k=1)
        scored.append((language, label, passes_checks, top[0][1] if top else 0.0))

    def rates(threshold: float, language: str) -> tuple[float, float]:
        def share(label: str) -> float:
            hits = [ok and score >= threshold for lang, lab, ok, score in scored if lang == language and lab == label]
            return sum(hits) / len(hits)
        return share("medical"), share("off_topic")

    thresholds = [round(0.20 + 0.01 * i, 2) for i in range(76)]
    gains = {t: sum(a - b for a, b in (rates(t, lang) for lang in DEV_PATHS)) / len(DEV_PATHS) for t in thresholds}
    best_gain = max(gains.values())
    best = [t for t in thresholds if gains[t] == best_gain]
    chosen = best[len(best) // 2]

    counts = Counter((lang, label) for lang, label, _, _ in scored)
    print("\n=== Threshold sweep on dev set ===")
    print("  " + ", ".join(f"{lang} {label}: {n}" for (lang, label), n in sorted(counts.items())))
    print(f"{'threshold':>9}  {'EN answered':>11}  {'EN off-topic in':>15}  {'AR answered':>11}  {'AR off-topic in':>15}")
    for t in thresholds:
        if t in (round(x * 0.05, 2) for x in range(4, 20)) or t == chosen:
            (en_a, en_o), (ar_a, ar_o) = rates(t, "English"), rates(t, "Arabic")
            mark = "  <- chosen" if t == chosen else ""
            print(f"{t:9.2f}  {en_a:11.1%}  {en_o:15.1%}  {ar_a:11.1%}  {ar_o:15.1%}{mark}")
    print(f"\nBest range {best[0]:.2f}-{best[-1]:.2f}; recommended MIN_RELEVANCE_SCORE = {chosen:.2f}")


class _CountRewrites(logging.Handler):
    """Counts how often generate_answer had to ask the model to remove CJK characters."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "CJK" in record.getMessage():
            self.count += 1


def check_answers(rag: ModuleType, safety: ModuleType, questions: list[str]) -> None:
    """Generate real answers and check speed and that each answer is in the question's language."""
    logging.disable(logging.NOTSET)
    rewrites = _CountRewrites()
    logging.getLogger("medical_chatbot").addHandler(rewrites)
    arabic_letters, latin_letters = re.compile(r"[\u0621-\u064A]"), re.compile(r"[A-Za-z]")

    rows = []
    for question in questions:
        route, docs = rag.route_question(question)
        if route != "medical":
            rows.append((question, route, None, 0.0, False)); continue
        language = safety.detect_language(question)
        started = time.time()
        answer = rag.generate_answer(question, rag.build_context(rag.select_passages(docs)), language)
        seconds = time.time() - started
        text = re.sub(r"\([^)]*\)", "", answer)  # English terms in parentheses are allowed
        arabic, latin = len(arabic_letters.findall(text)), len(latin_letters.findall(text))
        share = arabic / max(arabic + latin, 1)
        rows.append((question, route, answer, seconds, share >= 0.9 if language == "ar" else share < 0.1))
        print(f"{seconds:5.1f}s  {'ok ' if rows[-1][4] else 'BAD'}  {question}  ->  {answer[:90]!r}", flush=True)

    answered = [r for r in rows if r[2] is not None]
    print(f"\n=== Answer checks ({len(answered)} answers) ===")
    print(f"Average time        {sum(r[3] for r in answered) / len(answered):.1f}s per answer")
    print(f"Right language      {sum(r[4] for r in answered)}/{len(answered)}")
    print(f"CJK rewrites needed {rewrites.count}/{len(answered)} (characters left after the guard: "
          f"{sum(len(rag.CJK_CHARACTERS.findall(r[2])) for r in answered)})")
    refused = [(q, route) for q, route, answer, _, _ in rows if answer is None]
    if refused:
        print(f"Not answered        {refused}")


def print_details(results: list[Result]) -> None:
    print(f"\n{'rank':>4}  {'score':>5}  {'route':<10}  query")
    for r in results:
        rank = str(r.rank) if r.rank else "-"
        print(f"{rank:>4}  {r.top_score:5.2f}  {r.route:<10}  {r.query}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tune", action="store_true", help="choose MIN_RELEVANCE_SCORE on the dev set instead of reporting results")
    parser.add_argument("--answers", action="store_true", help="generate answers for the hand-written questions and check speed and language")
    args = parser.parse_args()

    # Import the RAG module here so --help works without loading models.
    logging.disable(logging.WARNING)
    import config
    import rag
    import safety

    answers_by_question = load_answers_by_question(DATASET_PATH)

    # The evaluation sample is fixed (seed and size never change) so results stay comparable.
    sample = random.Random(EVAL_SEED).sample(sorted(answers_by_question), EVAL_SAMPLE_SIZE)
    with open(PARAPHRASE_PATH, encoding="utf-8") as f:
        paraphrase_rows = list(csv.DictReader(f))
    with open(PARAPHRASE_AR_PATH, encoding="utf-8") as f:
        arabic_rows = list(csv.DictReader(f))

    if args.answers:
        check_answers(rag, safety, [row["query"] for row in paraphrase_rows + arabic_rows])
        return

    if args.tune:
        held_out = set(sample) | {row["expected_question"] for row in paraphrase_rows + arabic_rows}
        tune_threshold(rag, answers_by_question, held_out)
        return

    dataset_cases = [(q, answers_by_question[q]) for q in sample]
    paraphrase_cases = [(row["query"], answers_by_question[row["expected_question"]]) for row in paraphrase_rows]
    arabic_cases = [(row["query"], answers_by_question[row["expected_question"]]) for row in arabic_rows]
    missing = [q for q, answers in paraphrase_cases + arabic_cases if not answers]
    if missing:
        raise SystemExit(f"Expected question not found in {DATASET_PATH} for: {missing}")

    print(f"MIN_RELEVANCE_SCORE = {config.MIN_RELEVANCE_SCORE}")
    print_summary("Dataset questions", evaluate(dataset_cases, rag))
    paraphrase_results = evaluate(paraphrase_cases, rag)
    print_summary("Paraphrased questions", paraphrase_results)
    print_details(paraphrase_results)
    arabic_results = evaluate(arabic_cases, rag)
    print_summary("Arabic paraphrased questions", arabic_results)
    print_details(arabic_results)
    for name, path in SAFETY_PATHS.items():
        evaluate_safety(rag, name, path)


if __name__ == "__main__":
    main()

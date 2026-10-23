"""Evaluate retrieval quality and routing on questions from medquad.csv.

Two question sets:
  * dataset    - questions sampled from medquad.csv, in the dataset's own wording
  * paraphrase - hand-written natural questions from eval_questions.csv,
                 closer to what a real user would type

A retrieved passage counts as correct when it contains at least MIN_SHARED_CHARS
consecutive characters of the expected answer. Matching on answer text (not on
row ids) means a passage holding only the question text does not count, and the
metric keeps working when the ingestion format changes.

Metrics:
  Hit@k     share of questions with a correct passage in the top k results
  MRR       mean of 1 / rank of the first correct passage (0 if none in top k)
  Routed    share of questions the router sends down the medical (RAG) path
  Answered  share that is routed as medical AND passes the relevance threshold,
            i.e. questions that actually reach the LLM

Usage (with the stack running):
  docker compose exec api python evaluate.py
  docker compose exec api python evaluate.py --sample-size 500 --seed 7
"""

import argparse
import csv
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

DATASET_PATH = "medquad.csv"
PARAPHRASE_PATH = "eval_questions.csv"
TOP_K = 5
MIN_SHARED_CHARS = 60


@dataclass
class Result:
    query: str
    rank: int | None  # 1-based rank of the first correct passage, None if not in top k
    top_score: float
    route: str
    answered: bool


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


def evaluate(cases: list[tuple[str, list[str]]], app) -> list[Result]:
    """Run retrieval and routing for each (query, expected answers) pair."""
    results = []
    for query, answers in cases:
        docs_with_scores = app.vector_db.similarity_search_with_relevance_scores(query, k=TOP_K)
        rank = next(
            (i for i, (doc, _) in enumerate(docs_with_scores, start=1) if shares_answer_text(doc.page_content, answers)),
            None,
        )
        top_score = docs_with_scores[0][1] if docs_with_scores else 0.0
        route = app.route_question(query)
        answered = route == "medical" and top_score >= app.MIN_RELEVANCE_SCORE
        results.append(Result(query, rank, top_score, route, answered))
    return results


def print_summary(name: str, results: list[Result]) -> None:
    n = len(results)

    def hit_at(k: int) -> float:
        return sum(1 for r in results if r.rank is not None and r.rank <= k) / n

    mrr = sum(1 / r.rank for r in results if r.rank is not None) / n
    routes = Counter(r.route for r in results)

    print(f"\n=== {name} ({n} questions) ===")
    print(f"Retrieval   Hit@1 {hit_at(1):6.1%}   Hit@3 {hit_at(3):6.1%}   Hit@5 {hit_at(5):6.1%}   MRR {mrr:.3f}")
    print(f"Pipeline    Routed as medical {routes['medical'] / n:6.1%}   Answered {sum(r.answered for r in results) / n:6.1%}")
    print("Routes      " + ", ".join(f"{route}={count}" for route, count in routes.most_common()))


def print_details(results: list[Result]) -> None:
    print(f"\n{'rank':>4}  {'score':>5}  {'route':<10}  query")
    for r in results:
        rank = str(r.rank) if r.rank else "-"
        print(f"{rank:>4}  {r.top_score:5.2f}  {r.route:<10}  {r.query}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample-size", type=int, default=200, help="dataset questions to sample (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the sample (default: 42)")
    args = parser.parse_args()

    # Import the app here so --help works without loading models.
    logging.disable(logging.WARNING)
    import main as app

    answers_by_question = load_answers_by_question(DATASET_PATH)

    questions = sorted(answers_by_question)
    sample = random.Random(args.seed).sample(questions, min(args.sample_size, len(questions)))
    dataset_cases = [(q, answers_by_question[q]) for q in sample]

    with open(PARAPHRASE_PATH, encoding="utf-8") as f:
        paraphrase_cases = [(row["query"], answers_by_question[row["expected_question"]]) for row in csv.DictReader(f)]
    missing = [q for q, answers in paraphrase_cases if not answers]
    if missing:
        raise SystemExit(f"Expected question not found in {DATASET_PATH} for: {missing}")

    print_summary("Dataset questions", evaluate(dataset_cases, app))
    paraphrase_results = evaluate(paraphrase_cases, app)
    print_summary("Paraphrased questions", paraphrase_results)
    print_details(paraphrase_results)


if __name__ == "__main__":
    main()

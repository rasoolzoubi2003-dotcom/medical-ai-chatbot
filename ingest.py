"""Build the ChromaDB vector store from medquad.csv.

Each CSV row is one question/answer pair. Long answers are split into chunks,
and every chunk repeats its question, so no passage ends up holding only a
question or only a fragment with no topic. Source and topic are kept as
metadata so answers can cite where their information came from.

When a build finishes, a marker file records the number of chunks and a
fingerprint of the dataset and settings. A store without a matching marker
(an interrupted build, or a changed dataset or settings) is rebuilt.

Usage:
  python ingest.py            # build the store (skipped if a complete, current one exists)
  python ingest.py --rebuild  # delete the store and build it again
"""

import argparse
import csv
import hashlib
import json
import os
import time
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATASET_PATH = "medquad.csv"
PERSIST_DIRECTORY = "./chromadb_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 reads at most 256 tokens and silently ignores the rest,
# so every chunk (question + answer part + 2 special tokens) must fit in that.
MAX_TOKENS = 256
ANSWER_TOKENS = 200   # target size of the answer part of a chunk
OVERLAP_TOKENS = 25
BATCH_SIZE = 1000

# Bump this when the chunk format changes, so existing stores get rebuilt.
FORMAT_VERSION = 2


def marker_path() -> str:
    return os.path.join(PERSIST_DIRECTORY, "ingest_complete.json")


@lru_cache(maxsize=1)
def fingerprint() -> str:
    """Hash of the dataset and the settings that shape the chunks."""
    digest = hashlib.sha256()
    with open(DATASET_PATH, "rb") as f:
        digest.update(f.read())
    settings = [FORMAT_VERSION, EMBEDDING_MODEL, MAX_TOKENS, ANSWER_TOKENS, OVERLAP_TOKENS]
    digest.update(json.dumps(settings).encode())
    return digest.hexdigest()


def read_marker() -> dict | None:
    try:
        with open(marker_path(), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_marker(chunks: int) -> None:
    # Write to a temp file first, so a crash can't leave a half-written marker.
    tmp_path = marker_path() + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "fingerprint": fingerprint()}, f)
    os.replace(tmp_path, marker_path())


def store_is_complete(store: Chroma) -> bool:
    """True if a build finished and the store still holds every chunk it wrote."""
    marker = read_marker()
    return marker is not None and marker.get("chunks") == len(store.get(include=[])["ids"])


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer, logging as transformers_logging

    # Measuring whole answers before splitting triggers a harmless "too long" warning.
    transformers_logging.set_verbosity_error()
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    """Tokens the embedding model sees, including its [CLS] and [SEP] tokens."""
    return len(tokenizer()(text)["input_ids"])


def load_documents(path: str) -> list[Document]:
    """Turn every CSV row into one or more chunks of 'Question + part of the answer'."""
    documents: list[Document] = []
    seen: set[tuple[str, str]] = set()
    skipped_empty = skipped_duplicate = 0

    with open(path, encoding="utf-8") as f:
        for row_number, row in enumerate(csv.DictReader(f)):
            question = " ".join(row["question"].split())
            answer = row["answer"].strip()
            if not answer:
                skipped_empty += 1
                continue
            if (question, answer) in seen:
                skipped_duplicate += 1
                continue
            seen.add((question, answer))

            # The question prefix uses part of the token budget, so the answer
            # part gets whatever is left (and at most ANSWER_TOKENS).
            prefix = f"Question: {question}\nAnswer: "
            budget = min(ANSWER_TOKENS, MAX_TOKENS - count_tokens(prefix))
            splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer(), chunk_size=budget, chunk_overlap=OVERLAP_TOKENS
            )

            for chunk_number, chunk in enumerate(splitter.split_text(answer)):
                documents.append(Document(
                    page_content=prefix + chunk,
                    metadata={
                        "row": row_number,
                        "chunk": chunk_number,
                        "question": question,
                        "source": row["source"],
                        "focus_area": row["focus_area"],
                    },
                    id=f"{row_number}-{chunk_number}",
                ))

    print(f"Skipped {skipped_empty} rows with empty answers and {skipped_duplicate} duplicate rows.")
    return documents


def run_ingestion(rebuild: bool = False) -> None:
    """Embed the dataset into ChromaDB, unless a complete, current store already exists."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

    marker = read_marker()
    is_current = marker is not None and marker.get("fingerprint") == fingerprint()
    if not rebuild and is_current and store_is_complete(store):
        print(f"Vector store is complete ({marker['chunks']} chunks), skipping ingestion (use --rebuild to recreate it).")
        return

    if store.get(limit=1)["ids"]:
        reason = "--rebuild requested" if rebuild else (
            "dataset or settings changed" if marker and not is_current
            else "no matching completion marker: the last build did not finish or used an older format"
        )
        print(f"Deleting existing vector store ({reason})...")
    # Remove the marker before touching the data, so an interrupted rebuild is detected.
    if os.path.exists(marker_path()):
        os.remove(marker_path())
    store.delete_collection()

    print(f"Loading {DATASET_PATH}...")
    documents = load_documents(DATASET_PATH)
    token_counts = [count_tokens(doc.page_content) for doc in documents]
    longest = max(token_counts)
    print(f"Created {len(documents)} chunks, longest is {longest} tokens (limit {MAX_TOKENS}).")
    if longest > MAX_TOKENS:
        raise SystemExit("A chunk is longer than the embedding model's limit; lower ANSWER_TOKENS.")

    # Cosine similarity: relevance scores are then plain cosine similarity
    # (1.0 = same meaning), which keeps thresholds easy to reason about.
    store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("Embedding and saving chunks (this takes a while on CPU)...")
    started = time.time()
    for start in range(0, len(documents), BATCH_SIZE):
        store.add_documents(documents[start:start + BATCH_SIZE])
        done = min(start + BATCH_SIZE, len(documents))
        print(f"  {done}/{len(documents)} chunks ({time.time() - started:.0f}s)", flush=True)

    write_marker(len(documents))
    print(f"Vector store ready at {PERSIST_DIRECTORY}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the ChromaDB vector store from medquad.csv")
    parser.add_argument("--rebuild", action="store_true", help="delete the existing store and build it again")
    run_ingestion(rebuild=parser.parse_args().rebuild)

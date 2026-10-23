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

import config

DATASET_PATH = config.DATASET_PATH
PERSIST_DIRECTORY = config.CHROMA_DIR
EMBEDDING_MODEL = config.EMBEDDING_MODEL

ANSWER_TOKENS = 200   # target size of the answer part of a chunk
OVERLAP_TOKENS = 25
BATCH_SIZE = 1000

# Bump this when the chunk format changes, so existing stores get rebuilt.
FORMAT_VERSION = 3

# Vector index settings. Cosine similarity keeps scores easy to reason about
# (1.0 = same meaning). E5 scores sit close together, so Chroma's default index
# (16 links per vector) missed the true best passage for about 1% of questions
# and one of the true top 3 for about 15%; this graph is denser and searched wider.
COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:M": 48,
    "hnsw:construction_ef": 400,
    "hnsw:search_ef": 400,
}


def embedding_prefixes(model_name: str) -> tuple[str, str]:
    """(query prefix, passage prefix) the model was trained with.

    E5 models expect "query: " before questions and "passage: " before documents;
    without them retrieval quality drops. Other models use no prefixes.
    """
    return ("query: ", "passage: ") if "e5" in model_name.lower() else ("", "")


QUERY_PREFIX, PASSAGE_PREFIX = embedding_prefixes(EMBEDDING_MODEL)


def make_embeddings() -> HuggingFaceEmbeddings:
    """The embedding model, shared by ingestion and the app so both embed text the same way."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"prompt": PASSAGE_PREFIX},
        query_encode_kwargs={"prompt": QUERY_PREFIX},
    )


@lru_cache(maxsize=1)
def max_tokens() -> int:
    """Longest input the embedding model reads; anything after it is silently ignored."""
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(EMBEDDING_MODEL, "sentence_bert_config.json"), encoding="utf-8") as f:
        return json.load(f)["max_seq_length"]


def marker_path() -> str:
    return os.path.join(PERSIST_DIRECTORY, "ingest_complete.json")


@lru_cache(maxsize=1)
def fingerprint() -> str:
    """Hash of the dataset and the settings that shape the chunks."""
    digest = hashlib.sha256()
    with open(DATASET_PATH, "rb") as f:
        digest.update(f.read())
    settings = [
        FORMAT_VERSION, EMBEDDING_MODEL, QUERY_PREFIX, PASSAGE_PREFIX, max_tokens(),
        ANSWER_TOKENS, OVERLAP_TOKENS, COLLECTION_METADATA,
    ]
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
    """Tokens the embedding model sees for a passage, including its prefix and special tokens."""
    return len(tokenizer()(PASSAGE_PREFIX + text)["input_ids"])


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
            budget = min(ANSWER_TOKENS, max_tokens() - count_tokens(prefix))
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
    embeddings = make_embeddings()
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
    print(f"Created {len(documents)} chunks, longest is {longest} tokens (limit {max_tokens()}).")
    if longest > max_tokens():
        raise SystemExit("A chunk is longer than the embedding model's limit; lower ANSWER_TOKENS.")

    store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_metadata=COLLECTION_METADATA,
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

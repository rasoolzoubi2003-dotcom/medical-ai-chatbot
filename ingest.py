"""Build the ChromaDB vector store from medquad.csv.

Each CSV row is one question/answer pair. Long answers are split into chunks,
and every chunk repeats its question, so no passage ends up holding only a
question or only a fragment with no topic. Source and topic are kept as
metadata so answers can cite where their information came from.

Usage:
  python ingest.py            # build the store (skipped if it already exists)
  python ingest.py --rebuild  # delete the store and build it again
"""

import argparse
import csv
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATASET_PATH = "medquad.csv"
PERSIST_DIRECTORY = "./chromadb_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 reads at most 256 tokens (~1,000 characters); anything
# longer is cut off. 800 characters plus the question stays under that.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 1000


def load_documents(path: str) -> list[Document]:
    """Turn every CSV row into one or more chunks of 'Question + part of the answer'."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
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

            for chunk_number, chunk in enumerate(splitter.split_text(answer)):
                documents.append(Document(
                    page_content=f"Question: {question}\nAnswer: {chunk}",
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
    """Embed the dataset into ChromaDB, unless a store already exists and rebuild is False."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

    # Re-running used to add every chunk a second time. Now we skip if the
    # store is already built, or start from an empty store with --rebuild.
    if store.get(limit=1)["ids"]:
        if not rebuild:
            print("Vector store already exists, skipping ingestion (use --rebuild to recreate it).")
            return
        print("Deleting existing vector store...")
        store.delete_collection()

    print(f"Loading {DATASET_PATH}...")
    documents = load_documents(DATASET_PATH)
    print(f"Created {len(documents)} chunks.")

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

    print(f"Vector store ready at {PERSIST_DIRECTORY}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the ChromaDB vector store from medquad.csv")
    parser.add_argument("--rebuild", action="store_true", help="delete the existing store and build it again")
    run_ingestion(rebuild=parser.parse_args().rebuild)

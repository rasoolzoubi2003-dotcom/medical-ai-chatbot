"""All settings in one place, read from environment variables (or a local .env file).

Every value has a default that works with `docker compose up`, except
SECRET_KEY, which must always be set. See .env.example for descriptions.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# Auth
SECRET_KEY: str | None = os.getenv("SECRET_KEY")
ACCESS_TOKEN_EXPIRE_MINUTES: int = _int("ACCESS_TOKEN_EXPIRE_MINUTES", 60)

# Storage
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./medical_chatbot.db")
CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chromadb_store")
DATASET_PATH: str = os.getenv("DATASET_PATH", "medquad.csv")

# Embeddings. Multilingual, so Arabic questions can be matched against the English data.
# Changing the model rebuilds the store on the next `python ingest.py`.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# LLM
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
LLM_TEMPERATURE: float = _float("LLM_TEMPERATURE", 0.1)
LLM_MAX_TOKENS: int = _int("LLM_MAX_TOKENS", 200)
LLM_TIMEOUT_SECONDS: float = _float("LLM_TIMEOUT_SECONDS", 120)

# Retrieval
RETRIEVAL_K: int = _int("RETRIEVAL_K", 3)
MAX_CONTEXT_CHARS: int = _int("MAX_CONTEXT_CHARS", 3600)
# Tuned on dev_questions.csv with `python evaluate.py --tune`; re-tune if you change the data or model.
MIN_RELEVANCE_SCORE: float = _float("MIN_RELEVANCE_SCORE", 0.85)

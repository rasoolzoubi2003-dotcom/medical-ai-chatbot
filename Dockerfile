FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch first: the default PyPI wheel bundles CUDA and adds ~7 GB.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0

COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as a non-root user. The data folders are created here so the named
# volumes mounted on them start out owned by this user.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/chromadb_store /app/data \
    && chown -R app:app /app
USER app

# Bake the embedding model into the image so containers start without internet.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY --chown=app:app . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

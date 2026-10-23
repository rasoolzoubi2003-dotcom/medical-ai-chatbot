"""FastAPI app: authentication, the RAG chat endpoint, chat history and health check."""

import logging

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("medical_chatbot")

import config  # noqa: E402  (logging is configured first so module start-up messages are shown)
import database
import models
import rag
import safety
import schemas
from auth import create_access_token, get_current_user, hash_password, unauthorized, verify_password
from database import get_db

models.Base.metadata.create_all(bind=database.engine)
database.add_missing_columns()

if not rag.knowledge_base_ready():
    logger.warning("Vector store is missing or incomplete. Run `python ingest.py` first.")

app = FastAPI(title="Medical RAG Chatbot")


@app.exception_handler(RequestValidationError)
def readable_validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    """Return the first validation problem as one plain sentence the UI can show as-is."""
    first = error.errors()[0]
    message = str(first.get("msg", "Invalid request.")).removeprefix("Value error, ")
    if not first.get("ctx", {}).get("error"):  # built-in errors such as "Field required"
        field = ".".join(str(part) for part in first.get("loc", []) if part != "body")
        message = f"{field}: {message}" if field else message
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": message})


def find_user(db: Session, username: str) -> models.User | None:
    """Look up a user by name, ignoring upper/lower case."""
    return db.query(models.User).filter(func.lower(models.User.username) == username.strip().lower()).first()


def save_and_return_chat(
    db: Session, user_id: int, question: str, answer: str, sources: list[dict] | None = None
) -> dict:
    """Store the exchange in the user's history and return it as the API response."""
    sources = sources or []
    db.add(models.ChatHistory(user_id=user_id, question=question, answer=answer, sources=sources))
    db.commit()
    return {"question": question, "answer": answer, "sources": sources}


@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(user_data: schemas.UserCreate, db: Session = Depends(get_db)) -> dict:
    """Create an account. Usernames are unique regardless of upper/lower case."""
    if find_user(db, user_data.username):
        raise HTTPException(status.HTTP_409_CONFLICT, "That username is already taken.")

    db.add(models.User(username=user_data.username, hashed_password=hash_password(user_data.password)))
    db.commit()
    logger.info("New user registered: %s", user_data.username)
    return {"message": "User created successfully"}


@app.post("/token", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> dict:
    """Exchange username and password for a bearer token. Wrong credentials return 401."""
    user = find_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise unauthorized("Incorrect username or password.")
    return {"access_token": create_access_token(user.username), "token_type": "bearer"}


@app.get("/me", response_model=schemas.UserInfo)
def me(current_user: models.User = Depends(get_current_user)) -> models.User:
    """The signed-in user. The UI calls this on page load to check a saved token."""
    return current_user


@app.post("/chat", response_model=schemas.ChatResponse)
def chat(
    request: schemas.QueryRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
) -> dict:
    """Answer a health question with RAG and return the source passages used.

    Returns 503 if the knowledge base or the LLM is not ready yet.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not rag.knowledge_base_ready():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The knowledge base is still being built. Please try again soon.")

    route, docs_with_scores = rag.route_question(question)

    if route == "emergency":
        logger.warning("Emergency detected for user_id=%s", current_user.id)
        return save_and_return_chat(db, current_user.id, question, safety.EMERGENCY_RESPONSE)
    if route == "injection":
        logger.warning("Injection attempt detected for user_id=%s: %r", current_user.id, question)
        return save_and_return_chat(db, current_user.id, question, safety.OFF_TOPIC_RESPONSE)
    if route == "greeting":
        return save_and_return_chat(db, current_user.id, question, safety.greeting_reply(question))
    if route == "off_topic":
        return save_and_return_chat(db, current_user.id, question, safety.OFF_TOPIC_RESPONSE)

    # Medical question: the same passages go into the prompt and back to the user as sources.
    passages = rag.select_passages(docs_with_scores)
    context = rag.build_context(passages)
    if not context:
        return save_and_return_chat(db, current_user.id, question, safety.OFF_TOPIC_RESPONSE)

    try:
        answer = rag.generate_answer(question, context)
    except rag.LLMUnavailableError:
        # Not saved to history: it is a temporary failure, not an answer.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The AI model is not available right now. Please try again in a moment.",
        )

    if safety.mentions_emergency_condition(question):
        answer += safety.EMERGENCY_NOTE
    answer += safety.MEDICAL_DISCLAIMER

    sources = [rag.to_source(doc, score, text) for doc, score, text in passages]
    return save_and_return_chat(db, current_user.id, question, answer, sources)


@app.get("/history", response_model=list[schemas.ChatHistoryResponse])
def get_history(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)) -> list:
    """The current user's past questions and answers, newest first."""
    return (
        db.query(models.ChatHistory)
        .filter(models.ChatHistory.user_id == current_user.id)
        .order_by(models.ChatHistory.timestamp.desc())
        .all()
    )


@app.get("/health")
def health() -> JSONResponse:
    """Readiness check used by Docker: 200 when everything is ready, 503 otherwise."""
    checks = {
        "vector_store": rag.knowledge_base_ready(),
        "ollama_model": rag.ollama_model_ready(),
    }
    healthy = all(checks.values())
    return JSONResponse(
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "ok" if healthy else "unavailable", "model": config.OLLAMA_MODEL, "checks": checks},
    )


@app.get("/", response_class=HTMLResponse)
def serve_html() -> str:
    """The single-page web UI."""
    with open("index.html", encoding="utf-8") as f:
        return f.read()

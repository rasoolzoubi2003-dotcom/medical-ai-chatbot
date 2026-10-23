"""Request and response models for the API."""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator


class UserCreate(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class QueryRequest(BaseModel):
    question: str


class SourcePassage(BaseModel):
    """One passage an answer was based on."""

    question: str   # the dataset question this passage answers
    source: str     # organisation that published it, e.g. "National Cancer Institute"
    topic: str      # dataset focus area, e.g. "Glaucoma"
    excerpt: str    # the passage text the model was given
    score: float    # cosine similarity to the user's question (0-1)


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourcePassage] = []


class ChatHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    question: str
    answer: str
    sources: list[SourcePassage] = []
    timestamp: datetime

    @field_validator("sources", mode="before")
    @classmethod
    def empty_if_missing(cls, value: list | None) -> list:
        # Rows saved before sources existed have NULL here.
        return value or []

    @field_serializer("timestamp")
    def as_utc(self, value: datetime) -> str:
        # SQLite drops the timezone; mark it as UTC so browsers convert it to local time.
        return value.replace(tzinfo=value.tzinfo or timezone.utc).isoformat()

"""Request and response models for the API."""

import re
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,30}$")
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72  # bcrypt only uses the first 72 bytes of a password


class UserCreate(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def check_username(cls, value: str) -> str:
        value = value.strip()
        if not USERNAME_PATTERN.match(value):
            raise ValueError("Username must be 3-30 characters: letters, numbers and underscores only.")
        return value

    @field_validator("password")
    @classmethod
    def check_password(cls, value: str) -> str:
        if len(value) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
        if len(value.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise ValueError("Password is too long (maximum 72 bytes).")
        return value


class UserInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    username: str


class Token(BaseModel):
    access_token: str
    token_type: str


class QueryRequest(BaseModel):
    question: str = Field(max_length=2000)


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

from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import List, Optional

# 1. مخطط إنشاء حساب جديد (البيانات اللي رح يرسلها المستخدم)
class UserCreate(BaseModel):
    username: str
    password: str

# 2. مخطط إرجاع بيانات المستخدم (بدون كلمة السر لأغراض الأمان!)
class UserResponse(BaseModel):
    id: int
    username: str
    created_at: datetime

    class Config:
        from_attributes = True # لتسهيل تحويل بيانات SQLAlchemy إلى JSON تلقائياً

# 3. مخطط توكين تسجيل الدخول (JWT Token)
class Token(BaseModel):
    access_token: str
    token_type: str

# 4. مخطط استقبال السؤال الطبي
class QueryRequest(BaseModel):
    question: str

# 5. مصدر واحد من المصادر اللي انبنى عليها الجواب
class SourcePassage(BaseModel):
    question: str   # the dataset question this passage answers
    source: str     # organisation that published it, e.g. "National Cancer Institute"
    topic: str      # dataset focus area, e.g. "Glaucoma"
    excerpt: str    # the passage text the model was given
    score: float    # cosine similarity to the user's question (0-1)


# 6. مخطط رد الشات: الجواب ومعه مصادره
class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourcePassage] = []


# 7. مخطط إرجاع المحادثة من الـ SQL
class ChatHistoryResponse(BaseModel):
    id: int
    question: str
    answer: str
    sources: List[SourcePassage] = []
    timestamp: datetime

    @field_validator("sources", mode="before")
    @classmethod
    def empty_if_missing(cls, value):
        # Rows saved before sources existed have NULL here.
        return value or []

    class Config:
        from_attributes = True
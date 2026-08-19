from pydantic import BaseModel
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

# 5. مخطط إرجاع المحادثة من الـ SQL
class ChatHistoryResponse(BaseModel):
    id: int
    question: str
    answer: str
    timestamp: datetime

    class Config:
        from_attributes = True
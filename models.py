from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base  # استوردنا الـ Base اللي عملناه بالملف الأول

# 1. جدول المستخدمين
class User(Base):
    __tablename__ = "users"  # اسم الجدول بالـ Database

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False) # اسم مستخدم فريد
    hashed_password = Column(String, nullable=False) # كلمة السر بعد التشفير
    created_at = Column(DateTime, default=datetime.utcnow)

    # علاقة بربط المستخدم مع سجل محادثاته
    history = relationship("ChatHistory", back_populates="owner")

# 2. جدول سجل المحادثات الطبية
class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id")) # مفتاح أجنبي يربط المحادثة بـ id المستخدم
    question = Column(Text, nullable=False) # السؤال الطبي
    answer = Column(Text, nullable=False)   # الإجابة من الـ ChromaDB
    timestamp = Column(DateTime, default=datetime.utcnow)

    # علاقة عكسية ترجع للمستخدم صاحب المحادثة
    owner = relationship("User", back_populates="history")
import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. تحديد مكان ملف قاعدة البيانات (سيتم إنشاؤه تلقائياً باسم medical_chatbot.db)
# Docker sets DATABASE_URL to a file inside a named volume so data survives restarts.
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medical_chatbot.db")

# 2. إنشاء المحرك (Engine) الذي يتصل بقاعدة البيانات
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False} # خاص بـ SQLite ليتعامل مع طلبات FastAPI المتعددة
)

# 3. إنشاء جلسة (Session) نستخدمها لاحقاً للإضافة أو الاستعلام من الداتا بيز
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. الـ Base الذي ستورث منه جميع الجداول لاحقاً
Base = declarative_base()

def add_missing_columns() -> None:
    """Add columns introduced after a database was created.

    create_all() only creates missing tables, not missing columns, so databases
    from earlier versions need chat_history.sources added once.
    """
    columns = {column["name"] for column in inspect(engine).get_columns("chat_history")}
    if "sources" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE chat_history ADD COLUMN sources JSON"))


# 5. دالة تجلب الجلسة وتغلقها تلقائياً بعد كل طلب (Dependency Injection)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
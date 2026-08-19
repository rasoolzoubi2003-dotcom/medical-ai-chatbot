from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. تحديد مكان ملف قاعدة البيانات (سيتم إنشاؤه تلقائياً باسم medical_chatbot.db)
SQLALCHEMY_DATABASE_URL = "sqlite:///./medical_chatbot.db"

# 2. إنشاء المحرك (Engine) الذي يتصل بقاعدة البيانات
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False} # خاص بـ SQLite ليتعامل مع طلبات FastAPI المتعددة
)

# 3. إنشاء جلسة (Session) نستخدمها لاحقاً للإضافة أو الاستعلام من الداتا بيز
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. الـ Base الذي ستورث منه جميع الجداول لاحقاً
Base = declarative_base()

# 5. دالة تجلب الجلسة وتغلقها تلقائياً بعد كل طلب (Dependency Injection)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
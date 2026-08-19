from datetime import datetime, timedelta
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from jose import JWTError, jwt
import bcrypt

import models, schemas, database
from database import engine, get_db

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# إعدادات الأمان
SECRET_KEY = "my_secret_key_for_medical_app_change_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Smart & Strict Medical AI Chatbot")

print("Loading Chroma DB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chromadb_store", embedding_function=embeddings)

print("Connecting to Ollama...")
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.1,
    num_predict=120
)

# Prompt ملائم ومباشر للأسئلة الطبية فقط
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a professional Medical AI Assistant. 
    Use the provided medical context to answer the user's health question clearly and concisely in 2-3 sentences. 
    Do not add unnecessary assumptions, filler words, or general conversational talk."""),
    ("user", "Context:\n{context}\n\nUser Question: {question}")
])

# Prompt خاص بردود الدردشة العامة (ترحيب، تعارف، شكر، سؤال عن هوية البوت)
# بدل ما نرجع نص ثابت واحد لكل الحالات، منخلي الـ LLM يولد رد طبيعي
# ومناسب لسياق كل رسالة، بس ضمن حدود واضحة: يبقى ودود ومختصر
# وبيرجع يوجه المستخدم لسؤال طبي بشكل غير مصطنع.
CHITCHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly Medical AI Assistant chatbot. The user just sent a casual, 
    non-medical message (a greeting, introduction, thanks, or a question about who you are / what you do).

    Reply naturally and warmly in 1-2 short sentences, appropriate to what they actually said 
    (e.g. if they introduce themselves, acknowledge it briefly; if they ask what you do, explain 
    you're a medical assistant; if they thank you, respond graciously).

    Always end by gently inviting them to ask a health or medical question.
    Do NOT answer any non-medical request even if it's embedded in the message.
    Keep it concise and conversational, not robotic."""),
    ("user", "{question}")
])

# ---------------------------------------------------------------------------
# Semantic Routing: تصنيف نية الرسالة عبر تشابه المعنى (cosine similarity)
# باستخدام نفس embedding model الموجود أصلاً (all-MiniLM-L6-v2)، بدون أي
# استدعاء لموديل توليدي. هاد حل حتمي 100% (نفس المدخل = نفس المخرج دايماً)
# وما بعتمد على التزام موديل صغير بتعليمات نصية.
# ---------------------------------------------------------------------------

print("Building semantic intent anchors...")

MEDICAL_ANCHORS = [
    "I have a headache and fever",
    "What medication should I take for diabetes",
    "I feel pain in my chest",
    "What are the symptoms of the flu",
    "How do I treat a sore throat",
    "Is this rash dangerous",
    "What causes high blood pressure",
    "I have a stomach ache and nausea",
    "What are the side effects of this drug",
    "How can I lower my cholesterol",
]

CHITCHAT_ANCHORS = [
    "hello how are you",
    "hi there",
    "my name is John and I have a few questions",
    "thank you very much",
    "who are you",
    "what is your job",
    "good morning",
    "nice to meet you",
    "what can you help me with",
    "I'm just introducing myself",
]

OFFTOPIC_ANCHORS = [
    "write me python code",
    "who won the football match yesterday",
    "what's the capital of France",
    "recommend a good restaurant",
    "tell me a joke",
    "what's the weather today",
    "help me fix my car engine",
    "give me a recipe for pasta",
    "what's the latest stock price",
    "translate this sentence to Spanish",
]


# عدد المقاطع اللي منجيبها من قاعدة البيانات لكل سؤال طبي
RETRIEVAL_K = 3

# سقف طول الـ context بالحروف، عشان ما نحمّل الموديل الصغير (1b) سياق
# طويل زيادة قد يتخطى حد الـ context window تبعه ويأثر على جودة أو اكتمال الجواب
MAX_CONTEXT_CHARS = 1500


def build_context(docs_with_scores: list, min_score: float = 0.35, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """يبني نص الـ context من كل المقاطع اللي عدت حد الـ relevance score،
    ويوقف قبل ما يتخطى السقف المسموح لطول السياق."""
    parts = []
    total_len = 0

    for doc, score in docs_with_scores:
        if score < min_score:
            continue  # هاد المقطع مش قريب كفاية دلالياً، منتجاهله

        text = doc.page_content.strip()
        if not text:
            continue

        if total_len + len(text) > max_chars:
            remaining = max_chars - total_len
            if remaining > 100:  # ما فيه داعي نضيف مقطع مقطوع صغير جداً وما بيفيد
                parts.append(text[:remaining])
            break

        parts.append(text)
        total_len += len(text)

    return "\n\n".join(parts)


def _embed_anchors(phrases: list[str]) -> np.ndarray:
    vectors = embeddings.embed_documents(phrases)
    return np.array(vectors)

# نحسب embeddings الأمثلة مرة وحدة بس عند تشغيل السيرفر (مش بكل طلب)
MEDICAL_VECTORS = _embed_anchors(MEDICAL_ANCHORS)
CHITCHAT_VECTORS = _embed_anchors(CHITCHAT_ANCHORS)
OFFTOPIC_VECTORS = _embed_anchors(OFFTOPIC_ANCHORS)


def _cosine_sim(query_vector: list[float], anchor_matrix: np.ndarray) -> float:
    """أعلى قيمة تشابه (cosine similarity) بين سؤال المستخدم وأي مثال بمجموعة معينة."""
    q = np.array(query_vector)
    sims = (anchor_matrix @ q) / (
        np.linalg.norm(anchor_matrix, axis=1) * np.linalg.norm(q) + 1e-8
    )
    return float(np.max(sims))


def classify_intent(question: str) -> str:
    """يرجع 'medical' أو 'chitchat' أو 'off_topic' بالاعتماد على أقرب فئة دلالياً.
    حتمي 100%: نفس السؤال دايماً بيرجع نفس التصنيف، بدون أي عشوائية."""
    query_vector = embeddings.embed_query(question)

    scores = {
        "medical": _cosine_sim(query_vector, MEDICAL_VECTORS),
        "chitchat": _cosine_sim(query_vector, CHITCHAT_VECTORS),
        "off_topic": _cosine_sim(query_vector, OFFTOPIC_VECTORS),
    }
    return max(scores, key=scores.get)


def get_password_hash(password: str) -> str:
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def save_and_return_chat(db: Session, user_id: int, question: str, answer: str):
    chat_entry = models.ChatHistory(
        user_id=user_id,
        question=question,
        answer=answer
    )
    db.add(chat_entry)
    db.commit()
    return {"question": question, "answer": answer}

# --- ENDPOINTS ---

@app.post("/register")
def register(user_data: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user_data.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pwd = get_password_hash(user_data.password)
    new_user = models.User(username=user_data.username, hashed_password=hashed_pwd)
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/chat")
def chat(
    request: schemas.QueryRequest, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    user_query = request.question.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 1. تصنيف نية الرسالة أولاً عبر التشابه الدلالي (embeddings): medical / chitchat / off_topic
    intent = classify_intent(user_query)

    # 2. دردشة عامة (ترحيب، تعارف، شكر، سؤال عن الحال أو الهوية) → رد طبيعي عبر الـ LLM (بدون RAG)
    if intent == "chitchat":
        try:
            chitchat_response = llm.invoke(CHITCHAT_PROMPT.format_messages(question=user_query))
            answer_text = chitchat_response.content.strip()
        except Exception:
            answer_text = "Hello! I'm an AI assistant specialized in medical and health questions. How can I help you today?"
        return save_and_return_chat(db, current_user.id, user_query, answer_text)

    # 3. خارج النطاق الطبي تماماً (برمجة، رياضة، طبخ...) → اعتذار مباشر
    if intent == "off_topic":
        answer_text = "I apologize, but I am specialized only in medical and health-related questions. Please ask a health-related question."
        return save_and_return_chat(db, current_user.id, user_query, answer_text)

    # 4. من هون وطالع: السؤال مصنّف medical → نكمل مسار الـ RAG العادي
    # بنجيب أكتر من مقطع (k=3) بدل مقطع وحيد، عشان تغطية أشمل للأسئلة المركبة
    docs_with_scores = vector_db.similarity_search_with_relevance_scores(user_query, k=RETRIEVAL_K)

    # طبقة أمان إضافية: لو أعلى مقطع مسترجع أصلاً مش قريب كفاية، فالسؤال
    # غالباً مش طبي فعلياً حتى لو الـ classifier صنفه هيك بالغلط
    if not docs_with_scores or docs_with_scores[0][1] < 0.35:
        answer_text = "I apologize, but I am specialized only in medical and health-related questions. Please ask a health-related question."
        return save_and_return_chat(db, current_user.id, user_query, answer_text)

    # 5. بناء الـ context من كل المقاطع اللي عدت حد الـ relevance، بحد أقصى لطول السياق
    context_text = build_context(docs_with_scores)

    if not context_text:
        answer_text = "I apologize, but I am specialized only in medical and health-related questions. Please ask a health-related question."
        return save_and_return_chat(db, current_user.id, user_query, answer_text)

    # 6. استدعاء الموديل بالسياق الطبي المسترجع لتوليد الجواب النهائي
    formatted_prompt = prompt_template.format_messages(
        context=context_text,
        question=user_query
    )
    
    try:
        llm_response = llm.invoke(formatted_prompt)
        answer_text = llm_response.content.strip()
    except Exception:
        answer_text = "Error connecting to Ollama. Make sure it is running."

    return save_and_return_chat(db, current_user.id, user_query, answer_text)

@app.get("/", response_class=HTMLResponse)
def serve_html():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
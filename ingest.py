import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. تحديد مسارات الملفات والمجلدات
DATASET_PATH = "medquad.csv"  # غير اسم الملف حسب شو عندك (csv أو txt)
PERSIST_DIRECTORY = "./chromadb_store"

def run_ingestion():
    print("1. Checking dataset file...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Could not find '{DATASET_PATH}'. Make sure the file exists in your project folder.")
        return

    print("2. Loading dataset...")
    # تحميل البيانات حسب نوع الملف
    if DATASET_PATH.endswith(".csv"):
        loader = CSVLoader(file_path=DATASET_PATH, encoding="utf-8")
    else:
        loader = TextLoader(file_path=DATASET_PATH, encoding="utf-8")
        
    documents = loader.load()
    print(f"Loaded {len(documents)} raw records.")

    print("3. Splitting text into chunks...")
    # تقسيم النصوص لقطع صغيرة لتسهيل الاسترجاع الدقيق
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks.")

    print("4. Initializing Embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("5. Saving embeddings to ChromaDB...")
    # إنشاء وحفظ قاعدة البيانات المحلية
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print("Successfully created ChromaDB store at './chromadb_store'!")

if __name__ == "__main__":
    run_ingestion()
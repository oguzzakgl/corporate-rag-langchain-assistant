# faz4_langchain/proje_langchain_sifirdan/main_api.py

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

# Gemini API anahtarı kontrolü
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


# =====================================================================
# 🧩 PARÇA 1: FASTAPI SUNUCU VE PYDANTIC ŞEMALARI
# =====================================================================

from fastapi.responses import FileResponse

app = FastAPI(title="LangChain Corporate RAG Assistant API", version="1.0")

@app.get("/")
def home_page():
    return FileResponse("faz4_langchain/proje_langchain_sifirdan/index.html")


# API İstek ve Cevap Şemaları (Pydantic)
class ChatSoruRequest(BaseModel):
    soru: str

class ChatCevapResponse(BaseModel):
    soru: str
    cevap: str
    kaynaklar: list[str] = []

# Global RAG Referansımız (Yüklenen PDF ile güncellenecek)
GLOBAL_RAG_CHAIN = None

print("[OK] Parca 1 hazir: FastAPI sunucusu ve semalar tanimlandi.")



# =====================================================================
# 🧩 PARÇA 2: /upload-pdf ENDPOINT'İ (PDF İŞLEME VE RAG HAZIRLAMA)
# =====================================================================

@app.post("/upload-pdf")
def pdf_yukle_endpoint(file: UploadFile = File(...)):
    global GLOBAL_RAG_CHAIN
    
    # 1. Yüklenen PDF dosyasını geçici olarak kaydet
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. LangChain ile PDF Okuma ve Parçalama
        loader = PyPDFLoader(temp_path)
        sayfalar = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        parcalar = text_splitter.split_documents(sayfalar)
        
        # 3. LangChain ile ChromaDB Vektör Veritabanı Oluşturma
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
        vectorstore = Chroma.from_documents(parcalar, embeddings)
        
        # 4. RAG Sohbet Zincirini Hazırlayıp Global Değişkene Atama
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen resmi ve kurumsal bir AI Bilgi Asistanısın. 
Aşağıdaki kurumsal döküman bağlamını (context) kullanarak kullanıcının sorusuna derece derece liste maddeleri halinde, başlıklarla, son derece okunaklı ve profesyonel bir dille cevap ver.

Kurumsal Bağlam:
{context}"""),
            ("user", "{input}")
        ])

        
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        GLOBAL_RAG_CHAIN = create_retrieval_chain(retriever, combine_docs_chain)
        
        return {
            "mesaj": "PDF dosyası başarıyla LangChain ile işlendi ve RAG hazır hale getirildi!",
            "toplam_parca": len(parcalar)
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================================
# 🧩 PARÇA 3: /chat ENDPOINT'İ VE SUNUCU BAŞLATICI
# =====================================================================

@app.post("/chat", response_model=ChatCevapResponse)
def sohbet_endpoint(request: ChatSoruRequest):
    global GLOBAL_RAG_CHAIN
    
    # PDF yüklenmemişse uyarı ver
    if not GLOBAL_RAG_CHAIN:
        raise HTTPException(
            status_code=400, 
            detail="Henüz hiçbir PDF yüklenmedi! Lütfen önce /upload-pdf ile bir PDF yükleyin."
        )
        
    # LangChain RAG Zincirini Tek Satırda Çalıştırma (.invoke)
    rag_sonuc = GLOBAL_RAG_CHAIN.invoke({"input": request.soru})
    
    # Kullanılan kaynak paragraflar
    kaynak_metinler = [doc.page_content for doc in rag_sonuc.get("context", [])]
    
    return ChatCevapResponse(
        soru=request.soru,
        cevap=rag_sonuc.get("answer", ""),
        kaynaklar=kaynak_metinler
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

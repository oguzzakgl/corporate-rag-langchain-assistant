# faz4_langchain/proje_langchain_sifirdan/main_api.py
"""
HİBRİT AI ASİSTAN BACKEND SUNUCUSU (FASTAPI + LANGCHAIN RAG & SQL & EPOSTA TOOL)
-----------------------------------------------------------------------------
Bu sunucu dosyası 3 dev sistemi orkestre eder:
1. PDF RAG Sohbet Sistemi
2. Neon PostgreSQL SQL Ajanı
3. Kurumsal E-Posta Gönderme ve Taslak Aracı (@tool)
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# ---------------------------------------------------------------------
# 🔌 HARİCİ MODÜLLERİMİZİ İTHAL EDİYORUZ
# ---------------------------------------------------------------------
from intent_router import niyet_analizi_yap
from sql_agent import sql_ajani_kur
from email_tool import eposta_gonder

load_dotenv()

# API Anahtarı Doğrulama
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]




# =====================================================================
# 🧩 PARÇA 1: FASTAPI SUNUCUSU VE PYDANTIC VERİ ŞEMALARI
# =====================================================================

app = FastAPI(title="LangChain Corporate RAG, SQL & Email Assistant API", version="1.0")

@app.get("/")
def home_page():
    return FileResponse("faz4_langchain/proje_langchain_sifirdan/index.html")


class ChatSoruRequest(BaseModel):
    soru: str

class ChatCevapResponse(BaseModel):
    niyet: str
    gerekce: str
    soru: str
    cevap: str
    kaynaklar: list[str] = []

GLOBAL_RAG_CHAIN = None


# =====================================================================
# 🧩 PARÇA 2: /upload-pdf ENDPOINT'İ (PDF İŞLEME VE RAG HAZIRLAMA)
# =====================================================================

@app.post("/upload-pdf")
def pdf_yukle_endpoint(file: UploadFile = File(...)):
    global GLOBAL_RAG_CHAIN
    
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        loader = PyPDFLoader(temp_path)
        sayfalar = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        parcalar = text_splitter.split_documents(sayfalar)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
        vectorstore = Chroma.from_documents(parcalar, embeddings)
        
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
# 🧩 PARÇA 3: /chat ENDPOINT'İ (3'LÜ NİYET YÖNLENDİRMESİ)
# =====================================================================

@app.post("/chat", response_model=ChatCevapResponse)
def sohbet_endpoint(request: ChatSoruRequest):
    global GLOBAL_RAG_CHAIN
    
    # 1. Niyet Router ile Niyet Analizi Yap
    karar = niyet_analizi_yap(request.soru)
    print(f"[Niyet Router] Niyet: {karar.niyet} | Gerekce: {karar.gerekce}")
    
    # 2. Niyete Göre 3 Farklı Yola Dallanma (Routing)
    if karar.niyet == "BELGE_ARAMA":
        if not GLOBAL_RAG_CHAIN:
            raise HTTPException(
                status_code=400, 
                detail="Henüz hiçbir PDF yüklenmedi! Lütfen önce /upload-pdf ile bir PDF yükleyin."
            )
            
        rag_sonuc = GLOBAL_RAG_CHAIN.invoke({"input": request.soru})
        kaynak_metinler = [doc.page_content for doc in rag_sonuc.get("context", [])]
        
        return ChatCevapResponse(
            niyet=karar.niyet,
            gerekce=karar.gerekce,
            soru=request.soru,
            cevap=rag_sonuc.get("answer", ""),
            kaynaklar=kaynak_metinler
        )
        
    elif karar.niyet == "EPOSTA_ISLEMI":
        # 📌 KURUMSAL E-POSTA ARAÇ (TOOL) TETİKLEME
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        llm_with_tools = llm.bind_tools([eposta_gonder])
        
        tool_res = llm_with_tools.invoke(request.soru)
        
        # Eğer model aracı çağırmak istediyse fonksiyonu tetikliyoruz
        if tool_res.tool_calls and len(tool_res.tool_calls) > 0:
            call = tool_res.tool_calls[0]
            # eposta_gonder.invoke() çağırıp sonucunu 'tool_cikti' değişkenine atadık
            tool_cikti = eposta_gonder.invoke(call["args"])
            cevap_metni = str(tool_cikti)
        else:
            raw_c = tool_res.content
            if isinstance(raw_c, list) and len(raw_c) > 0:
                cevap_metni = raw_c[0].get("text", str(raw_c))
            else:
                cevap_metni = str(raw_c)
            
        return ChatCevapResponse(
            niyet=karar.niyet,
            gerekce=karar.gerekce,
            soru=request.soru,
            cevap=cevap_metni,
            kaynaklar=["LangChain Custom @tool eposta_gonder() Fonksiyonu Tetiklendi"]
        )

        
    else:
        # VERITABANI_ANALIZ Niyeti -> LangChain SQL Agent Çalıştırılıyor!
        sql_agent = sql_ajani_kur()
        sql_sonuc = sql_agent.invoke({"input": request.soru})
        
        raw_output = sql_sonuc.get("output", "")
        if isinstance(raw_output, list) and len(raw_output) > 0:
            cevap_metni = raw_output[0].get("text", str(raw_output))
        else:
            cevap_metni = str(raw_output)
            
        return ChatCevapResponse(
            niyet=karar.niyet,
            gerekce=karar.gerekce,
            soru=request.soru,
            cevap=cevap_metni,
            kaynaklar=["PostgreSQL Veritabanı Otomatik İncendi ve SQL Sorgusu Çalıştırıldı"]
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

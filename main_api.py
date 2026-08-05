# faz4_langchain/proje_langchain_sifirdan/main_api.py
"""
HİBRİT AI ASİSTAN BACKEND SUNUCUSU (FASTAPI + LANGCHAIN RAG & SQL AGENT)
---------------------------------------------------------------------
Bu sunucu dosyası tüm modülleri orkestre eder. Kullanıcıdan gelen soruları
önce Niyet Router ile analiz eder, ardından PDF RAG Zincirine veya SQL Ajanına yönlendirir.
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
# 1. intent_router.py dosyasından 'niyet_analizi_yap' fonksiyonunu çağırmak üzere içe aktarıyoruz.
from intent_router import niyet_analizi_yap

# 2. sql_agent.py dosyasından 'sql_ajani_kur' fonksiyonunu çağırmak üzere içe aktarıyoruz.
from sql_agent import sql_ajani_kur

load_dotenv()

# API Anahtarı Doğrulama
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]



# =====================================================================
# 🧩 PARÇA 1: FASTAPI SUNUCUSU VE PYDANTIC VERİ ŞEMALARI
# =====================================================================

app = FastAPI(title="LangChain Corporate RAG & SQL Assistant API", version="1.0")

@app.get("/")
def home_page():
    """Web arayüzümüz olan index.html dosyasını tarayıcıya döndürür."""
    return FileResponse("faz4_langchain/proje_langchain_sifirdan/index.html")


class ChatSoruRequest(BaseModel):
    soru: str

class ChatCevapResponse(BaseModel):
    niyet: str
    gerekce: str
    soru: str
    cevap: str
    kaynaklar: list[str] = []

# Global RAG Sohbet Zincirimiz (PDF yüklendiğinde bu değişkene atanacak)
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
        # 1. PyPDFLoader ile PDF dosyasını okuyup 'sayfalar' değişkenine atadık
        loader = PyPDFLoader(temp_path)
        sayfalar = loader.load()
        
        # 2. Metin bölücüyü çalıştırıp parçalanmış belgeleri 'parcalar' değişkenine atadık
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        parcalar = text_splitter.split_documents(sayfalar)
        
        # 3. Vektör gömme modelini oluşturup 'embeddings' değişkenine atadık
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
        
        # 4. ChromaDB veritabanını oluşturup 'vectorstore' değişkenine atadık
        vectorstore = Chroma.from_documents(parcalar, embeddings)
        
        # 5. Gemini yapay zeka modelini oluşturup 'llm' değişkenine atadık
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        
        # 6. Vektör veritabanından veri getiriciyi 'retriever' değişkenine atadık
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # 7. Prompt şablonunu oluşturup 'prompt' değişkenine atadık
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen resmi ve kurumsal bir AI Bilgi Asistanısın. 
Aşağıdaki kurumsal döküman bağlamını (context) kullanarak kullanıcının sorusuna derece derece liste maddeleri halinde, başlıklarla, son derece okunaklı ve profesyonel bir dille cevap ver.

Kurumsal Bağlam:
{context}"""),
            ("user", "{input}")
        ])
        
        # 8. Döküman birleştirme zincirini 'combine_docs_chain' değişkenine atadık
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        
        # 9. Tam RAG zincirini oluşturup global 'GLOBAL_RAG_CHAIN' değişkenine atadık
        GLOBAL_RAG_CHAIN = create_retrieval_chain(retriever, combine_docs_chain)
        
        return {
            "mesaj": "PDF dosyası başarıyla LangChain ile işlendi ve RAG hazır hale getirildi!",
            "toplam_parca": len(parcalar)
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =====================================================================
# 🧩 PARÇA 3: /chat ENDPOINT'İ (ROUTER -> RAG VEYA SQL AJANI YÖNLENDİRMESİ)
# =====================================================================

@app.post("/chat", response_model=ChatCevapResponse)
def sohbet_endpoint(request: ChatSoruRequest):
    global GLOBAL_RAG_CHAIN
    
    # 📌 NİYET ANALİZİ FONKSİYONU ÇAĞRILDI:
    # intent_router.py dosyasındaki niyet_analizi_yap() fonksiyonunu çağırdık 
    # ve dönen Pydantic NiyetKarari nesnesini 'karar' değişkenine atadık.
    karar = niyet_analizi_yap(request.soru)
    print(f"[Niyet Router] Niyet: {karar.niyet} | Gerekce: {karar.gerekce}")
    
    # 📌 NİYETE GÖRE DALLANMA (ROUTING):
    if karar.niyet == "BELGE_ARAMA":
        if not GLOBAL_RAG_CHAIN:
            raise HTTPException(
                status_code=400, 
                detail="Henüz hiçbir PDF yüklenmedi! Lütfen önce /upload-pdf ile bir PDF yükleyin."
            )
            
        # 📌 RAG ZİNCİRİ ÇAĞRILDI:
        # Hazırlanmış olan 'GLOBAL_RAG_CHAIN' zincirinin .invoke() metodunu çağırdık
        # ve üretilen cevabı 'rag_sonuc' değişkenine atadık.
        rag_sonuc = GLOBAL_RAG_CHAIN.invoke({"input": request.soru})
        
        # 📌 KAYNAKLAR ÇEKİLDİ:
        # 'rag_sonuc' sözlüğünün içindeki 'context' dökümanlarını gezerek metinleri 'kaynak_metinler' değişkenine atadık.
        kaynak_metinler = [doc.page_content for doc in rag_sonuc.get("context", [])]
        
        return ChatCevapResponse(
            niyet=karar.niyet,
            gerekce=karar.gerekce,
            soru=request.soru,
            cevap=rag_sonuc.get("answer", ""),
            kaynaklar=kaynak_metinler
        )
    else:
        # 📌 SQL AJANI KURULUM FONKSİYONU ÇAĞRILDI:
        # sql_agent.py dosyasındaki sql_ajani_kur() fonksiyonunu çağırdık 
        # ve oluşturulan SQL Ajan nesnesini 'sql_agent' değişkenine atadık.
        sql_agent = sql_ajani_kur()
        
        # 📌 SQL AJANI TETİKLENDİ:
        # 'sql_agent' nesnesinin .invoke() metodunu çağırdık 
        # ve veritabanı analiz sonucunu 'sql_sonuc' değişkenine atadık.
        sql_sonuc = sql_agent.invoke({"input": request.soru})
        
        # 📌 ÇIKTI METNİ AYIKLANDI:
        # 'sql_sonuc' içindeki "output" verisini temizleyip 'cevap_metni' değişkenine atadık.
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

# faz4_langchain/proje_langchain_sifirdan/main_api.py
"""
HİBRİT AI ASİSTAN BACKEND SUNUCUSU (FASTAPI + LANGCHAIN RAG & SQL & EPOSTA TOOL)
-----------------------------------------------------------------------------
Bu sunucu dosyası 3 dev sistemi orkestre eder:
1. PDF RAG Sohbet Sistemi (ChromaDB + create_retrieval_chain)
2. Neon PostgreSQL SQL Ajanı (sql_agent.py)
3. Kurumsal E-Posta Gönderme ve Taslak Aracı (email_tool.py + @tool)
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
# 1. intent_router.py dosyasındaki niyet_analizi_yap() fonksiyonunu çağırmak üzere ithal ettik
from intent_router import niyet_analizi_yap

# 2. sql_agent.py dosyasındaki sql_ajani_kur() fonksiyonunu çağırmak üzere ithal ettik
from sql_agent import sql_ajani_kur

# 3. email_tool.py dosyasındaki eposta_gonder() @tool fonksiyonunu çağırmak üzere ithal ettik
from email_tool import eposta_gonder

load_dotenv()

# API Anahtarı Doğrulama
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


# =====================================================================
# 🧩 PARÇA 1: FASTAPI SUNUCUSU VE PYDANTIC VERİ ŞEMALARI
# =====================================================================

# 1. FastAPI sınıfından sunucu nesnesi oluşturup 'app' değişkenine atadık
app = FastAPI(title="LangChain Corporate RAG, SQL & Email Assistant API", version="1.0")

@app.get("/")
def home_page():
    """Web arayüzümüz olan index.html dosyasını tarayıcıya yollar."""
    return FileResponse("faz4_langchain/proje_langchain_sifirdan/index.html")


# Kullanıcıdan gelen HTTP POST sohbet isteğinin Pydantic doğrulama şeması
class ChatSoruRequest(BaseModel):
    soru: str

# Kullanıcıya geri dönülecek yanıtın Pydantic doğrulama şeması
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
    
    # Yüklenen dosyayı geçici olarak kaydediyoruz
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. PyPDFLoader sınıfından yükleyici oluşturup .load() ile PDF sayfalarını 'sayfalar' değişkenine atadık
        loader = PyPDFLoader(temp_path)
        sayfalar = loader.load()
        
        # 2. RecursiveCharacterTextSplitter sınıfını çalıştırıp parçalanmış belgeleri 'parcalar' değişkenine atadık
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        parcalar = text_splitter.split_documents(sayfalar)
        
        # 3. GoogleGenerativeAIEmbeddings sınıfından gömme modelini oluşturup 'embeddings' değişkenine atadık
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
        
        # 4. Chroma.from_documents() metodunu çağırıp vektör veritabanını 'vectorstore' değişkenine atadık
        vectorstore = Chroma.from_documents(parcalar, embeddings)
        
        # 5. ChatGoogleGenerativeAI sınıfından Gemini modelini oluşturup 'llm' değişkenine atadık
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        
        # 6. 'vectorstore' nesnesinin .as_retriever() metodunu çağırıp veri getiriciyi 'retriever' değişkenine atadık
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # 7. ChatPromptTemplate.from_messages() metodunu çağırıp prompt şablonunu 'prompt' değişkenine atadık
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Sen resmi ve kurumsal bir AI Bilgi Asistanısın. 
Aşağıdaki kurumsal döküman bağlamını (context) kullanarak kullanıcının sorusuna derece derece liste maddeleri halinde, başlıklarla, son derece okunaklı ve profesyonel bir dille cevap ver.

Kurumsal Bağlam:
{context}"""),
            ("user", "{input}")
        ])
        
        # 8. create_stuff_documents_chain(llm, prompt) fonksiyonunu çağırıp döküman birleştirme zincirini 'combine_docs_chain' değişkenine atadık
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        
        # 9. create_retrieval_chain(retriever, combine_docs_chain) fonksiyonunu çağırıp tam RAG zincirini global 'GLOBAL_RAG_CHAIN' değişkenine atadık
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
    
    # 📌 1. NİYET ANALİZİ FONKSİYONU ÇAĞRILDI:
    # intent_router.py dosyasındaki niyet_analizi_yap() fonksiyonunu çağırdık 
    # ve dönen Pydantic NiyetKarari nesnesini 'karar' değişkenine atadık.
    karar = niyet_analizi_yap(request.soru)
    print(f"[Niyet Router] Niyet: {karar.niyet} | Gerekce: {karar.gerekce}")
    
    # 📌 2. NİYETE GÖRE 3 FARKLI YOLA DALLANMA (ROUTING):
    
    # =================================================================
    # 🟡 YOL A: BELGE_ARAMA Niyeti (PDF RAG Sohbet Sistemi)
    # =================================================================
    if karar.niyet == "BELGE_ARAMA":
        if not GLOBAL_RAG_CHAIN:
            raise HTTPException(
                status_code=400, 
                detail="Henüz hiçbir PDF yüklenmedi! Lütfen önce /upload-pdf ile bir PDF yükleyin."
            )
            
        # Hazırlanmış olan 'GLOBAL_RAG_CHAIN' zincirinin .invoke() metodunu çağırdık
        # ve üretilen cevabı 'rag_sonuc' değişkenine atadık.
        rag_sonuc = GLOBAL_RAG_CHAIN.invoke({"input": request.soru})
        
        # 'rag_sonuc' içindeki 'context' dökümanlarını gezerek metinleri 'kaynak_metinler' değişkenine atadık.
        kaynak_metinler = [doc.page_content for doc in rag_sonuc.get("context", [])]
        
        return ChatCevapResponse(
            niyet=karar.niyet,
            gerekce=karar.gerekce,
            soru=request.soru,
            cevap=rag_sonuc.get("answer", ""),
            kaynaklar=kaynak_metinler
        )
        
    # =================================================================
    # 🟢 YOL B: EPOSTA_ISLEMI Niyeti (Custom @tool eposta_gonder)
    # =================================================================
    elif karar.niyet == "EPOSTA_ISLEMI":
        # 1. ChatGoogleGenerativeAI sınıfından model nesnesi oluşturup 'llm' değişkenine atadık
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        
        # 2. email_tool.py dosyasından ithal ettiğimiz 'eposta_gonder' aracını llm.bind_tools() ile modele bağladık ve 'llm_with_tools' değişkenine atadık
        llm_with_tools = llm.bind_tools([eposta_gonder])
        
        # 3. 'llm_with_tools' nesnesinin .invoke() metodunu çağırdık ve yanıtı 'tool_res' değişkenine atadık
        tool_res = llm_with_tools.invoke(request.soru)
        
        # 4. Eğer model eposta_gonder aracını çağırmak istediyse fonksiyonu tetikliyoruz
        if tool_res.tool_calls and len(tool_res.tool_calls) > 0:
            call = tool_res.tool_calls[0]
            
            # email_tool.py dosyasındaki eposta_gonder() fonksiyonunu .invoke() ile çalıştırdık ve 'tool_cikti' değişkenine atadık
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
        
    # =================================================================
    # 🟣 YOL C: VERITABANI_ANALIZ Niyeti (Neon DB SQL Agent)
    # =================================================================
    else:
        # 1. sql_agent.py dosyasındaki sql_ajani_kur() fonksiyonunu çağırdık ve oluşturulan Ajan nesnesini 'sql_agent' değişkenine atadık
        sql_agent = sql_ajani_kur()
        
        # 2. 'sql_agent' nesnesinin .invoke() metodunu çağırdık ve veritabanı analiz sonucunu 'sql_sonuc' değişkenine atadık
        sql_sonuc = sql_agent.invoke({"input": request.soru})
        
        # 3. 'sql_sonuc' içindeki "output" verisini temizleyip 'cevap_metni' değişkenine atadık
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

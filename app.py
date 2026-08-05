# faz4_langchain/proje_langchain_sifirdan/app.py
"""
LANGCHAIN İLE SIFIRDAN CORPORATE AI DATA ASSISTANT (TEMİZ KOD)
"""

import os
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
# 🧩 ADIM 1: PDF OKUMA
# =====================================================================
loader = PyPDFLoader("sirket_mevzuati.pdf")
sayfalar = loader.load()
print(f"✅ ADIM 1 TAMAM: {len(sayfalar)} sayfa okundu.")


# =====================================================================
# 🧩 ADIM 2: METNİ PARÇALAMA (CHUNKİNG)
# =====================================================================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
parcalar = text_splitter.split_documents(sayfalar)
print(f"✅ ADIM 2 TAMAM: {len(parcalar)} parçaya bölündü.")


# =====================================================================
# 🧩 ADIM 3: VEKTÖRLEŞTİRME VE CHROMADB KAYDI
# =====================================================================
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
vectorstore = Chroma.from_documents(parcalar, embeddings, persist_directory="./chroma_db")
print("✅ ADIM 3 TAMAM: ChromaDB veritabanına indekslendi.")


# =====================================================================
# 🧩 ADIM 4: RAG SOHBET ZİNCİRİ VE SORGULAMA
# =====================================================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Kurumsal dökümanlara göre cevap ver:\n{context}"),
    ("user", "{input}")
])

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Canlı Soru
soru = "Yazilim ekibinin haftada kac gün uzaktan (remote) calisma hakki vardir?"
print(f"\n[SORU]: {soru}")

cevap = rag_chain.invoke({"input": soru})

print("\n--- LANGCHAIN SOHBET CEVABI ---")
print(cevap.get("answer"))
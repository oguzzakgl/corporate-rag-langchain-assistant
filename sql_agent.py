# faz4_langchain/proje_langchain_sifirdan/sql_agent.py
"""
LANGCHAIN SQL AJANI MODÜLÜ (TEXT-TO-SQL)
---------------------------------------
Neon PostgreSQL veritabanına bağlanır, tablo şemalarını inceler,
kullanıcı sorusundan SQL üretip çalıştırır ve sonucu özetler.
"""

import os
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# API Anahtarı Ayarı
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


# Neon PostgreSQL Veritabanı Adresi
DATABASE_URL = "postgresql+psycopg2://neondb_owner:npg_tIU6WbK5ysYA@ep-noisy-shadow-ayl4dhke-pooler.c-5.us-east-2.aws.neon.tech/neondb?sslmode=require"


def sql_ajani_kur():
    """SQLDatabase bağlantısını kurar ve LangChain SQL Ajanını döndürür."""
    
    # 1. SQLDatabase sınıfının .from_uri() metodunu çağırıp veritabanı bağlantısını 'db' değişkenine atadık
    db = SQLDatabase.from_uri(DATABASE_URL)
    
    # 2. ChatGoogleGenerativeAI sınıfından model nesnesi oluşturup 'llm' değişkenine atadık
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    
    # 3. create_sql_agent() fonksiyonunu çağırarak oluşturulan SQL Ajan nesnesini 'agent_executor' değişkenine atadık
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="tool-calling",
        verbose=True
    )
    
    # 4. Hazırlanan 'agent_executor' ajan nesnesini fonksiyon çıktısı olarak döndürdük
    return agent_executor


if __name__ == "__main__":
    print("[OK] LangChain SQL Ajani Test Ediliyor...")
    
    # 📌 SQL AJANI KURULUMU: sql_ajani_kur() fonksiyonunu çağırdık ve 'agent' değişkenine atadık
    agent = sql_ajani_kur()
    
    test_sorusu = "Veritabanindaki mevcut tablolari ve calisan sayilarini listele."
    
    # 📌 SQL AJANI ÇALIŞTIRMA: 'agent' nesnesinin .invoke() metodunu çağırdık ve çıktıyı 'sonuc' değişkenine atadık
    sonuc = agent.invoke({"input": test_sorusu})
    
    print("\n--- SQL AJANI CEVABI ---")
    print(sonuc.get("output"))

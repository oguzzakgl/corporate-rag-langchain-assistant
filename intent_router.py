# faz4_langchain/proje_langchain_sifirdan/intent_router.py
"""
LANGCHAIN İLE NİYET ANALİZİ MODÜLÜ (INTENT ROUTER)
------------------------------------------------
Bu dosya kullanıcı sorusunu inceler, Pydantic NiyetKarari şablonuna göre 
sorunun 'BELGE_ARAMA' mı yoksa 'VERITABANI_ANALIZ' mi olduğunu tespit eder.
"""

import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# API Anahtarı Kontrolü ve Ayarlanması
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]



# 📌 NİYET KARARI PYDANTIC ŞEMASI:
# Yapay zekanın döneceği çıktının tipini ve yapısını garanti altına alır.
class NiyetKarari(BaseModel):
    niyet: str = Field(description="Karar türü: 'BELGE_ARAMA' veya 'VERITABANI_ANALIZ'")
    gerekce: str = Field(description="Bu kararın alınma nedeni")


def niyet_analizi_yap(soru: str) -> NiyetKarari:
    """Kullanıcı sorusunu alır ve LangChain LCEL zinciri ile niyeti belirler."""
    
    # 1. ChatGoogleGenerativeAI sınıfından model nesnesi oluşturup 'llm' değişkenine atadık
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    
    # 2. PydanticOutputParser sınıfını NiyetKarari ile başlatıp 'parser' değişkenine atadık
    parser = PydanticOutputParser(pydantic_object=NiyetKarari)
    
    system_prompt = """
    Sen kurumsal bir AI yönlendirici asistanısın. Kullanıcı sorularını sınıflandırmakla görevlisin.
    
    Sınıflandırma Kuralları:
    1. 'VERITABANI_ANALIZ': Eğer soru şirket sayısal verileri, maaşlar, çalışan sayıları, istatistikler veya tablo verileri ile ilgiliyse bu niyeti seç.
    2. 'BELGE_ARAMA': Eğer soru şirket mevzuatları, izin politikası kuralları, dökümanlar veya nasıl yapılır rehberleri ile ilgiliyse bu niyeti seç.
    
    Uyman Gereken Çıktı Formatı:
    {format_instructions}
    """
    
    # 3. ChatPromptTemplate sınıfının .from_messages() metodunu çağırıp 'prompt' değişkenine atadık
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Kullanıcı Sorusu: {soru}")
    ])
    
    # 4. Pipe (|) operatörü ile (prompt -> llm -> parser) LCEL zincirini kurup 'router_chain' değişkenine atadık
    router_chain = prompt | llm | parser
    
    # 5. 'router_chain' nesnesinin .invoke() metodunu çağırıp sonucu 'karar' (NiyetKarari) değişkenine atadık
    karar: NiyetKarari = router_chain.invoke({
        "soru": soru,
        "format_instructions": parser.get_format_instructions()
    })
    
    # 6. Dönen 'karar' nesnesini ana fonksiyona geri döndürdük
    return karar


if __name__ == "__main__":
    test_soru = "Şirket çalışanlarının kıdem süresine göre yıllık izin hakları nedir?"
    print(f"❓ Soru: {test_soru}")
    
    # niyet_analizi_yap() fonksiyonunu çağırıp dönen sonucu 'sonuc' değişkenine atadık
    sonuc = niyet_analizi_yap(test_soru)
    print(f"🎯 Karar Niyeti: {sonuc.niyet} | Gerekçe: {sonuc.gerekce}")

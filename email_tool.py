# faz4_langchain/proje_langchain_sifirdan/email_tool.py
"""
KURUMSAL E-POSTA GÖNDERME VE TASLAK OLUŞTURMA ARACI
--------------------------------------------------
LangChain '@tool' decorator'ü kullanılarak yazılmış kurumsal e-posta aracı.
"""

from langchain_core.tools import tool

@tool
def eposta_gonder(alici_eposta: str, konu: str, mesaj_icerigi: str) -> str:
    """
    Kurumsal e-posta taslağı oluşturur ve belirtilen alıcı e-posta adresine gönderir.
    
    Parametreler:
    - alici_eposta: E-postanın gönderileceği adres (örn: 'ik@sirket.com', 'yonetici@sirket.com')
    - konu: E-postanın konu başlığı (örn: 'Yıllık İzin Talebi', 'Haftalık Rapor')
    - mesaj_icerigi: Gönderilecek resmi kurumsal e-posta metni
    """
    return f"✅ E-posta başarıyla gönderildi!\n📧 Alıcı: {alici_eposta}\n📌 Konu: {konu}\n📝 İçerik:\n{mesaj_icerigi}"

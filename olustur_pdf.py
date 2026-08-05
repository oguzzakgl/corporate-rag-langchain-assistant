# faz4_langchain/proje_langchain_sifirdan/olustur_pdf.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def pdf_olustur():
    pdf_yolu = "sirket_mevzuati.pdf"
    c = canvas.Canvas(pdf_yolu, pagesize=letter)
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "KURUMSAL SIRKET MEVZUATI VE IZIN POLITIKASI")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, 710, "1. YILLIK IZIN HAKKI:")
    c.drawString(100, 690, "   Sirket calisanlarinin 1 yildan 5 yila kadar kidemlerinde 14 gün yillik izin hakki vardir.")
    c.drawString(100, 670, "   5 yildan fazla kidemi olan calisanlar ise 20 gün yillik izin hakkina sahiptir.")
    
    c.drawString(100, 630, "2. MAZERET VE EVLILIK IZNI:")
    c.drawString(100, 610, "   Calisanlarin evlilik durumunda 3 gün, birinci derece yakin kaybinda 3 gün mazeret izni vardir.")
    
    c.drawString(100, 570, "3. UZAKTAN CALISMA POLITIKASI:")
    c.drawString(100, 550, "   Yazilim ekibi haftada 2 gün (Sali ve Persembe) evden (remote) calisma hakkina sahiptir.")
    
    c.drawString(100, 510, "4. MAAS VE PRIM ODEMELERI:")
    c.drawString(100, 490, "   Maaslar her ayin 1. günü banka hesaplarina yatirilir.")
    
    c.save()
    print(f"[OK] Test PDF dosyasi olusturuldu: {pdf_yolu}")


if __name__ == "__main__":
    pdf_olustur()

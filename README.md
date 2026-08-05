# 🚀 Corporate AI Data Assistant (LangChain & RAG Architecture)

> **LangChain, Gemini 2.5 Flash, ChromaDB ve FastAPI kullanılarak geliştirilmiş, PDF dökümanları üzerinden anlamsal arama ve soru-cevap yapan Kurumsal AI Asistanı.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)

---

## 🌟 Öne Çıkan Özellikler

- 📄 **Akıllı PDF İşleme:** `PyPDFLoader` ve `RecursiveCharacterTextSplitter` ile dökümanları anlam bütünlüğü bozulmadan küçük parçalara (chunks) ayırma.
- ⚡ **Vektör Veritabanı (ChromaDB):** Google `gemini-embedding-2` modeli ile üretilen vektörlerin yerel ChromaDB üzerinde hızlı indekslenmesi.
- 🤖 **Otomatik RAG Zinciri:** `create_retrieval_chain` ile soruya en alakalı döküman parçalarını çekip `Gemini 2.5 Flash` modeline resmi ve kurumsal dille cevaplatma.
- 🎨 **Modern Dark-Glass Web Arayüzü:** Sürükle-bırak PDF yükleme, canlı mesajlaşma ve kaynak döküman kartları gösterimi.
- 📌 **Markdown & HTML Render:** `marked.js` ile yapay zeka yanıtlarının başlık, liste ve kalın yazı biçiminde profesyonel sunumu.

---

## 🏗️ Mimari Yapı (Architecture)

```text
[Kullanıcı PDF Yükler] ➔ [PyPDFLoader] ➔ [RecursiveCharacterTextSplitter]
                                                     │
                                                     ▼
                                      [Google Gemini Embeddings]
                                                     │
                                                     ▼
                                            [ChromaDB Store]
                                                     │
[Kullanıcı Soru Sorar] ➔ [RetrievalChain] ➔ [Gemini 2.5 Flash] ➔ [Resmi Markdown Cevap]
```

---

## ⚙️ Kurulum ve Çalıştırma

### 1. Projeyi Klonlayın
```bash
git clone https://github.com/KULLANICI_ADI/corporate-rag-langchain-assistant.git
cd corporate-rag-langchain-assistant
```

### 2. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 3. API Anahtarını Tanımlayın
Kök dizinde `.env` dosyası oluşturup Gemini API anahtarınızı ekleyin:
```env
GEMINI_API_KEY=AIzaSy...senin_api_anahtarin...
```

### 4. Sunucuyu Başlatın
```bash
python main_api.py
```

Tarayıcınızdan **`http://127.0.0.1:8000`** adresine giderek web arayüzünü kullanmaya başlayabilirsiniz!

---

## 🎯 API Uç Noktaları (Endpoints)

- `GET /` ➔ Modern Dark-Glass Web Arayüzü
- `POST /upload-pdf` ➔ PDF dosyası yükleme ve ChromaDB'ye indeksleme
- `POST /chat` ➔ RAG Sohbet isteği gönderme ve yanıt alma

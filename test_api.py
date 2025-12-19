import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 1. Ayarları Yükle
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

print("⚡ Groq ile ışık hızında bağlantı deneniyor (Yeni Model)...")

if not api_key:
    print("❌ HATA: .env dosyasında GROQ_API_KEY bulunamadı!")
    exit()

try:
    # 2. Modeli Tanımla
    # GÜNCELLEME: Eski model yerine en yeni "Llama-3.3-70b" kullanıyoruz.
    # Bu model hem çok daha zeki hem de bedava.
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=api_key, 
        model_name="llama-3.3-70b-versatile" 
    )

    # 3. Soru Sor
    soru = "Yazılım öğrenen birine tek cümlelik Türkçe tavsiye ver."
    print(f"❓ Soru: {soru}")
    
    cevap = llm.invoke(soru)
    
    print("\n--- 🚀 Modelin Cevabı ---")
    print(cevap.content)
    print("\n✅ ZAFER! Bağlantı mükemmel çalışıyor.")

except Exception as e:
    print(f"\n❌ HATA: {e}")
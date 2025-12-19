import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# 1. Sayfa Ayarları
st.set_page_config(page_title="PDF ile Sohbet", page_icon="📚")
st.header("📚 PDF Dosyanla Sohbet Et")

# 2. Yan Menü (API Key Kontrolü için)
with st.sidebar:
    st.subheader("Ayarlar")
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if api_key:
        st.success("✅ API Anahtarı Aktif (Groq)")
    else:
        st.error("❌ API Anahtarı Bulunamadı! .env dosyanı kontrol et.")

# 3. Dosya Yükleme Alanı
pdf_dosyasi = st.file_uploader("PDF dosyanı buraya yükle", type="pdf")

# 4. Dosya Yüklendiyse İşlemleri Başlat
if pdf_dosyasi is not None:
    st.write("---")
    st.info("📄 Dosya yüklendi, içeriği okunuyor...")

    # PDF'i Oku
    pdf_okuyucu = PdfReader(pdf_dosyasi)
    metin = ""
    for sayfa in pdf_okuyucu.pages:
        metin += sayfa.extract_text()
        
    # Başarılı Mesajı
    st.success(f"Başarılı! Toplam {len(metin)} karakter okundu.")
    
    # Okunan metnin ilk 500 karakterini göster (Test amaçlı)
    with st.expander("PDF İçeriğinin Önizlemesini Gör"):
        st.write(metin[:1000] + "...")
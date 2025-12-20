import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import hashlib
from pathlib import Path

# LangChain
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ----------------- YARDIMCI FONKSİYON -----------------
def get_pdf_hash(pdf_file):
    pdf_bytes = pdf_file.getvalue()
    return hashlib.md5(pdf_bytes).hexdigest()


# ----------------- CACHE: EMBEDDINGS -----------------
@st.cache_resource
def load_embeddings():
    # Kalite için en stabil embedding modeli
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ----------------- SAYFA AYARLARI -----------------
st.set_page_config(page_title="PDF Asistanı", page_icon="🤖")
st.header("🤖 PDF Dosyanla Sohbet Et (Groq + Llama 3.3)")

# ----------------- API -----------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("❌ API Anahtarı Bulunamadı!")
    st.stop()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("📂 PDF Yükle")
    pdf_dosyasi = st.file_uploader("Dosyanı buraya bırak", type="pdf")
    st.write("---")
    st.write("Profil: Kalite > Hız")
    st.write("Model: Llama-3.3-70B")

# ----------------- ANA AKIŞ -----------------
if pdf_dosyasi is not None:
    # A) PDF OKUMA
    reader = PdfReader(pdf_dosyasi)
    metin = ""

    for sayfa in reader.pages:
        yazi = sayfa.extract_text()
        if yazi:
            metin += yazi

    if not metin.strip():
        st.error("⚠️ Bu PDF'ten metin okunamadı!")
        st.stop()

    st.success(f"✅ Dosya analiz edildi! ({len(metin)} karakter)")

    # B) METNİ PARÇALA
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(metin)

    # C) EMBEDDINGS (CACHE’Lİ)
    embeddings = load_embeddings()

    # ----------------- FAISS CACHE -----------------
    CACHE_DIR = Path("faiss_cache")
    CACHE_DIR.mkdir(exist_ok=True)

    pdf_hash = get_pdf_hash(pdf_dosyasi)
    faiss_path = CACHE_DIR / pdf_hash

    if faiss_path.exists():
        st.info("📦 Önceden işlenmiş PDF bulundu, cache kullanılıyor...")
        vectorstore = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        st.info("🧠 PDF ilk kez işleniyor, yüksek kaliteli embedding oluşturuluyor...")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local(faiss_path)

    # D) SORU
    st.write("---")
    soru = st.text_input(
        "Bu doküman hakkında ne bilmek istiyorsun?",
        placeholder="Örn: Beowulf'taki pagan ögeler nelerdir?"
    )

    if soru:
        with st.spinner("🧠 Derinlemesine analiz ediliyor..."):
            # 🔥 KALİTE ODAKLI LLM
            llm = ChatGroq(
                model_name="llama-3.3-70b-versatile",  # KALİTE
                groq_api_key=api_key,
                temperature=0.2,  # daha tutarlı cevaplar
                max_tokens=1024   # uzun ve detaylı cevaplar
            )

            # 🎯 KALİTE ODAKLI PROMPT
            prompt = ChatPromptTemplate.from_template("""
Aşağıdaki bağlamı dikkatlice analiz ederek soruyu cevapla.
Cevabını yalnızca verilen bağlama dayandır.
Bağlamda yoksa açıkça "Bu dokümanda buna dair bilgi yok." de.

Bağlam:
{context}

Soru:
{question}

Detaylı ve tutarlı bir cevap ver:
""")

            # 🔎 Daha fazla bağlam → kalite ↑
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )

            chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            cevap = chain.invoke(soru)

            st.markdown("### 🤖 Cevap:")
            st.write(cevap)

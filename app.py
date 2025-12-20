import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# LangChain - GÜNCEL ve DOĞRU IMPORTLAR
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# 1. Sayfa Ayarları
st.set_page_config(page_title="PDF Asistanı", page_icon="🤖")
st.header("🤖 PDF Dosyanla Sohbet Et (Groq + Llama 3.3)")

# 2. API Kontrolü
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("❌ API Anahtarı Bulunamadı! .env dosyanı kontrol et.")
    st.stop()

# 3. Yan Menü (Dosya Yükleme)
with st.sidebar:
    st.title("📂 PDF Yükle")
    pdf_dosyasi = st.file_uploader("Dosyanı buraya bırak", type="pdf")
    st.write("---")
    st.write("Model: Llama-3.3-70b (Groq)")

# 4. Ana Akış
if pdf_dosyasi is not None:
    # --- A) PDF OKUMA ---
    pdf_okuyucu = PdfReader(pdf_dosyasi)
    metin = ""
    for sayfa in pdf_okuyucu.pages:
        yazi = sayfa.extract_text()
        if yazi:
            metin += yazi

    if len(metin) == 0:
        st.error("⚠️ Bu PDF'ten metin okunamadı!")
    else:
        st.success(f"✅ Dosya analiz edildi! ({len(metin)} karakter)")

        # --- B) METNİ PARÇALA ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(metin)

        # --- C) EMBEDDING + FAISS ---
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # --- D) SORU ---
        st.write("---")
        soru = st.text_input(
            "Bu doküman hakkında ne bilmek istiyorsun?",
            placeholder="Örn: Beowulf'taki pagan ögeler nelerdir?"
        )

        if soru:
            with st.spinner("🧠 Düşünüyor..."):

                # LLM
                llm = ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    groq_api_key=api_key,
                    temperature=0.3
                )

                # PROMPT
                prompt = ChatPromptTemplate.from_template("""
Aşağıdaki bağlamı kullanarak soruyu cevapla.
Eğer cevap bağlamda yoksa "Bu dokümanda buna dair bilgi yok." de.

Bağlam:
{context}

Soru:
{question}
""")

                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

                # 🔥 MODERN RAG ZİNCİRİ
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

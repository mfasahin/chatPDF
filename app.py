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


# ----------------- SESSION STATE (SAFE INIT) -----------------
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("last_pdf_hash", None)


# ----------------- HELPERS -----------------
def get_pdf_hash(pdf_file):
    return hashlib.md5(pdf_file.getvalue()).hexdigest()


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ----------------- PAGE -----------------
st.set_page_config(page_title="PDF Asistanı", page_icon="🤖")
st.header("🤖 PDF Dosyanla Sohbet Et")

# ----------------- API -----------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ GROQ_API_KEY bulunamadı")
    st.stop()


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("📂 PDF Yükle")
    pdf_dosyasi = st.file_uploader("PDF seç", type="pdf")
    st.write("---")
    st.write("Model: **Llama-3.3-70B**")
    st.write("Profil: **Kalite > Hız**")


# ----------------- MAIN -----------------
if pdf_dosyasi:
    pdf_hash = get_pdf_hash(pdf_dosyasi)

    # PDF değiştiyse sohbeti sıfırla
    if pdf_hash != st.session_state["last_pdf_hash"]:
        st.session_state["chat_history"] = []
        st.session_state["last_pdf_hash"] = pdf_hash

    # PDF OKU
    reader = PdfReader(pdf_dosyasi)
    text = "".join(page.extract_text() or "" for page in reader.pages)

    if not text.strip():
        st.error("⚠️ PDF'ten metin okunamadı")
        st.stop()

    st.success(f"✅ PDF yüklendi ({len(text)} karakter)")

    # SPLIT
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    embeddings = load_embeddings()

    # FAISS CACHE
    cache_dir = Path("faiss_cache")
    cache_dir.mkdir(exist_ok=True)
    faiss_path = cache_dir / pdf_hash

    if faiss_path.exists():
        vectorstore = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local(faiss_path)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


    # ----------------- CHAT HISTORY UI -----------------
    for role, msg in st.session_state.get("chat_history", []):
        with st.chat_message(role):
            st.markdown(msg)


    # ----------------- INPUT -----------------
    soru = st.chat_input("PDF hakkında bir soru sor...")

    if soru:
        # user message
        st.session_state["chat_history"].append(("user", soru))
        with st.chat_message("user"):
            st.markdown(soru)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Düşünüyorum..."):
                llm = ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    groq_api_key=api_key,
                    temperature=0.2,
                    max_tokens=900
                )

                prompt = ChatPromptTemplate.from_template("""
Aşağıda bir PDF ve bu PDF hakkında yapılan sohbet vardır.
Sadece PDF bağlamına dayanarak cevap ver.

SOHBET:
{chat_history}

PDF BAĞLAMI:
{context}

SORU:
{question}

Kurallar:
- PDF'te yoksa: "Bu dokümanda buna dair bilgi yok." de
- Akademik, net ve tutarlı cevap ver
""")

                chain = (
                    {
                        "context": retriever,
                        "question": RunnablePassthrough(),
                        "chat_history": lambda _: "\n".join(
                            f"{r}: {m}"
                            for r, m in st.session_state.get("chat_history", [])[-6:]
                        )
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                cevap = chain.invoke(soru)
                st.markdown(cevap)

        st.session_state["chat_history"].append(("assistant", cevap))

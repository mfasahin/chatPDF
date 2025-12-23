import streamlit as st
from langchain_community.retrievers import BM25Retriever

from core.session import init_session, reset_chat_on_new_pdf
from services.pdf_loader import load_pdf_text
from services.embeddings import get_embeddings
from services.vectorstore import load_or_create_faiss
from services.retriever import hybrid_retrieve_factory
from services.llm import get_llm
from ui.chat import render_chat
from ui.sidebar import render_sidebar
from ui.loading import show_loading


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PDF Asistanı",
    page_icon="🤖",
    layout="centered"
)

st.header("🤖 PDF Dosyanla Sohbet Et")


# ---------------- SESSION ----------------
init_session()


# ---------------- SIDEBAR ----------------
pdf_file = render_sidebar()


# ---------------- MAIN FLOW ----------------
if pdf_file:
    loading = show_loading("PDF analiz ediliyor...")

    reset_chat_on_new_pdf(pdf_file)

    # 1️⃣ PDF → text
    text, chunks = load_pdf_text(pdf_file)

    # 2️⃣ Embeddings
    embeddings = get_embeddings()

    # 3️⃣ Vectorstore (FAISS)
    vectorstore = load_or_create_faiss(chunks, embeddings, pdf_file)

    # 4️⃣ LLM
    llm = get_llm()

    loading.empty()

    # 5️⃣ Retrievers
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = 4

    # 6️⃣ Hybrid retriever (factory)
    retriever = hybrid_retrieve_factory(
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever
    )

    # 7️⃣ UI
    render_chat(
        llm=llm,
        retriever=retriever
    )

else:
    st.info("📂 Soldan bir PDF yükleyerek başlayabilirsin")

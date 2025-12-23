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

# ---------------- CSS YÜKLEME ----------------

def load_css():
    # Buraya encoding="utf-8" ekliyoruz
    with open("styles/chat.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------- SESSION ----------------
init_session()

# ---------------- SIDEBAR ----------------
pdf_file = render_sidebar()

# ---------------- MAIN FLOW ----------------
if pdf_file:
    # 1. Kontrol: Bu dosya daha önce işlendi mi?
    # Eğer session_state içinde 'processed_file' yoksa veya dosya değişmişse analiz başlasın.
    if "processed_file" not in st.session_state or st.session_state.processed_file != pdf_file:
        
        # --- ANALİZ AŞAMASI (Sadece dosya değişince çalışır) ---
        
        # Sohbet geçmişini sıfırla (Dosya değiştiği için)
        reset_chat_on_new_pdf(pdf_file)
        
        loading_container = st.empty()
        with loading_container:
            show_loading("PDF analiz ediliyor...")

            # 1️⃣ PDF → text
            text, chunks = load_pdf_text(pdf_file)

            # 2️⃣ Embeddings
            embeddings = get_embeddings()

            # 3️⃣ Vectorstore (FAISS)
            vectorstore = load_or_create_faiss(chunks, embeddings, pdf_file)

            # 4️⃣ LLM
            llm = get_llm()

            # 5️⃣ Retrievers
            faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            bm25_retriever = BM25Retriever.from_texts(chunks)
            bm25_retriever.k = 4

            # 6️⃣ Hybrid retriever
            retriever = hybrid_retrieve_factory(
                faiss_retriever=faiss_retriever,
                bm25_retriever=bm25_retriever
            )
            
            # --- ÖNEMLİ: Sonuçları Session State'e Kaydet ---
            st.session_state.chain_llm = llm
            st.session_state.chain_retriever = retriever
            st.session_state.processed_file = pdf_file # Dosyanın işlendiğini işaretle

        # Yükleme ekranını temizle
        loading_container.empty()

    # --- SOHBET AŞAMASI (Her zaman çalışır ama yükleme yapmaz) ---
    # Artık analiz yapmıyoruz, state'ten hazır objeleri çekiyoruz
    if "chain_llm" in st.session_state and "chain_retriever" in st.session_state:
        render_chat(
            llm=st.session_state.chain_llm,
            retriever=st.session_state.chain_retriever
        )

else:
    # st.info yerine doğrudan bir chat mesajı gibi gösterelim
    st.session_state.messages = [] # Geçmişi temiz tutuyoruz
    
    with st.chat_message("assistant"):
        st.write("Merhaba! 👋 Ben PDF Asistanın.")
        st.write("Sohbete başlamak için lütfen sol menüden bir PDF dosyası yükle.")
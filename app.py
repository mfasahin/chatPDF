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


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PDF Asistanı",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"  # <-- KOMUTANIMIZ BU. BURASI "expanded" OLDUĞU İÇİN AÇIK GELECEK.
)

# ---------------- CSS YÜKLEME ----------------
def load_css():
    try:
        # Encoding hatası olmasın diye utf-8 ekli
        with open("styles/chat.css", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS dosyası bulunamadı: styles/chat.css")

load_css()


# ---------------- SESSION ----------------
init_session()


# ---------------- SIDEBAR (ENGELSİZ) ----------------
# BAK BURASI ÇOK ÖNEMLİ:
# Eskiden burada "if sidebar_open:" falan vardı. ONLARI SİLDİM ATTIM.
# Artık doğrudan render ediyoruz. Streamlit "expanded" ayarını görüp otomatik açacak.
pdf_file = render_sidebar()


# ---------------- MAIN FLOW ----------------
if pdf_file:
    # Dosya kimliği (ID) oluştur
    file_id = f"{pdf_file.name}_{pdf_file.size}"
    
    # Dosya değişti mi kontrol et
    needs_processing = (
        "processed_file_id" not in st.session_state or 
        st.session_state.processed_file_id != file_id
    )
    
    if needs_processing:
        # Yeni dosya yüklendiyse sohbeti sıfırla
        reset_chat_on_new_pdf(file_id)
        
        # Yükleme Ekranı (Animasyonlu)
        with st.container():
            st.markdown("""
            <div style="
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 16px;
                padding: 2rem;
                border-radius: 24px;
                background: rgba(102, 126, 234, 0.1);
                border: 1px solid rgba(102, 126, 234, 0.2);
                backdrop-filter: blur(10px);
                margin: 2rem 0;
            ">
                <div style="
                    width: 48px;
                    height: 48px;
                    border: 4px solid rgba(102, 126, 234, 0.2);
                    border-top: 4px solid #667eea;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                "></div>
                <div style="
                    color: #b8b8ff;
                    font-size: 1rem;
                    font-weight: 600;
                    letter-spacing: 0.5px;
                ">PDF Analiz Ediliyor...</div>
                <div style="
                    color: #9898dd;
                    font-size: 0.85rem;
                    opacity: 0.8;
                ">Bu birkaç saniye sürebilir</div>
            </div>
            <style>
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            </style>
            """, unsafe_allow_html=True)
        
        # PDF işleme
        try:
            text, chunks = load_pdf_text(pdf_file)
            embeddings = get_embeddings()
            vectorstore = load_or_create_faiss(chunks, embeddings, pdf_file)
            llm = get_llm()

            faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            bm25_retriever = BM25Retriever.from_texts(chunks)
            bm25_retriever.k = 4

            retriever = hybrid_retrieve_factory(
                faiss_retriever=faiss_retriever,
                bm25_retriever=bm25_retriever
            )
            
            # Sonuçları kaydet
            st.session_state.chain_llm = llm
            st.session_state.chain_retriever = retriever
            st.session_state.processed_file_id = file_id
            
            # Sayfayı yenile (Yükleme ekranını kaldırmak için)
            st.rerun()
            
        except Exception as e:
            st.error(f"PDF işlenirken hata oluştu: {str(e)}")
            

    # Sohbet render et (Sadece analiz bitmişse)
    if "chain_llm" in st.session_state and "chain_retriever" in st.session_state:
        render_chat(
            llm=st.session_state.chain_llm,
            retriever=st.session_state.chain_retriever
        )

else:
    # PDF yoksa sadece boş sohbet ekranı (Karşılama mesajı render_chat içinde olmalı)
    render_chat(llm=None, retriever=None)
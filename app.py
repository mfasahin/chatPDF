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
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ----------------- PAGE -----------------
st.set_page_config(page_title="PDF Asistanı", page_icon="🤖")

# CSS
st.markdown(
    "<style>" + open("styles/chat.css", encoding="utf-8").read() + "</style>",
    unsafe_allow_html=True
)

st.header("🤖 PDF Dosyanla Sohbet Et")


# ----------------- SESSION STATE -----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_pdf_hash" not in st.session_state:
    st.session_state.last_pdf_hash = None


# ----------------- HELPERS -----------------
def get_pdf_hash(pdf_file):
    return hashlib.md5(pdf_file.getvalue()).hexdigest()


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def question_profile(question: str):
    q = question.lower()
    wc = len(q.split())

    semantic = [
        "neden", "nasıl", "açıkla", "yorumla",
        "avantaj", "dezavantaj", "etki", "önem"
    ]

    hits = sum(1 for k in semantic if k in q)

    if wc <= 6 and hits == 0:
        return "keyword"
    elif hits >= 1 or wc >= 10:
        return "semantic"
    return "balanced"


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

    if pdf_hash != st.session_state.last_pdf_hash:
        st.session_state.chat_history = []
        st.session_state.last_pdf_hash = pdf_hash

    reader = PdfReader(pdf_dosyasi)
    text = "".join(p.extract_text() or "" for p in reader.pages)

    if not text.strip():
        st.error("⚠️ PDF'ten metin okunamadı")
        st.stop()

    st.success("✅ PDF yüklendi")

    # -------- SPLIT --------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    embeddings = load_embeddings()

    # -------- FAISS CACHE --------
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

    # -------- RETRIEVERS --------
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = 4

    # -------- HYBRID RETRIEVE --------
    def hybrid_retrieve(query: str, k=5):
        profile = question_profile(query)

        if profile == "keyword":
            w_bm25, w_faiss = 0.6, 0.4
        elif profile == "semantic":
            w_bm25, w_faiss = 0.3, 0.7
        else:
            w_bm25, w_faiss = 0.5, 0.5

        docs_faiss = faiss_retriever.invoke(query)
        docs_bm25 = bm25_retriever.invoke(query)

        scores = {}

        for i, d in enumerate(docs_faiss):
            scores[d.page_content] = scores.get(d.page_content, 0) + (4 - i) * w_faiss

        for i, d in enumerate(docs_bm25):
            scores[d.page_content] = scores.get(d.page_content, 0) + (4 - i) * w_bm25

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return "\n\n".join(t for t, _ in top)

    # -------- CHAT HISTORY --------
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # -------- CHAT INPUT --------
    soru = st.chat_input("PDF hakkında bir soru sor...")

    if soru:
        st.session_state.chat_history.append(("user", soru))
        with st.chat_message("user"):
            st.markdown(soru)

        chat_snapshot = st.session_state.chat_history[-6:]

        context_text = hybrid_retrieve(soru)

        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=api_key,
            temperature=0.2,
            max_tokens=700,
            streaming=True,
            timeout=60
        )

        prompt = ChatPromptTemplate.from_template("""
Sadece PDF içeriğine dayanarak cevap ver.

SOHBET:
{chat_history}

PDF BAĞLAMI:
{context}

SORU:
{question}
""")

        chain = (
            {
                "context": lambda _: context_text,
                "question": RunnablePassthrough(),
                "chat_history": lambda _: "\n".join(
                    f"{r}: {m}" for r, m in chat_snapshot
                )
            }
            | prompt
            | llm
        )

        # -------- ASSISTANT --------
        with st.chat_message("assistant"):
            thinking = st.markdown(
                """
                <div class="assistant-thinking">
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="thinking-text">Asistan düşünüyor</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            answer_box = st.empty()
            full_answer = ""
            cleared = False

            try:
                for chunk in chain.stream(soru):
                    if hasattr(chunk, "content") and chunk.content:
                        if not cleared:
                            thinking.empty()
                            cleared = True

                        full_answer += chunk.content
                        answer_box.markdown(full_answer)

            except Exception:
                thinking.empty()
                st.error("⚠️ Bağlantı hatası oluştu. Lütfen tekrar deneyin.")

        st.session_state.chat_history.append(("assistant", full_answer))

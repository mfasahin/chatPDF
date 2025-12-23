import hashlib
from pathlib import Path
from langchain_community.vectorstores import FAISS


def get_pdf_hash(uploaded_file) -> str:
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def load_or_create_faiss(text: str, embeddings, uploaded_file):
    cache_dir = Path("faiss_cache")
    cache_dir.mkdir(exist_ok=True)

    pdf_hash = get_pdf_hash(uploaded_file)
    path = cache_dir / pdf_hash

    if path.exists():
        return FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    vectorstore = FAISS.from_texts([text], embeddings)
    vectorstore.save_local(path)
    return vectorstore

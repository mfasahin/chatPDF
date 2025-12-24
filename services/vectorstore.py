import hashlib
from pathlib import Path
from langchain_community.vectorstores import FAISS


def get_pdf_hash(uploaded_file) -> str:
    """PDF dosyasının hash'ini döndürür"""
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def load_or_create_faiss(chunks: list, embeddings, uploaded_file):
    """
    FAISS vektör veritabanını yükler veya oluşturur.
    
    Args:
        chunks: Metin parçaları listesi (text değil!)
        embeddings: Embedding modeli
        uploaded_file: Yüklenmiş PDF dosyası
    
    Returns:
        FAISS vectorstore
    """
    cache_dir = Path("faiss_cache")
    cache_dir.mkdir(exist_ok=True)

    pdf_hash = get_pdf_hash(uploaded_file)
    path = cache_dir / pdf_hash

    # Cache varsa yükle
    if path.exists():
        return FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Yoksa oluştur - chunks kullan!
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local(str(path))
    
    return vectorstore
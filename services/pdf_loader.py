# services/pdf_loader.py

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_text(pdf_file):
    """
    PDF dosyasını okur ve:
    - full_text: tüm metin
    - chunks: retrieval için bölünmüş parçalar
    döndürür
    """

    reader = PdfReader(pdf_file)
    full_text = "".join(page.extract_text() or "" for page in reader.pages)

    if not full_text.strip():
        raise ValueError("PDF içeriği okunamadı")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(full_text)

    return full_text, chunks

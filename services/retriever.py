# services/retriever.py

from typing import Callable


def question_profile(question: str) -> str:
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


def hybrid_retrieve(
    query: str,
    faiss_retriever,
    bm25_retriever,
    k: int = 5
) -> str:
    """
    BM25 + FAISS skorlarını ağırlıklı birleştirir
    """

    profile = question_profile(query)

    weights = {
        "keyword": (0.6, 0.4),
        "semantic": (0.3, 0.7),
        "balanced": (0.5, 0.5)
    }

    w_bm25, w_faiss = weights[profile]
    scores = {}

    for i, d in enumerate(faiss_retriever.invoke(query)):
        scores[d.page_content] = scores.get(d.page_content, 0) + (4 - i) * w_faiss

    for i, d in enumerate(bm25_retriever.invoke(query)):
        scores[d.page_content] = scores.get(d.page_content, 0) + (4 - i) * w_bm25

    return "\n\n".join(
        t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    )


def hybrid_retrieve_factory(
    faiss_retriever,
    bm25_retriever
) -> Callable[[str], str]:
    """
    UI katmanı için sade bir retriever üretir
    """

    def retrieve(query: str) -> str:
        return hybrid_retrieve(
            query=query,
            faiss_retriever=faiss_retriever,
            bm25_retriever=bm25_retriever
        )

    return retrieve

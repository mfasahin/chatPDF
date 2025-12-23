# Groq / model

from langchain_groq import ChatGroq
from core.config import GROQ_API_KEY


def get_llm(streaming=True):
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=800,
        streaming=streaming,
        timeout=60
    )

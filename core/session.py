# session_state yönetimi

import streamlit as st
from typing import List, Tuple


ChatHistory = List[Tuple[str, str]]


def init_session():
    """Gerekli session alanlarını başlatır"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: ChatHistory = []

    if "last_pdf_hash" not in st.session_state:
        st.session_state.last_pdf_hash: str | None = None


def reset_chat_on_new_pdf(pdf_hash: str):
    """
    Yeni PDF yüklendiğinde sohbet geçmişini sıfırlar
    """
    if st.session_state.last_pdf_hash != pdf_hash:
        st.session_state.chat_history.clear()
        st.session_state.last_pdf_hash = pdf_hash

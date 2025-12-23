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
    Yeni PDF yüklendiğinde sohbet geçmişini sıfırlar ve açılış mesajını ekler
    """
    if st.session_state.last_pdf_hash != pdf_hash:
        # Eski yöntem: st.session_state.chat_history.clear()
        
        # YENİ YÖNTEM: Listeyi boşaltmak yerine ilk mesajı ekliyoruz
        # Tuple yapısı: (Rol, Mesaj) -> ("assistant", "Merhaba...")
        st.session_state.chat_history = [
            ("assistant", "PDF dosyanı başarıyla analiz ettim. İçeriğiyle ilgili merak ettiğin her şeyi sorabilirsin.")
        ]
        
        st.session_state.last_pdf_hash = pdf_hash

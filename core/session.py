import streamlit as st
from typing import List, Tuple


ChatHistory = List[Tuple[str, str]]


def init_session():
    """Gerekli session alanlarını başlatır"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: ChatHistory = []

    if "last_pdf_id" not in st.session_state:
        st.session_state.last_pdf_id: str | None = None
    
    if "rendering" not in st.session_state:
        st.session_state.rendering = False

    # ✅ Sidebar state
    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = True

def reset_chat_on_new_pdf(file_id: str):
    """
    Yeni PDF yüklendiğinde sohbet geçmişini sıfırlar ve analiz mesajını ekler
    """
    if st.session_state.last_pdf_id != file_id:
        # Analiz tamamlandı mesajı
        analysis_msg = """
Harika! 🎉 PDF dosyanı analiz ettim ve hazırım.

Artık belgenin içeriği hakkında bana istediğin soruları sorabilirsin. Ben sana en doğru cevapları vereceğim! 

Ne öğrenmek istersin? 💬
        """
        
        # Chat history'yi tamamen sıfırla
        st.session_state.chat_history = [
            ("assistant", analysis_msg.strip())
        ]
        
        # ID'yi güncelle
        st.session_state.last_pdf_id = file_id
        
        # Rendering flag'i sıfırla
        st.session_state.rendering = False
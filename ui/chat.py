import streamlit as st
from prompts.qa_prompt import QA_PROMPT


def render_chat(llm, retriever, pdf_just_loaded=False):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # İlk açılış - PDF yüklenmeden
    if llm is None and "initial_greeting" not in st.session_state:
        welcome_msg = """
Merhaba! 👋 Ben senin PDF asistanınım.

Başlamak için soldaki menüden bir PDF dosyası yükle, ben onu analiz edeyim. Sonra içeriği hakkında istediğin soruları sorabilirsin! 

Ne tür sorular sorabileceğini merak ediyorsan:
- 📄 "Bu belge ne hakkında?"
- 🔍 "X konusuyla ilgili ne diyor?"
- 📊 "Önemli noktaları özetle"
- 💡 "Y hakkında detaylı bilgi ver"

Hadi, PDF'ini yükle ve başlayalım! 🚀
        """
        st.session_state.chat_history.append(("assistant", welcome_msg.strip()))
        st.session_state.initial_greeting = True
    
    # PDF yeni yüklendi
    elif pdf_just_loaded and "pdf_analyzed" not in st.session_state:
        analysis_msg = """
Harika! 🎉 PDF dosyanı analiz ettim ve hazırım.

Artık belgenin içeriği hakkında bana istediğin soruları sorabilirsin. Ben sana en doğru cevapları vereceğim! 

Ne öğrenmek istersin? 💬
        """
        st.session_state.chat_history.append(("assistant", analysis_msg.strip()))
        st.session_state.pdf_analyzed = True

    # Geçmiş mesajları göster
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # PDF yüklenmemişse input gösterme
    if llm is None:
        return

    # Kullanıcı sorusu
    soru = st.chat_input("Bir şey sor...")

    if not soru:
        return

    # Kullanıcı mesajını kaydet ve göster
    st.session_state.chat_history.append(("user", soru))
    with st.chat_message("user"):
        st.markdown(soru)

    # Context al
    context = retriever(soru)

    # Chain oluştur
    chain = QA_PROMPT | llm

    # Asistan cevabı
    with st.chat_message("assistant"):
        # Thinking animasyonu
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            """
            <div class="thinking-container">
                <span class="thinking-text">Düşünüyor</span>
                <div class="thinking-dots">
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Cevap için placeholder
        answer_placeholder = st.empty()
        full_answer = ""
        thinking_cleared = False

        # Stream cevap
        for chunk in chain.stream({
            "context": context,
            "question": soru,
            "chat_history": st.session_state.chat_history[-6:]
        }):
            if hasattr(chunk, "content") and chunk.content:
                # İlk chunk geldiğinde thinking'i kaldır
                if not thinking_cleared:
                    thinking_placeholder.empty()
                    thinking_cleared = True

                full_answer += chunk.content
                answer_placeholder.markdown(full_answer)

    # Cevabı geçmişe ekle
    st.session_state.chat_history.append(("assistant", full_answer))
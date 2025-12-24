import streamlit as st
from prompts.qa_prompt import QA_PROMPT


def render_chat(llm, retriever):
    """Chat arayüzünü render eder"""
    
    # Chat history başlat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Mesaj render edilme kontrolü için flag
    if "rendering" not in st.session_state:
        st.session_state.rendering = False
    
    # İLK AÇILIŞ - PDF yüklenmemiş
    if llm is None:
        if len(st.session_state.chat_history) == 0:
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
        
        # Karşılama mesajını göster
        with st.chat_message("assistant"):
            st.markdown(st.session_state.chat_history[0][1])
        
        return

    # PDF YÜKLENMİŞ - Normal chat akışı
    
    # Geçmiş mesajları göster (sadece render edilmemişse)
    if not st.session_state.rendering:
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

    # Kullanıcı sorusu
    soru = st.chat_input("Bir şey sor...")

    if not soru:
        return

    # Render flag'i aktif et
    st.session_state.rendering = True
    
    # Kullanıcı mesajını kaydet
    st.session_state.chat_history.append(("user", soru))
    
    # Kullanıcı mesajını göster
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
        try:
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
                    
        except Exception as e:
            thinking_placeholder.empty()
            answer_placeholder.error(f"Bir hata oluştu: {str(e)}")
            st.session_state.rendering = False
            return

    # Cevabı kaydet
    st.session_state.chat_history.append(("assistant", full_answer))
    
    # Render flag'i kapat
    st.session_state.rendering = False
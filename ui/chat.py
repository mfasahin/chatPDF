import streamlit as st
from prompts.qa_prompt import QA_PROMPT


def render_chat(llm, retriever):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # geçmişi göster
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    soru = st.chat_input("PDF hakkında soru sor")

    if not soru:
        return

    # kullanıcı mesajı
    st.session_state.chat_history.append(("user", soru))
    with st.chat_message("user"):
        st.markdown(soru)

    # context
    context = retriever(soru)

    chain = QA_PROMPT | llm

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown("🤖 *Asistan düşünüyor…*")

        answer_box = st.empty()
        full_answer = ""
        cleared = False

        for chunk in chain.stream({
            "context": context,
            "question": soru,
            "chat_history": st.session_state.chat_history[-6:]
        }):
            if hasattr(chunk, "content") and chunk.content:
                if not cleared:
                    thinking.empty()
                    cleared = True

                full_answer += chunk.content
                answer_box.markdown(full_answer)

    st.session_state.chat_history.append(("assistant", full_answer))

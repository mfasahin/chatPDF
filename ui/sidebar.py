# sidebar UI

import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.title("📂 PDF Yükle")
        pdf = st.file_uploader("PDF seç", type="pdf")
        st.write("---")
        st.write("Model: **Llama-3.3-70B**")
        st.write("Profil: **Kalite > Hız**")
    return pdf

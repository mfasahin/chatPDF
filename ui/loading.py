import streamlit as st


def show_loading(message="PDF hazırlanıyor..."):
    """Modern loading overlay gösterir"""
    st.markdown(
        f"""
        <div class="loading-overlay">
            <div class="loading-box">
                <div class="loading-spinner"></div>
                <div class="loading-text">{message}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
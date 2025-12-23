# embedding ve cache

from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

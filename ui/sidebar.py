import streamlit as st


def render_sidebar():
    """Modern sidebar UI render eder"""
    with st.sidebar:
        st.title("📂 PDF Yükle")
        
        pdf = st.file_uploader(
            "PDF dosyasını seç",
            type="pdf",
            help="Sohbet etmek istediğin PDF'i buraya yükle"
        )
        
        if pdf:
            st.success(f"✅ {pdf.name}")
        
        st.markdown("---")
        
        # Model bilgileri
        st.markdown("### ⚙️ Sistem Bilgisi")
        st.markdown("""
        **🤖 Model:** Llama-3.3-70B  
        **⚡ Mod:** Kalite Odaklı  
        **🔍 Arama:** Hybrid (FAISS + BM25)  
        **📊 Chunk:** 4 sonuç
        """)
        
        st.markdown("---")
        
        # Bilgi kutusu
        with st.expander("💡 İpuçları"):
            st.markdown("""
            **Nasıl soru sorabilirim?**
            
            ✨ Spesifik ol:  
            "X konusu hakkında ne diyor?"
            
            📝 Özet iste:  
            "Ana noktaları özetle"
            
            🔎 Detay ara:  
            "Y hakkında detaylı bilgi ver"
            
            📊 Karşılaştır:  
            "A ile B arasındaki fark nedir?"
            """)
        
        # Footer
        st.markdown("---")
        st.caption("🤖 PDF Chat Assistant v2.0")
    
    return pdf
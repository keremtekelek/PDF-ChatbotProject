import streamlit as st
import atexit
import os

# Mevcut modüller
import llm_manager
import pdf_manager
import data_manager
import chain_manager

from langchain_huggingface import HuggingFaceEmbeddings

# Sayfa Yapılandırması
st.set_page_config(
    page_title="PDF AI Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kapanışta Ollama sürecini temizleme
atexit.register(llm_manager.close_llm)

# Özel CSS ile Modern Arayüz Teması
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1E88E5;
        padding: 10px;
        border-radius: 4px;
        margin-top: 8px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Session State Başlatma
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "available_sources" not in st.session_state:
    st.session_state.available_sources = []

if "embedding_model" not in st.session_state:
    with st.spinner("Embedding modeli yükleniyor..."):
        st.session_state.embedding_model = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

# --- Yan Panel (Sidebar) ---
with st.sidebar:
    st.title("⚙️ Kontrol Paneli")
    
    # Ollama Servis Kontrolü
    llm_status = llm_manager.start_llm_automatically()
    if llm_status:
        st.success("🟢 Local LLM (Ollama) Aktif", icon="✅")
    else:
        st.error("🔴 Local LLM Başlatılamadı! Ollama servisinin açık olduğundan emin olun.", icon="⚠️")

    st.markdown("---")
    
    # PDF Sürükle-Bırak Yükleme Alanı
    st.subheader("📁 Belge Yükleme")
    uploaded_files = st.file_uploader(
        "PDF dosyalarını buraya sürükleyin veya seçin",
        type=["pdf"],
        accept_multiple_files=True,
        help="Bir veya birden fazla PDF dosyası yükleyebilirsiniz."
    )

    if uploaded_files and st.button("🚀 PDF'leri İşle ve İndeksle", use_container_width=True):
        with st.spinner("PDF sayfaları okunuyor ve vektör veritabanı oluşturuluyor..."):
            try:
                # 1. PDF'leri Oku
                docs = pdf_manager.read_uploaded_pdfs(uploaded_files)
                
                if docs:
                    # 2. Chunk'lara böl
                    chunks = data_manager.chunk(docs)
                    
                    # 3. Vektör Veritabanı Oluştur
                    vector_db = data_manager.create_vector_database(chunks, st.session_state.embedding_model)
                    st.session_state.vector_db = vector_db
                    
                    # 4. Kaynakları listele
                    sources = data_manager.list_sources(chunks)
                    st.session_state.available_sources = sources
                    
                    st.success(f"{len(uploaded_files)} PDF ({len(chunks)} parça) başarıyla indekslendi!")
                else:
                    st.warning("Yüklenen dosyalarda okunabilir metin bulunamadı.")
            except Exception as e:
                st.error(f"İşlem sırasında hata oluştu: {e}")

    # Kaynak Filtresi (Opsiyonel)
    if st.session_state.available_sources:
        st.markdown("---")
        st.subheader("🎯 Arama Filtresi")
        selected_source = st.selectbox(
            "Hedef Belge:",
            options=["Tüm Belgeler"] + st.session_state.available_sources,
            index=0
        )
        
        # Filtreye göre retriever ve RAG zincirini güncelle
        filter_val = None if selected_source == "Tüm Belgeler" else selected_source
        retriever = data_manager.create_retriever(
            st.session_state.vector_db,
            k=3,
            source_filter=filter_val
        )
        llm = llm_manager.create_llm(model_name="llama3.1")
        st.session_state.rag_chain = chain_manager.create_rag_chain(retriever, llm)

    st.markdown("---")
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Ana Ekran (Sohbet Alanı) ---
st.markdown('<div class="main-header">📄 PDF AI Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Belgelerinizi yükleyin ve içerikleri hakkında doğrudan soru sorun.</div>', unsafe_allow_html=True)

# Önceki Mesajları Görüntüle
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "references" in msg and msg["references"]:
            with st.expander("📚 İncelenen Kaynaklar"):
                for src, page in msg["references"]:
                    st.markdown(f"- **{src}** (Sayfa {page})")

# Kullanıcı Soru Girişi
if prompt := st.chat_input("PDF'ler hakkında bir soru sorun..."):
    if not st.session_state.rag_chain:
        st.info("Lütfen önce sol panelden PDF yükleyip 'PDF'leri İşle ve İndeksle' butonuna basın.")
    else:
        # Kullanıcı mesajını kaydet ve göster
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Asistan Cevabı Üret
        with st.chat_message("assistant"):
            with st.spinner("Belgeler taranıyor ve cevap üretiliyor..."):
                try:
                    result = st.session_state.rag_chain.invoke({"input": prompt})
                    answer = result.get("answer", "")
                    
                    # Referans kaynakları topla
                    used_references = []
                    for doc in result.get("context", []):
                        ref = (doc.metadata.get("source", "Bilinmeyen"), doc.metadata.get("page", 0) + 1)
                        if ref not in used_references:
                            used_references.append(ref)

                    st.markdown(answer)
                    
                    if used_references:
                        with st.expander("📚 İncelenen Kaynaklar"):
                            for src, page in used_references:
                                st.markdown(f"- **{src}** (Sayfa {page})")

                    # Asistan cevabını ve kaynakları kaydet
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "references": used_references
                    })

                except Exception as e:
                    st.error(f"Cevap üretilirken bir hata oluştu: {e}")
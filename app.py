import streamlit as st
import atexit
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings

import llm_manager
import pdf_manager
import data_manager
import agent_tools
import graph_agent

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Agentic PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

atexit.register(llm_manager.close_llm)

# Özel CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1rem; color: #6c757d; margin-bottom: 1.5rem; }
    .agent-step { background-color: #f1f3f4; border-left: 4px solid #f2994a; padding: 8px 12px; margin: 6px 0; border-radius: 4px; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # LangChain message nesneleri

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # UI gösterim listesi

if "agent_graph" not in st.session_state:
    st.session_state.agent_graph = None

if "available_sources" not in st.session_state:
    st.session_state.available_sources = []

if "embedding_model" not in st.session_state:
    with st.spinner("Embedding modeli hazırlanıyor..."):
        st.session_state.embedding_model = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

# --- Sol Panel (Sidebar) ---
with st.sidebar:
    st.title("🤖 Ajan Kontrol Paneli")
    
    # LLM Durumu
    if llm_manager.start_llm_automatically():
        st.success("🟢 Local LLM (Ollama) Aktif")
    else:
        st.error("🔴 Local LLM Kapalı! Ollama'yı başlatın.")

    st.markdown("---")
    st.subheader("📄 PDF Yükleme")
    uploaded_files = st.file_uploader(
        "PDF dosyalarını sürükleyip bırakın:",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Ajanı Hazırla (İndeksle)", use_container_width=True):
        with st.spinner("PDF'ler taranıyor ve Ajan araçları kuruluyor..."):
            docs = pdf_manager.read_uploaded_pdfs(uploaded_files)
            if docs:
                chunks = data_manager.chunk(docs)
                vector_db = data_manager.create_vector_database(chunks, st.session_state.embedding_model)
                retriever = data_manager.create_retriever(vector_db, k=3)
                
                # Dinamik PDF tool'u ve matematik araçlarını birleştir
                pdf_tool = agent_tools.build_pdf_search_tool(retriever)
                all_tools = agent_tools.base_math_tools + [pdf_tool]
                
                llm = llm_manager.create_llm(model_name="llama3.1")
                st.session_state.agent_graph = graph_agent.create_agent_graph(llm, all_tools)
                st.session_state.available_sources = data_manager.list_sources(chunks)
                
                st.success(f"Ajan hazır! {len(uploaded_files)} PDF ve {len(all_tools)} Araç bağlandı.")
            else:
                st.warning("Okunabilir metin bulunamadı.")

    st.markdown("---")
    st.subheader("🛠️ Ajanın Araçları")
    st.markdown("- 🔍 `search_pdf` (PDF Tarayıcı)\n- ✖️ `multiply_operation` (Çarpma)\n- ➕ `addition_operation` (Toplama)\n- ➖ `subtract_operation` (Çıkarma)\n- ➗ `division_operation` (Bölme)")

    st.markdown("---")
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.display_messages = []
        st.rerun()

# --- Ana Ekran ---
st.markdown('<div class="main-header">🤖 Agentic PDF Chatbot (LangGraph)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">PDF belgeleriniz hakkında sorular sorun; ajanınız gerektiğinde belgeleri tarar, gerektiğinde hesaplama araçlarını kullanır.</div>', unsafe_allow_html=True)

# Önceki Mesajları Göster
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "steps" in msg and msg["steps"]:
            with st.expander("🔍 Ajanın Düşünce ve Araç Çağrıları"):
                for step in msg["steps"]:
                    st.markdown(f"**Araç:** `{step['tool']}`\n\n**Parametre:** `{step['args']}`\n\n**Sonuç:**\n```\n{step['result']}\n```")

# Soru Girişi
if prompt := st.chat_input("Bir soru sorun (Örn: 'PDF'teki satış tutarını bul ve 4 ile çarp')"):
    if not st.session_state.agent_graph:
        st.info("Lütfen önce sol panelden PDF yükleyip 'Ajanı Hazırla' butonuna basın.")
    else:
        # Kullanıcı mesajını göster ve ekle
        st.session_state.display_messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Ajanı Çalıştır
        with st.chat_message("assistant"):
            agent_placeholder = st.empty()
            with st.spinner("Ajan düşünüyor ve araçları yönetiyor..."):
                try:
                    tool_steps = []
                    final_answer = ""
                    
                    # LangGraph akışını yürüt
                    state_input = {"messages": st.session_state.chat_history}
                    for event in st.session_state.agent_graph.stream(state_input, stream_mode="values"):
                        last_message = event["messages"][-1]
                        
                        # Araç çağrısı varsa kaydet
                        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                            for tc in last_message.tool_calls:
                                tool_steps.append({
                                    "tool": tc["name"],
                                    "args": tc["args"],
                                    "result": "Çalıştırılıyor..."
                                })
                        
                        # Araç sonucu döndüyse güncelle
                        if isinstance(last_message, ToolMessage):
                            if tool_steps:
                                tool_steps[-1]["result"] = str(last_message.content)
                                
                        # Nihai cevap
                        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                            final_answer = last_message.content

                    agent_placeholder.markdown(final_answer)
                    
                    if tool_steps:
                        with st.expander("🔍 Ajanın Düşünce ve Araç Çağrıları"):
                            for step in tool_steps:
                                st.markdown(f"**Araç:** `{step['tool']}`  \n**Parametreler:** `{step['args']}`  \n**Çıktı:**\n```\n{step['result']}\n```")

                    # Geçmişi güncelle
                    st.session_state.chat_history.append(AIMessage(content=final_answer))
                    st.session_state.display_messages.append({
                        "role": "assistant",
                        "content": final_answer,
                        "steps": tool_steps
                    })

                except Exception as e:
                    st.error(f"Ajan çalışırken hata oluştu: {e}")
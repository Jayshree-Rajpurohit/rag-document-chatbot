import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1D6E6E, #534AB7);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }
    .chat-user {
        background: #E8F4FD;
        padding: 0.75rem 1rem;
        border-radius: 12px 12px 4px 12px;
        margin: 0.5rem 0;
        margin-left: 20%;
        color: #1A1A1A;
    }
    .chat-bot {
        background: #F0F0FF;
        padding: 0.75rem 1rem;
        border-radius: 12px 12px 12px 4px;
        margin: 0.5rem 0;
        margin-right: 20%;
        color: #1A1A1A;
    }
    .source-box {
        background: #F8F9FA;
        border-left: 3px solid #1D6E6E;
        padding: 0.5rem 0.75rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.8rem;
        color: #444;
        margin-top: 0.4rem;
    }
    .status-ready   { color: #1D6E6E; font-weight: 500; }
    .status-pending { color: #C05A1F; font-weight: 500; }
    .stButton > button {
        background: linear-gradient(135deg, #1D6E6E, #534AB7);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 RAG Document Chatbot</h1>
    <p>Upload a PDF and ask questions — answers come directly from your document</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "chain"         not in st.session_state: st.session_state.chain          = None
if "doc_processed" not in st.session_state: st.session_state.doc_processed  = False
if "doc_name"      not in st.session_state: st.session_state.doc_name       = ""

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Get your key from platform.openai.com"
    )

    st.divider()
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Upload any PDF — research paper, company policy, notes, etc."
    )

    chunk_size    = st.slider("Chunk size",    300, 1500, 800,  50,  help="Size of each text chunk")
    chunk_overlap = st.slider("Chunk overlap", 0,   300,  100,  25,  help="Overlap between chunks for context continuity")

    process_btn = st.button("🚀 Process Document", use_container_width=True)

    if st.session_state.doc_processed:
        st.success(f"✅ Ready: **{st.session_state.doc_name}**")
    else:
        st.info("⬆️ Upload a PDF to get started")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history  = []
        st.session_state.chain         = None
        st.session_state.doc_processed = False
        st.session_state.doc_name      = ""
        st.rerun()

    st.divider()
    st.markdown("""
    **How it works:**
    1. Upload any PDF document
    2. Document is split into chunks
    3. Chunks stored in ChromaDB (vector DB)
    4. Your question → finds relevant chunks → LLM answers

    **Built with:**
    `Python` · `LangChain` · `ChromaDB` · `OpenAI` · `Streamlit`
    """)

# ── Process document ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def process_document(file_bytes, filename, _api_key, chunk_size, chunk_overlap):
    """Load PDF → split → embed → store in ChromaDB → return chain."""

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Load PDF
    loader   = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Embed + store in ChromaDB
    embeddings   = OpenAIEmbeddings(openai_api_key=_api_key)
    vectorstore  = Chroma.from_documents(chunks, embeddings)

    # LLM
    llm = ChatOpenAI(
        openai_api_key=_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.2
    )

    # Memory for multi-turn conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # RAG chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
    )

    os.unlink(tmp_path)
    return chain, len(chunks)

# ── Handle process button ─────────────────────────────────────
if process_btn:
    if not api_key:
        st.sidebar.error("⚠️ Please enter your OpenAI API key first!")
    elif not uploaded_file:
        st.sidebar.error("⚠️ Please upload a PDF file first!")
    else:
        with st.spinner("📚 Processing document — chunking, embedding, storing..."):
            try:
                chain, num_chunks = process_document(
                    uploaded_file.read(),
                    uploaded_file.name,
                    api_key,
                    chunk_size,
                    chunk_overlap
                )
                st.session_state.chain         = chain
                st.session_state.doc_processed = True
                st.session_state.doc_name      = uploaded_file.name
                st.session_state.chat_history  = []
                st.sidebar.success(f"✅ Processed into {num_chunks} chunks!")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")

# ── Main chat area ────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat with your document")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            if st.session_state.doc_processed:
                st.info(f"📄 **{st.session_state.doc_name}** is ready! Ask me anything about it.")
            else:
                st.info("👈 Upload a PDF from the sidebar and click **Process Document** to start chatting.")
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                    if msg.get("sources"):
                        with st.expander("📎 Source passages used", expanded=False):
                            for i, src in enumerate(msg["sources"], 1):
                                st.markdown(f'<div class="source-box"><b>Source {i} — Page {src["page"]}:</b><br>{src["text"]}</div>', unsafe_allow_html=True)

    st.divider()

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question about your document:",
            placeholder="e.g. What are the main findings? Summarise chapter 2. What does it say about X?",
            disabled=not st.session_state.doc_processed
        )
        submit = st.form_submit_button("Send ➤", use_container_width=True)

    if submit and user_input and st.session_state.chain:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("🔍 Searching document and generating answer..."):
            try:
                result  = st.session_state.chain({"question": user_input})
                answer  = result["answer"]
                sources = []

                for doc in result.get("source_documents", []):
                    sources.append({
                        "page": doc.metadata.get("page", "?") + 1,
                        "text": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    })

                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources
                })
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")

with col2:
    st.subheader("📊 Document Stats")

    if st.session_state.doc_processed:
        st.metric("Document", st.session_state.doc_name[:25] + "..." if len(st.session_state.doc_name) > 25 else st.session_state.doc_name)
        st.metric("Messages",  len([m for m in st.session_state.chat_history if m["role"] == "user"]))
        st.metric("Status", "✅ Ready")

        st.divider()
        st.markdown("**💡 Example questions:**")
        example_questions = [
            "Summarise this document",
            "What are the key points?",
            "What does it say about [topic]?",
            "List the main conclusions",
            "Explain [specific term]",
        ]
        for q in example_questions:
            st.markdown(f"• *{q}*")
    else:
        st.markdown("""
        **📌 Steps to start:**
        1. Enter OpenAI API key
        2. Upload a PDF
        3. Click Process Document
        4. Start asking questions!
        """)
        st.divider()
        st.markdown("""
        **🎯 Good PDFs to try:**
        - Research papers
        - Company reports
        - Study notes
        - Policy documents
        - Any textbook chapter
        """)

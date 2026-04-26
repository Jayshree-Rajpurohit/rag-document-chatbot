import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Chatbot",
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
    .stButton > button {
        background: linear-gradient(135deg, #1D6E6E, #534AB7);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 RAG Document Chatbot</h1>
    <p>Upload a PDF and ask questions — answers come directly from your document</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "vectorstore"   not in st.session_state: st.session_state.vectorstore   = None
if "doc_processed" not in st.session_state: st.session_state.doc_processed = False
if "doc_name"      not in st.session_state: st.session_state.doc_name      = ""

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    st.markdown("💡 **Powered by GitHub Models + LangChain RAG**")
    api_key = st.text_input(
        "GitHub Token",
        type="password",
        placeholder="ghp_...",
        help="Get your free key from aistudio.google.com/app/apikey"
    )

    st.divider()
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    chunk_size    = st.slider("Chunk size",    300, 1500, 800, 50)
    chunk_overlap = st.slider("Chunk overlap", 0,   300,  100, 25)
    process_btn   = st.button("🚀 Process Document", use_container_width=True)

    if st.session_state.doc_processed:
        st.success(f"✅ Ready: **{st.session_state.doc_name}**")
    else:
        st.info("⬆️ Upload a PDF to get started")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history  = []
        st.session_state.vectorstore   = None
        st.session_state.doc_processed = False
        st.session_state.doc_name      = ""
        st.rerun()

    st.divider()
    st.markdown("""
    **How it works:**
    1. Upload any PDF document
    2. Document split into chunks
    3. Chunks stored in ChromaDB
    4. Question → finds relevant chunks → Gemini answers

    **Built with:**
    `Python` · `LangChain` · `ChromaDB` · `Gemini` · `Streamlit`

    🔗 Get free key:
    aistudio.google.com/app/apikey
    """)

# ── Process document ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def process_document(file_bytes, filename, chunk_size, chunk_overlap):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader    = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    # Free local embeddings — no API key needed!
    with st.spinner("📥 Loading embedding model (first time only ~1 min)..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

    vectorstore = Chroma.from_documents(chunks, embeddings)
    os.unlink(tmp_path)
    return vectorstore, len(chunks)

# ── Format docs ───────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Get answer ────────────────────────────────────────────────
def get_answer(question, vectorstore, chat_history, api_key):
    llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    base_url="https://models.inference.ai.azure.com",
    temperature=0.2
)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Format chat history
    history_text = ""
    for msg in chat_history[-6:]:
        if isinstance(msg, HumanMessage):
            history_text += f"Human: {msg.content}\n"
        else:
            history_text += f"Assistant: {msg.content}\n"

    docs    = retriever.invoke(question)
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided document context.
Answer only from the context below. If the answer is not in the context, say 'I could not find that in the document.'

Previous conversation:
{history}

Document context:
{context}"""),
        ("human", "{question}")
    ])

    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "history":  history_text,
        "context":  context,
        "question": question
    })

    return answer, docs

# ── Handle process button ─────────────────────────────────────
if process_btn:
    if not uploaded_file:
        st.sidebar.error("⚠️ Please upload a PDF file!")
    else:
        with st.spinner("📚 Processing document..."):
            try:
                vectorstore, num_chunks = process_document(
                    uploaded_file.read(),
                    uploaded_file.name,
                    chunk_size,
                    chunk_overlap
                )
                st.session_state.vectorstore   = vectorstore
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

    if not st.session_state.chat_history:
        if st.session_state.doc_processed:
            st.info(f"📄 **{st.session_state.doc_name}** is ready! Ask me anything.")
        else:
            st.info("👈 Upload a PDF from the sidebar and click Process Document to start.")
    else:
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                st.markdown(f'<div class="chat-user">🧑 {msg.content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">🤖 {msg.content}</div>', unsafe_allow_html=True)

    st.divider()

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question:",
            placeholder="e.g. Summarise this document. What does it say about X?",
            disabled=not st.session_state.doc_processed
        )
        submit = st.form_submit_button("Send ➤", use_container_width=True)

    if submit and user_input and st.session_state.vectorstore:
        if not api_key:
            st.error("⚠️ Please enter your Gemini API key in the sidebar!")
        else:
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            with st.spinner("🔍 Searching and generating answer..."):
                try:
                    answer, source_docs = get_answer(
                        user_input,
                        st.session_state.vectorstore,
                        st.session_state.chat_history,
                        api_key
                    )
                    st.session_state.chat_history.append(AIMessage(content=answer))

                    if source_docs:
                        with st.expander("📎 Source passages used", expanded=False):
                            for i, doc in enumerate(source_docs, 1):
                                page = doc.metadata.get("page", "?")
                                st.markdown(f"**Source {i} — Page {page}:**")
                                st.caption(doc.page_content[:300] + "...")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with col2:
    st.subheader("📊 Stats")
    if st.session_state.doc_processed:
        name = st.session_state.doc_name
        st.metric("Document", name[:20] + "..." if len(name) > 20 else name)
        st.metric("Messages", len([m for m in st.session_state.chat_history if isinstance(m, HumanMessage)]))
        st.metric("Status", "✅ Ready")
        st.divider()
        st.markdown("**💡 Try asking:**")
        for q in ["Summarise this document", "What are the key points?", "Explain [topic]", "List the main conclusions"]:
            st.markdown(f"• *{q}*")
    else:
        st.markdown("""
        **📌 Steps:**
        1. Get free Gemini API key
        2. Upload a PDF
        3. Click Process Document
        4. Start chatting!

        🔗 Get key:
        aistudio.google.com/app/apikey
        """)

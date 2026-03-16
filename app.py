import os
import sys
import tempfile
import traceback
import streamlit as st
from dotenv import load_dotenv

# --- RAG Pipeline Imports ---
from rag.loader import load_single_pdf
from rag.chunking import split_documents
from rag.embeddings import get_embedding_model
from rag.vectorstore import VectorStoreManager
from tools.retrieval_tool import HybridRetriever
from agents.pdf_agent import PDFAgent

load_dotenv()

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="AI PDF Agent",
    page_icon="📄",
    layout="wide"
)

# ============================================================
# Session State Initialization
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "document_summary" not in st.session_state:
    st.session_state.document_summary = ""
if "pdfs_uploaded" not in st.session_state:
    st.session_state.pdfs_uploaded = False
if "processing_error" not in st.session_state:
    st.session_state.processing_error = ""

# ============================================================
# Cached Resource: Embedding Model (loaded once, survives reruns)
# ============================================================
@st.cache_resource(show_spinner="Loading embedding model (first time may take a minute)...")
def load_embedding_model():
    return get_embedding_model()

# ============================================================
# Core Processing Function
# ============================================================
def process_uploaded_pdfs(uploaded_files):
    """
    Full pipeline: PDF → chunks → FAISS + BM25 → LangGraph Agent → Summary.
    All in-process, no HTTP calls.
    """
    st.session_state.processing_error = ""

    # Ollama configuration (no API key required)
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")  # optional, default is Ollama's local URL
    use_reranker = os.getenv("USE_RERANKER", "0").strip().lower() in {"1", "true", "yes", "y"}

    status = st.status("Processing documents...", expanded=True)

    try:
        all_docs = []
        file_names = []

        # --- Save & Load PDFs ---
        with tempfile.TemporaryDirectory() as tmpdir:
            for file in uploaded_files:
                status.write(f"📄 Loading {file.name}...")
                filepath = os.path.join(tmpdir, file.name)
                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())
                docs = load_single_pdf(filepath)
                all_docs.extend(docs)
                file_names.append(file.name)

        if not all_docs:
            st.session_state.processing_error = "❌ No text could be extracted from the uploaded PDFs."
            status.update(label="Failed", state="error")
            return False

        # --- Chunking ---
        status.write("✂️ Splitting into chunks...")
        chunks = split_documents(all_docs)

        # --- Embeddings + FAISS ---
        status.write("🧠 Building vector database (embedding model loading)...")
        embeddings = load_embedding_model()
        vectorstore_manager = VectorStoreManager(embeddings, persist_directory=None)
        vectorstore_manager.add_documents(chunks)

        # --- Hybrid Retriever ---
        status.write("🔍 Building hybrid retriever (FAISS + BM25)...")
        hybrid_retriever = HybridRetriever(vectorstore_manager)
        hybrid_retriever.build_ensemble_retriever(chunks)

        # --- LangGraph Agent ---
        status.write("🤖 Initializing LangGraph agent...")
        st.session_state.agent = PDFAgent(
            retriever_callable=hybrid_retriever.get_retriever(),
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
            use_reranker=use_reranker,
        )

        # --- Auto Summary ---
        status.write("📝 Generating document summary...")
        summary_result = st.session_state.agent.run(
            "Provide a concise high-level summary of the document(s). What are the main topics and key takeaways?",
            chat_history=[]
        )
        st.session_state.document_summary = summary_result.get("answer", "Summary unavailable.")

        # Reset chat for new docs
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.pdfs_uploaded = True

        status.update(label=f"✅ Processed {len(file_names)} file(s) successfully!", state="complete")
        return True

    except Exception as e:
        error_detail = traceback.format_exc()
        print(error_detail, file=sys.stderr)
        st.session_state.processing_error = f"❌ Error: {e}"
        status.update(label="Processing failed", state="error")
        status.write(f"```\n{error_detail}\n```")
        return False


# ============================================================
# UI Layout
# ============================================================
st.title("📄 Context-Aware AI PDF Assistant")
st.markdown("Upload your PDFs, review the automatic summary, and ask questions!")

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("📁 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("⚙️ Process Documents", disabled=not uploaded_files):
        success = process_uploaded_pdfs(uploaded_files)
        if success:
            st.rerun()

# Show processing error if any (persists across reruns)
if st.session_state.processing_error:
    st.error(st.session_state.processing_error)

# --- Main Area ---
if st.session_state.pdfs_uploaded:

    # Document Summary
    with st.expander("📝 Document Summary", expanded=True):
        st.write(st.session_state.document_summary)

    st.divider()
    st.subheader("💬 Chat with your Documents")

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("📚 Sources"):
                    for cite in message["citations"]:
                        st.markdown(f"- `{cite}`")

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.agent.run(
                        prompt,
                        chat_history=st.session_state.chat_history
                    )

                    answer = result.get("answer", "Sorry, I could not generate an answer.")
                    raw_citations = result.get("citations", [])
                    st.session_state.chat_history = result.get("chat_history", [])

                    # Format citations
                    citations = []
                    for c in raw_citations:
                        src = os.path.basename(c.get("source", "Unknown"))
                        page = c.get("page", -1)
                        if page != -1:
                            citations.append(f"{src} (Page {page})")
                        else:
                            citations.append(src)
                    citations = list(set(citations))

                    st.markdown(answer)
                    if citations:
                        with st.expander("📚 Sources"):
                            for cite in citations:
                                st.markdown(f"- `{cite}`")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })

                except Exception as e:
                    err_msg = f"❌ Error: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

else:
    st.info("👈 Upload PDF documents using the sidebar to get started.")

    st.markdown("""
    ### What this assistant can do:
    | Feature | Description |
    |---------|-------------|
    | 📤 Multi-PDF Upload | Process multiple documents at once |
    | 📝 Auto Summary | Instant overview on upload |
    | 🔍 Hybrid Search | FAISS semantic + BM25 keyword retrieval |
    | 🤖 Self-Reflection | Agent critiques & refines its answers |
    | 📌 Source Citations | Exact filename + page number per answer |
    | 💬 Conversation Memory | Supports natural follow-up questions |
    """)

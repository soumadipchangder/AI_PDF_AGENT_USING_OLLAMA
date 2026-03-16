## AI PDF Agent

A local-first Retrieval-Augmented Generation (RAG) system for querying and summarizing PDF documents. Uses **Ollama** as the LLM backend and **LangGraph** with self-reflection for high-quality, citation-backed answers.

## Features

- **Multi-Document Ingestion**: Upload and process multiple PDFs simultaneously.
- **Hybrid Retrieval**: Custom retriever combining FAISS (dense semantic) + BM25 (sparse keyword) for high recall.
- **Self-Reflecting Agent**: LangGraph workflow that generates, evaluates, and optionally rewrites answers.
- **Automatic Summarization**: Generates a high-level summary upon document upload.
- **Source Citations**: Returns exact source filename and page numbers with every answer.
- **Conversational Memory**: Maintains context for natural follow-up questions.

## Architecture

```
Streamlit UI (app.py)
│
├── PDF Upload → loader.py → chunking.py → embeddings.py → vectorstore.py (FAISS)
│
├── Hybrid Retriever (tools/retrieval_tool.py) → FAISS + BM25
│
└── LangGraph Agent (agents/pdf_agent.py) → Ollama LLM (e.g. llama3.1:latest)
```

Single-process architecture — no separate backend server needed.

## Local Setup (Ollama)

### 1. Prerequisites
- **Python**: 3.10+ recommended (project also works with a venv on newer 3.13)
- **Ollama** installed and running locally  
  - Install from `https://ollama.com`
  - Make sure the service is running (Ollama app open or `ollama serve`)
- A model available in Ollama, e.g.:

```bash
ollama pull llama3.1
```

### 2. Create and activate virtual environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate  # on macOS / Linux
# .venv\Scripts\activate   # on Windows PowerShell
```

### 3. Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

### 4. Configure environment (optional)

Create a `.env` file in the project root if you want to override defaults:

```bash
echo 'OLLAMA_MODEL="llama3.1:latest"' >> .env
echo 'OLLAMA_BASE_URL="http://localhost:11434"' >> .env  # only if you changed Ollama port/host
echo 'USE_RERANKER="0"' >> .env                          # set to "1" to enable cross-encoder reranker
```

Defaults if you do nothing:
- `OLLAMA_MODEL`: `llama3.1:latest`
- `OLLAMA_BASE_URL`: Ollama’s default local URL
- `USE_RERANKER`: `"0"` (disabled to avoid large model download)

### 5. Run the app locally

From the project root, with the venv active:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

## Directory Structure

```
├── app.py                     # Streamlit App (single entry point)
├── agents/
│   └── pdf_agent.py           # LangGraph Agent (retrieve → generate → reflect)
├── rag/
│   ├── loader.py              # PDF ingestion via PyPDF
│   ├── chunking.py            # RecursiveCharacterTextSplitter
│   ├── embeddings.py          # HuggingFace sentence-transformer
│   └── vectorstore.py         # FAISS vector database
├── tools/
│   └── retrieval_tool.py      # Custom FAISS + BM25 hybrid retriever
├── requirements.txt
├── Dockerfile
└── README.md
```

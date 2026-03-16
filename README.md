---
title: AI PDF Agent
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# AI PDF Agent

A production-ready Retrieval-Augmented Generation (RAG) system for querying and summarizing PDF documents. Powered by LangGraph with self-reflection for high-quality, citation-backed answers.

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
└── LangGraph Agent (agents/pdf_agent.py) → Groq LLM (llama-3.1-8b-instant)
```

Single-process architecture — no backend server needed.

## Deployment on Hugging Face Spaces

### 1. Create a New Space
- Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create New Space**
- Select **Docker** as the SDK

### 2. Set the GROQ_API_KEY Secret
- Go to your Space **Settings → Repository Secrets**
- Add a secret named `GROQ_API_KEY` with your key from [console.groq.com](https://console.groq.com/keys)

### 3. Push the Code
```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git add .
git commit -m "Initial commit"
git push space main
```

The Space will build and expose the Streamlit UI on port 7860 automatically.

---

## Local Development

### Prerequisites
- Python 3.10+
- A valid [Groq API Key](https://console.groq.com/keys)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
echo 'GROQ_API_KEY="your-key-here"' > .env

# Run the app
streamlit run app.py
```

### Docker (Local)
```bash
docker build -t pdf-agent .
docker run -p 7860:7860 -e GROQ_API_KEY="your-groq-api-key" pdf-agent
```
Access the app at [http://localhost:7860](http://localhost:7860)

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

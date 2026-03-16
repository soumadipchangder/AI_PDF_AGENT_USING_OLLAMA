# ================================================================
# Hugging Face Spaces compatible Dockerfile
# Single-process: only Streamlit on port 7860
# Flask has been removed — Streamlit calls the RAG pipeline directly
# ================================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies for building native packages (faiss, tokenizers, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# HuggingFace Spaces requires exactly port 7860
EXPOSE 7860

# Start the Streamlit app — this is the only process needed
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]

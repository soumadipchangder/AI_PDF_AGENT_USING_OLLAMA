import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdfs_from_directory(directory_path: str) -> List[Document]:
    """
    Loads all PDF files from a given directory and its subdirectories.
    
    Args:
        directory_path (str): The path to the directory containing PDF files.
        
    Returns:
        List[Document]: A list of LangChain Document objects containing the text
                       and metadata (source file, page numbers) of the PDFs.
    """
    documents = []
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        return documents

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded {len(docs)} pages from {file}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    
    return documents

def load_single_pdf(file_path: str) -> List[Document]:
    """
    Loads a single PDF file.
    
    Args:
        file_path (str): Path to the PDF file.
        
    Returns:
        List[Document]: The loaded documents from the PDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at {file_path}")
        
    loader = PyPDFLoader(file_path)
    return loader.load()

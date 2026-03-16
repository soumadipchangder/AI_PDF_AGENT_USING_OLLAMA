import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class VectorStoreManager:
    """
    Manages the FAISS vector database for semantic retrieval of PDF chunks.
    Supports both persistent (disk) and in-memory (transient) modes.
    """
    
    def __init__(self, embeddings: Embeddings, persist_directory: Optional[str] = "faiss_index"):
        self.embeddings = embeddings
        self.persist_directory = persist_directory  # None = in-memory only
        self.vectorstore: Optional[FAISS] = None
        
        # Load existing vector db only if persist_directory is provided and exists
        if self.persist_directory and os.path.exists(self.persist_directory):
            print(f"Loading existing vectorstore from {self.persist_directory}")
            try:
                self.vectorstore = FAISS.load_local(
                    folder_path=self.persist_directory,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Failed to load vectorstore: {e}. Starting fresh.")
                self.vectorstore = None
                
    def add_documents(self, documents: List[Document]):
        """
        Adds new documents to the vectorstore. Persists to disk if persist_directory is set.
        """
        if not documents:
            return
            
        if self.vectorstore is None:
            print(f"Initializing new vectorstore with {len(documents)} chunks...")
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            print(f"Adding {len(documents)} chunks to existing vectorstore...")
            self.vectorstore.add_documents(documents)
            
        # Only save if a persist path is provided
        if self.persist_directory:
            self.vectorstore.save_local(self.persist_directory)
            print(f"Vectorstore saved to {self.persist_directory}")
        
    def get_retriever(self, search_kwargs: dict = {"k": 4}):
        """Returns a base retriever from the vector store."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore has not been initialized. Please upload documents first.")
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        
    def get_vectorstore(self):
        return self.vectorstore

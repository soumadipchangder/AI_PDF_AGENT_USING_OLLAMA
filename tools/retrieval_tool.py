from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import Field
from rag.vectorstore import VectorStoreManager


class HybridCustomRetriever(BaseRetriever):
    """
    A custom retriever that manually combines results from FAISS (dense) and BM25 (sparse)
    without depending on EnsembleRetriever — which has unstable import paths across
    LangChain versions.
    """
    faiss_retriever: Any = Field(description="FAISS vector similarity retriever")
    bm25_retriever: Any = Field(description="BM25 keyword-based retriever")
    k: int = Field(default=6, description="Number of final results to return")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Fetch results from both retrievers, merge, deduplicate, and return top-k.
        """
        faiss_docs = self.faiss_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # Merge: prioritize documents that appear in both (interleave strategy)
        seen_contents = set()
        merged = []

        # Interleave results from both sources for diversity
        max_len = max(len(faiss_docs), len(bm25_docs))
        for i in range(max_len):
            if i < len(faiss_docs):
                doc = faiss_docs[i]
                key = doc.page_content[:200]
                if key not in seen_contents:
                    seen_contents.add(key)
                    merged.append(doc)
            if i < len(bm25_docs):
                doc = bm25_docs[i]
                key = doc.page_content[:200]
                if key not in seen_contents:
                    seen_contents.add(key)
                    merged.append(doc)

        return merged[:self.k]


class HybridRetriever:
    """
    Wrapper that builds the HybridCustomRetriever from a VectorStoreManager
    and a set of documents (for BM25 indexing).
    """

    def __init__(self, vectorstore_manager: VectorStoreManager):
        self.vectorstore_manager = vectorstore_manager
        self.retriever: Optional[HybridCustomRetriever] = None
        self.all_documents: List[Document] = []

    def build_ensemble_retriever(self, documents: List[Document]):
        """
        Builds the combined retriever from chunked documents.
        Must be called after documents are added to the vector store.
        """
        if not documents:
            return

        self.all_documents.extend(documents)

        print(f"Building BM25 retriever with {len(self.all_documents)} chunks...")
        bm25 = BM25Retriever.from_documents(self.all_documents)
        bm25.k = 8

        faiss_ret = self.vectorstore_manager.get_retriever(search_kwargs={"k": 8})

        self.retriever = HybridCustomRetriever(
            faiss_retriever=faiss_ret,
            bm25_retriever=bm25,
            k=10
        )
        print("HybridCustomRetriever built successfully.")

    def get_retriever(self) -> HybridCustomRetriever:
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call build_ensemble_retriever() first.")
        return self.retriever

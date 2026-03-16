from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFace sentence-transformer embeddings model.
    The default is all-MiniLM-L6-v2 as requested.
    
    Args:
        model_name (str): The name of the sentence-transformer model to load.
        
    Returns:
        HuggingFaceEmbeddings: The initialized embeddings instance.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}, # Use CPU for general compatibility; change to cuda or mps if available
            encode_kwargs={'normalize_embeddings': False}
        )
        return embeddings
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        raise e

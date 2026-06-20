"""
Text embedding module.
Generates vector embeddings for text chunks using sentence-transformers.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Handles text embedding using a local sentence-transformers model.
    The model is loaded lazily and cached to avoid reloading.
    """
    
    _model: Optional[SentenceTransformer] = None
    _model_name: str = "all-MiniLM-L6-v2"
    _dimension: int = 384
    
    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """Get or load the embedding model (cached after first load)."""
        if cls._model is None:
            cls._model = SentenceTransformer(cls._model_name)
        return cls._model
    
    @classmethod
    def embed_texts(cls, texts: List[str]) -> np.ndarray:
        """
        Convert a list of text chunks into embedding vectors.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dimension)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, cls._dimension)
        
        model = cls.get_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        return embeddings.astype(np.float32)
    
    @classmethod
    def get_dimension(cls) -> int:
        """Return the embedding dimension."""
        return cls._dimension

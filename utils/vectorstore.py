"""
Vector store module.
Manages FAISS index for storing and searching text embeddings.
"""

from typing import List, Tuple, Optional
import numpy as np
import faiss
from .embedder import Embedder


class VectorStore:
    """
    FAISS-based vector store for similarity search on text chunks.
    Supports adding documents and searching by query.
    """
    
    def __init__(self):
        """Initialize an empty vector store."""
        self.index: Optional[faiss.Index] = None
        self.chunks: List[str] = []
        self.dimension = Embedder.get_dimension()
        self.is_built = False
    
    def add_documents(self, chunks: List[str]) -> None:
        """
        Add text chunks and their embeddings to the vector store.
        
        Args:
            chunks: List of text chunks to add
        """
        if not chunks:
            return
        
        # Generate embeddings for all chunks
        embeddings = Embedder.embed_texts(chunks)
        
        if self.index is None:
            # Create new index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine similarity for normalized vectors
        
        # Add embeddings to index
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.is_built = True
    
    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for the most relevant chunks to a query.
        
        Args:
            query: Search query text
            k: Number of top results to return
            
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        if not self.is_built or self.index is None or self.index.ntotal == 0:
            return []
        
        # Embed the query
        query_embedding = Embedder.embed_texts([query])
        
        # Search the index
        k_actual = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k_actual)
        
        # Return results with scores (normalized to 0-1 range)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                # Convert inner product to a 0-1 similarity score
                score = float(dist)  # Already between -1 and 1 due to normalization
                # Shift from [-1, 1] to [0, 1]
                normalized_score = (score + 1) / 2
                results.append((self.chunks[idx], normalized_score))
        
        return results
    
    def clear(self) -> None:
        """Reset the vector store."""
        self.index = None
        self.chunks = []
        self.is_built = False
    
    @property
    def document_count(self) -> int:
        """Return the number of chunks stored."""
        return len(self.chunks)
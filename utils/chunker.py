"""
Text chunking module.
Splits extracted text into manageable chunks for embedding and retrieval.
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks of approximately equal size.
    Chunks are split on sentence boundaries where possible.
    
    Args:
        text: Input text to split
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        
        # Try to find a sentence boundary near the end point
        search_start = max(start + chunk_size // 2, end - chunk_size // 4)
        search_end = min(end + chunk_size // 4, len(text))
        
        # Look for sentence-ending punctuation followed by space or newline
        best_split = -1
        for delimiter in [". ", "! ", "? ", "\n\n"]:
            pos = text.rfind(delimiter, search_start, search_end)
            if pos > best_split:
                best_split = pos + len(delimiter) - 1  # Include the delimiter
        
        if best_split > start:
            end = best_split + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start with overlap
        start = end - chunk_overlap if end > chunk_overlap else end
    
    return chunks
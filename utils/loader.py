"""
PDF text extraction module.
Uses PyPDF to extract text content from uploaded PDF files.
"""

import io
from typing import Optional
from pypdf import PdfReader


def extract_text_from_pdf(pdf_bytes: bytes) -> Optional[str]:
    """
    Extract text from a PDF file given its raw bytes.
    
    Args:
        pdf_bytes: Raw bytes of the PDF file
        
    Returns:
        Extracted text as a string, or None if extraction fails
    """
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        
        pages_text = []
        total_pages = len(reader.pages)
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(text.strip())
        
        if not pages_text:
            return None
        
        return "\n\n".join(pages_text)
    
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")
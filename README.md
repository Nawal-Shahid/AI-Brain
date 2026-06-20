# AI Study Brain

AI Study Brain is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content. The system extracts text from PDFs, splits it into searchable chunks, generates vector embeddings, and uses the Groq API to answer questions with source citations.

## Architecture

The application follows a modular RAG pipeline:

```
PDF Upload -> Text Extraction -> Text Chunking -> Vector Embedding -> FAISS Index -> Similarity Search -> LLM Answer Generation
```

### Modules

| Module | File | Purpose |
|--------|------|---------|
| Loader | `utils/loader.py` | Extracts text from PDF files using PyPDF |
| Chunker | `utils/chunker.py` | Splits extracted text into overlapping chunks at sentence boundaries |
| Embedder | `utils/embedder.py` | Generates vector embeddings using sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | `utils/vectorstore.py` | Stores and searches embeddings using FAISS with cosine similarity |
| LLM Client | `utils/llm.py` | Communicates with the Groq API for question answering |
| App | `app.py` | Streamlit UI orchestrating the full pipeline |

## Features

- **PDF Text Extraction** - Extracts text from uploaded PDF files using PyPDF
- **Intelligent Chunking** - Splits text into configurable chunks with overlapping boundaries at sentence breaks
- **Local Embeddings** - Generates 384-dimensional vector embeddings locally using all-MiniLM-L6-v2 via sentence-transformers
- **FAISS Vector Search** - Stores embeddings in a FAISS index and performs fast similarity search using inner product (cosine similarity for normalized vectors)
- **Two-Stage LLM Pipeline** - Uses a hidden reasoner model followed by a strict formatter model for cleaner, more accurate answers
- **Source Citations** - Displays retrieved source chunks with relevance scores alongside each answer
- **Configurable Parameters** - Adjustable chunk size, chunk overlap, temperature, and number of sources to retrieve

## Prerequisites

- Python 3.9 or higher
- Groq API key (free tier available at https://console.groq.com)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Nawal-Shahid/AI-Brain.git
cd AI-Brain
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure the Groq API key:

Option A - Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_api_key_here
```

Option B - Enter the API key through the Streamlit sidebar when the application launches.

## Usage

1. Start the application:

```bash
streamlit run app.py
```

2. Open the provided URL (typically http://localhost:8501) in your browser.

3. Enter your Groq API key in the sidebar if not already configured.

4. Upload a PDF document using the file uploader.

5. Wait for the processing pipeline to complete:
   - Text extraction
   - Chunking
   - Model loading (first load may take extra time)
   - Vector indexing

6. Ask questions about the document using the chat input at the bottom of the screen.

## Configuration

All adjustable parameters are available in the Streamlit sidebar:

### Response Settings

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Temperature | 0.0 | 0.0 - 1.0 | Lower values produce more factual answers; higher values add creativity |

### Chunking Settings

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Chunk Size | 1000 | 500 - 3000 | Maximum characters per text chunk |
| Overlap | 200 | 0 - 500 | Number of overlapping characters between consecutive chunks |

### Retrieval Settings

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Sources to Retrieve | 3 | 1 - 10 | Number of relevant chunks to retrieve for each query |

## Pipeline Details

### 1. Text Extraction (utils/loader.py)

Uses `pypdf.PdfReader` to extract text from each page of the uploaded PDF. Returns text as a single string with pages separated by double newlines.

### 2. Text Chunking (utils/chunker.py)

Splits text into chunks of approximately equal size. The chunker normalizes whitespace, then iterates through the text attempting to split at sentence boundaries (. ! ?) near the target chunk size to avoid breaking mid-sentence. Consecutive chunks overlap by a configurable number of characters to preserve context across boundaries.

### 3. Embedding Generation (utils/embedder.py)

Uses the `all-MiniLM-L6-v2` model from sentence-transformers to convert text chunks into 384-dimensional vectors. Embeddings are normalized to unit length to enable cosine similarity via inner product. The model is loaded once and cached for reuse across the session.

### 4. Vector Storage & Retrieval (utils/vectorstore.py)

FAISS index using `IndexFlatIP` (inner product) stores the normalized embeddings. On query, the user's question is embedded with the same model, and the index returns the top-k most similar chunks. Inner product scores are normalized from [-1, 1] to [0, 1] for display as relevance percentages.

### 5. Answer Generation (utils/llm.py)

Uses the Groq API with the `qwen/qwen3.6-27b` model in a two-stage pipeline:

**Stage 1 - Reasoner:** A hidden LLM call at temperature 0.3 that can freely reason through the question and context. Its output is discarded and not shown to the user.

**Stage 2 - Formatter:** A strict LLM call at temperature 0.0 that receives both the original context and the hidden reasoning, and outputs only the final answer. Post-processing removes any residual reasoning artifacts, thinking tags, meta-commentary, and step labels.

If no relevant context is found, the response defaults to: "The document doesn't contain information about that."

## Models

| Component | Model | Type |
|-----------|-------|------|
| Text Embeddings | all-MiniLM-L6-v2 | sentence-transformers (run locally) |
| Answer Generation | qwen/qwen3.6-27b | Groq API (requires internet) |
| Embedding Dimension | 384 | - |

## Dependencies

- **streamlit** - Web application framework for the UI
- **pypdf** - PDF text extraction
- **sentence-transformers** - Local text embedding generation
- **faiss-cpu** - Vector similarity search
- **groq** - Groq API client for LLM access
- **python-dotenv** - Environment variable management
- **numpy** - Numerical operations for embedding vectors

## Project Structure

```
AI-Brain/
  app.py                  # Main Streamlit application
  requirements.txt        # Python dependencies
  .env                    # Environment variables (API key)
  .gitignore              # Git ignore rules
  utils/
    loader.py             # PDF text extraction
    chunker.py            # Text chunking logic
    embedder.py           # Vector embedding generation
    vectorstore.py        # FAISS vector store
    llm.py                # Groq API LLM client
```

## Limitations

- PDFs with scanned images or non-selectable text will not yield usable text (OCR is not supported)
- Very large PDFs may take significant time to process due to local embedding generation
- The application requires an active internet connection for Groq API calls
- The sentence-transformers model downloads on first use (approximately 80 MB)
- Only PDF format is supported for document upload

## Developer

Developed by Nawal Shahid.
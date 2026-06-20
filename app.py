import os
import time
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from utils.loader import extract_text_from_pdf
from utils.chunker import chunk_text
from utils.embedder import Embedder
from utils.vectorstore import VectorStore
from utils.llm import LLMClient

st.set_page_config(page_title="AI Study Brain", page_icon=":brain:", layout="wide")

def init():
    for k, v in {
        "vs": None, "file": None, "msgs": [], "busy": False,
        "chunks": 0, "key_ok": False, "llm": None
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init()

@st.cache_resource
def load_model():
    return Embedder.get_model()

def process(uploaded, cs, co):
    try:
        s = st.status("Extracting text from PDF...", expanded=True, state="running")
        text = extract_text_from_pdf(uploaded.getvalue())
        if not text or len(text.strip()) < 20:
            s.update(label="No text found. Try another file.", state="error", expanded=False)
            return None, 0
        s.update(label=f"Extracted {len(text):,} characters", state="complete", expanded=False)

        s = st.status("Splitting into chunks...", expanded=True, state="running")
        chunks = chunk_text(text, chunk_size=cs, chunk_overlap=co)
        n = len(chunks)
        if n == 0:
            s.update(label="No chunks generated.", state="error", expanded=False)
            return None, 0
        s.update(label=f"Created {n} chunks", state="complete", expanded=False)

        s = st.status("Loading AI model... (first load may take a moment)", expanded=True, state="running")
        load_model()
        s.update(label="AI model ready", state="complete", expanded=False)

        vs = VectorStore()
        bar = st.progress(0.0)
        for i in range(0, n, 50):
            vs.add_documents(chunks[i:i+50])
            bar.progress(min((i+50)/n, 1.0))
            time.sleep(0.01)
        bar.empty()
        return vs, n
    except Exception as e:
        st.error(str(e))
        return None, 0

with st.sidebar:
    st.markdown("### AI Study Brain")
    st.caption("Document Q&A with RAG")

    st.divider()

    key = os.getenv("GROQ_API_KEY")
    if key and key != "your_groq_api_key_here":
        st.session_state.key_ok = True
        st.session_state.llm = LLMClient(api_key=key)
        st.success("API connected")
        st.caption("Model: qwen/qwen3.6-27b")
    else:
        st.session_state.key_ok = False
        with st.expander("API Key", expanded=True):
            k = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="gsk_...",
                help="Required to use the AI. Get a free key at console.groq.com"
            )
            if k:
                os.environ["GROQ_API_KEY"] = k
                st.session_state.key_ok = True
                st.session_state.llm = LLMClient(api_key=k)
                st.rerun()

    st.divider()

    with st.expander("Response", expanded=True):
        temp = st.slider(
            "Temperature", 0.0, 1.0, 0.0, 0.1,
            help="Lower values produce more factual, deterministic answers. Higher values add creativity but may hallucinate."
        )
        st.caption("Lower = factual | Higher = creative")

    with st.expander("Chunking", expanded=False):
        cs = st.slider("Chunk size", 500, 3000, 1000, 100)
        co = st.slider("Overlap", 0, 500, 200, 50)
        st.caption("Adjust if answers seem incomplete or off-topic")

    with st.expander("Retrieval", expanded=False):
        kc = st.slider("Sources to retrieve", 1, 10, 3)

    if st.session_state.file:
        st.divider()
        st.markdown(f"**Document:** {st.session_state.file}")
        st.markdown(f"**Chunks:** {st.session_state.chunks}")
        if st.button("Clear document", use_container_width=True, type="secondary"):
            for k in ["vs", "file", "msgs", "chunks"]:
                st.session_state[k] = [] if k == "msgs" else (0 if k == "chunks" else None)
            st.rerun()

    st.divider()
    st.caption("Groq qwen/qwen3.6-27b")
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.divider()
    st.caption("Developed by Nawal Shahid")

st.title("AI Study Brain")
st.markdown("##### Upload a PDF and ask questions about its content.")
st.caption("The system extracts text, splits it into searchable chunks, and uses AI to answer your questions with source citations.")

if not st.session_state.key_ok:
    st.info("Enter your Groq API key in the sidebar to get started.")
    with st.expander("How it works"):
        st.markdown("""
        1. **Get an API key** from [console.groq.com](https://console.groq.com) (free)
        2. **Paste it** in the sidebar
        3. **Upload a PDF** below
        4. **Ask questions** about the document
        """)
    st.stop()

if not st.session_state.vs:
    st.info("Upload a PDF to begin.")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        with st.spinner("Processing..."):
            vs, n = process(uploaded, cs, co)
        if vs:
            st.session_state.vs = vs
            st.session_state.file = uploaded.name
            st.session_state.chunks = n
            st.session_state.msgs = []
            st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload PDF**")
        st.caption("PDF only. Text is extracted using PyPDF.")
    with col2:
        st.markdown("**2. Chunk & Index**")
        st.caption(f"Split into pieces of {cs} chars, stored in FAISS vector index.")
    with col3:
        st.markdown("**3. Ask Questions**")
        st.caption("Relevant chunks are retrieved & sent to Groq AI for answers.")
    st.stop()

st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"**{st.session_state.file}**")
with col2:
    st.markdown(f"{st.session_state.chunks} chunks")

st.divider()

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "src" in m and m["src"]:
            with st.expander("View sources"):
                for i, (t, s) in enumerate(m["src"], 1):
                    st.caption(f"Source {i} (Relevance: {s:.0%})")
                    st.code(t[:300] + ("..." if len(t) > 300 else ""), language="text")

q = st.chat_input("Ask a question about the document...")
if q:
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        r = st.empty()
        r.markdown("Let me look that up...")
        try:
            results = st.session_state.vs.search(q, k=kc)
            if not results:
                ans = "No relevant information found in the document to answer this question."
                r.markdown(ans)
                srcs = []
            else:
                ans = st.session_state.llm.generate_answer(q, results, temp)
                r.markdown(ans)
                srcs = [(t, sc) for t, sc in results]
                with st.expander("View sources"):
                    for i, (t, sc) in enumerate(srcs, 1):
                        st.caption(f"Source {i} (Relevance: {sc:.0%})")
                        st.code(t[:300] + ("..." if len(t) > 300 else ""), language="text")
            st.session_state.msgs.append({"role": "assistant", "content": ans, "src": srcs if results else []})
        except Exception as e:
            r.error(str(e))
            st.session_state.msgs.append({"role": "assistant", "content": f"Error: {e}", "src": []})
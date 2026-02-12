"""
ArXiv RAG — Streamlit Web Interface
====================================
Interactive UI for querying research papers using the RAG pipeline.

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st

from config import (
    FAISS_INDEX_DIR, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME, LLM_PROVIDER,
)
from rag_pipeline import EmbeddingEngine, VectorStore, RAGPipeline


# ──────────────────────────── Page Config ────────────────────────────

st.set_page_config(
    page_title="ArXiv RAG — Research Paper Q&A",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Custom CSS ────────────────────────────

st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Result cards */
    .source-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a3a5c;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(100, 150, 255, 0.15);
    }
    
    .score-badge {
        display: inline-block;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .paper-title {
        color: #a8c7fa;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .chunk-text {
        color: #c9d1d9;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .main-header h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
    }
    
    /* Stats row */
    .stat-box {
        background: #1a1a2e;
        border: 1px solid #2a3a5c;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────── Load Pipeline ────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the embedding engine and FAISS index (cached)."""
    engine = EmbeddingEngine()
    store = VectorStore(dimension=engine.dimension)
    
    index_path = FAISS_INDEX_DIR / "index.faiss"
    if not index_path.exists():
        return None, None, None
    
    store.load(FAISS_INDEX_DIR)
    llm = RAGPipeline.create_llm()
    return engine, store, llm


# ──────────────────────────── Sidebar ────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    top_k = st.slider(
        "Number of sources (top-k)",
        min_value=1, max_value=20, value=TOP_K,
        help="Number of relevant chunks to retrieve per query",
    )
    
    st.markdown("---")
    st.markdown("### 📐 Pipeline Config")
    st.markdown(f"""
    | Setting | Value |
    |---------|-------|
    | Chunk Size | `{CHUNK_SIZE}` chars |
    | Chunk Overlap | `{CHUNK_OVERLAP}` chars |
    | Embedding | `{EMBEDDING_MODEL_NAME}` |
    | LLM | `{LLM_PROVIDER}` |
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Sample Queries")
    sample_queries = [
        "What are transformer architectures?",
        "How does self-attention work?",
        "Compare CNNs and transformers",
        "Latest advances in NLP",
        "Reinforcement learning applications",
    ]
    for q in sample_queries:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state["query_input"] = q
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#8b949e; font-size:0.8rem;'>"
        "Built with 🔬 ArXiv + FAISS + LangChain"
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────── Main Content ────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🔬 ArXiv Research Paper RAG</h1>
    <p style="color: #8b949e; font-size: 1.1rem;">
        Ask questions about AI/ML research papers — powered by semantic search & LLM generation
    </p>
</div>
""", unsafe_allow_html=True)

# Load pipeline
with st.spinner("🔄 Loading pipeline..."):
    engine, store, llm = load_pipeline()

if engine is None or store is None:
    st.error(
        "⚠️ **FAISS index not found.** "
        "Run the notebook or build the index first:\n\n"
        "```bash\n"
        "python src/collect_papers.py\n"
        "# Then run the notebook cells 1-6\n"
        "```"
    )
    st.stop()

# Index stats
stats = store.get_stats()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{stats['total_vectors']:,}</div>
        <div class="stat-label">Indexed Chunks</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{stats['unique_papers']}</div>
        <div class="stat-label">Research Papers</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{stats['dimension']}</div>
        <div class="stat-label">Embedding Dims</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{top_k}</div>
        <div class="stat-label">Top-K Sources</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Build pipeline
rag = RAGPipeline(
    embedding_engine=engine,
    vector_store=store,
    llm=llm,
    top_k=top_k,
)

# Query input
query = st.text_input(
    "🔍 Ask a research question",
    value=st.session_state.get("query_input", ""),
    placeholder="e.g., What are the key innovations in transformer architectures?",
    key="query_box",
)

if query:
    with st.spinner("🔄 Searching and generating answer..."):
        result = rag.query(query, top_k=top_k)
    
    # Answer section
    st.markdown("### 💬 Answer")
    st.markdown(result["answer"])
    
    # Sources section
    st.markdown(f"### 📚 Sources ({result['num_sources']})")
    
    for i, source in enumerate(result["sources"], 1):
        score_pct = source["score"] * 100
        
        with st.expander(
            f"**[{i}]** {source['paper_title'][:80]} — Score: {source['score']:.4f}",
            expanded=(i <= 2),
        ):
            st.markdown(f"""
            <div class="source-card">
                <span class="score-badge">Score: {source['score']:.4f} ({score_pct:.1f}%)</span>
                <div class="paper-title">📄 {source['paper_title']}</div>
                <div style="color: #8b949e; font-size: 0.8rem; margin-top: 0.3rem;">
                    Paper ID: {source['paper_id']} · 
                    Chunk {source['chunk_index']+1}/{source['total_chunks']}
                </div>
                <div class="chunk-text">{source['text'][:500]}{'...' if len(source['text']) > 500 else ''}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Full text toggle
            if len(source["text"]) > 500:
                if st.checkbox(f"Show full text", key=f"full_{i}"):
                    st.text(source["text"])

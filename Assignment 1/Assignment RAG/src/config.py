"""
Centralized configuration for the ArXiv RAG Pipeline.
Adjust these settings to experiment with different parameters.
"""

import os
from pathlib import Path

# ──────────────────────────── Paths ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
METADATA_FILE = DATA_DIR / "papers_metadata.json"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# Create directories if they don't exist
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Data Collection ────────────────────────────
ARXIV_QUERY = "cat:cs.AI OR cat:cs.CL OR cat:cs.CV"   # ArXiv search query
MAX_PAPERS = 75                                         # Number of papers to download
ARXIV_SORT_BY = "submittedDate"                         # Sort criterion

# ──────────────────────────── Chunking ────────────────────────────
CHUNK_SIZE = 512          # Tokens per chunk
CHUNK_OVERLAP = 50        # Overlap between consecutive chunks
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # Split hierarchy

# ──────────────────────────── Embedding ────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384   # Output dimension of all-MiniLM-L6-v2

# ──────────────────────────── Retrieval ────────────────────────────
TOP_K = 5                  # Number of chunks to retrieve per query
SIMILARITY_METRIC = "cosine"  # cosine | inner_product | l2

# ──────────────────────────── LLM (Generation) ────────────────────────────
# Option 1: OpenAI  (set OPENAI_API_KEY in .env)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")       # "openai" | "ollama"
OPENAI_MODEL = "gpt-3.5-turbo"

# Option 2: Local Ollama
OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434"

# ──────────────────────────── Prompt Template ────────────────────────────
RAG_PROMPT_TEMPLATE = """You are a research assistant specializing in AI/ML.
Use ONLY the following retrieved context from academic papers to answer the question.
If the context does not contain enough information, say so — do not make things up.

Context:
{context}

Question: {question}

Answer (cite the source paper when possible):"""

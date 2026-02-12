"""
RAG Pipeline Module
===================
End-to-end Retrieval-Augmented Generation pipeline for research papers.

Components:
    1. PDFTextExtractor  — Extract text from PDF files
    2. TextChunker       — Split text into overlapping chunks
    3. EmbeddingEngine   — Generate dense vector embeddings
    4. VectorStore       — FAISS-based similarity search index
    5. RAGPipeline       — Orchestrates retrieval + generation
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

try:
    from config import (
        PAPERS_DIR, METADATA_FILE, FAISS_INDEX_DIR,
        CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS,
        EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION,
        TOP_K, RAG_PROMPT_TEMPLATE,
        LLM_PROVIDER, OPENAI_MODEL, OLLAMA_MODEL, OLLAMA_BASE_URL,
    )
except ImportError:
    from src.config import (
        PAPERS_DIR, METADATA_FILE, FAISS_INDEX_DIR,
        CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS,
        EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION,
        TOP_K, RAG_PROMPT_TEMPLATE,
        LLM_PROVIDER, OPENAI_MODEL, OLLAMA_MODEL, OLLAMA_BASE_URL,
    )

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. PDF Text Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PDFTextExtractor:
    """Extract text content from PDF research papers."""

    @staticmethod
    def extract_from_file(pdf_path: str | Path) -> str:
        """
        Extract all text from a single PDF file.
        Uses pdfplumber as primary, falls back to PyPDF2.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            return ""

        # Try pdfplumber first (better for academic PDFs with tables/columns)
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            text = "\n\n".join(text_parts)
            if text.strip():
                return text
        except Exception as e:
            logger.debug(f"pdfplumber failed for {pdf_path.name}: {e}")

        # Fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(pdf_path))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.warning(f"All extractors failed for {pdf_path.name}: {e}")
            return ""

    @classmethod
    def extract_all(
        cls,
        papers_dir: Path = PAPERS_DIR,
        metadata_file: Path = METADATA_FILE,
    ) -> list[dict]:
        """
        Extract text from all downloaded papers.

        Returns
        -------
        list[dict]
            Each dict has: id, title, authors, abstract, text, local_path
        """
        # Load metadata
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}\n"
                "Run collect_papers.py first."
            )

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)

        documents = []
        for meta in tqdm(metadata_list, desc="Extracting text from PDFs"):
            pdf_path = Path(meta["local_path"])
            text = cls.extract_from_file(pdf_path)

            if not text.strip():
                logger.warning(f"No text extracted: {meta['title']}")
                continue

            documents.append({
                "id": meta["id"],
                "title": meta["title"],
                "authors": meta["authors"],
                "abstract": meta.get("abstract", ""),
                "categories": meta.get("categories", []),
                "published": meta.get("published", ""),
                "text": text,
                "local_path": str(pdf_path),
            })

        logger.info(f"Extracted text from {len(documents)}/{len(metadata_list)} papers")
        return documents


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Text Chunking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TextChunker:
    """
    Split documents into overlapping chunks for embedding.

    Strategy: RecursiveCharacterTextSplitter with 512-token chunks
    and 50-token overlap.

    Why 512 tokens?
        - Academic paragraphs average 150–300 words (~200–400 tokens)
        - 512 tokens captures 1–2 complete paragraphs
        - Large enough for context, small enough for precise retrieval

    Why 50-token overlap?
        - ~10% of chunk size prevents boundary information loss
        - Ensures sentences split across chunks are recoverable
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: list[str] = SEPARATORS,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,      # Character-based (approx tokens × 4)
            is_separator_regex=False,
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """
        Split a list of documents into chunks, preserving metadata.

        Parameters
        ----------
        documents : list[dict]
            Each dict must have 'text', 'id', and 'title' keys.

        Returns
        -------
        list[dict]
            Chunks with keys: chunk_id, text, paper_id, paper_title,
            authors, chunk_index
        """
        all_chunks = []
        for doc in tqdm(documents, desc="Chunking documents"):
            text = doc["text"]
            splits = self.splitter.split_text(text)

            for i, chunk_text in enumerate(splits):
                all_chunks.append({
                    "chunk_id": f"{doc['id']}_chunk_{i}",
                    "text": chunk_text,
                    "paper_id": doc["id"],
                    "paper_title": doc["title"],
                    "authors": doc.get("authors", []),
                    "categories": doc.get("categories", []),
                    "chunk_index": i,
                    "total_chunks": len(splits),
                })

        logger.info(
            f"Created {len(all_chunks)} chunks from {len(documents)} documents "
            f"(avg {len(all_chunks)/max(len(documents),1):.1f} chunks/doc)"
        )
        return all_chunks

    def get_stats(self, chunks: list[dict]) -> dict:
        """Return chunking statistics."""
        lengths = [len(c["text"]) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": np.mean(lengths),
            "min_chunk_length": np.min(lengths),
            "max_chunk_length": np.max(lengths),
            "std_chunk_length": np.std(lengths),
            "chunk_size_setting": self.chunk_size,
            "chunk_overlap_setting": self.chunk_overlap,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Embedding Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EmbeddingEngine:
    """
    Generate dense vector embeddings using sentence-transformers.

    Model: all-MiniLM-L6-v2
        - 384 dimensions, 80M parameters, ~14 MB
        - Trained on 1B+ sentence pairs
        - Best quality/speed ratio for semantic similarity
        - Maps sentences to a 384-dimensional dense vector space
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded — dimension: {self.dimension}")

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into dense vector embeddings.

        Parameters
        ----------
        texts : list[str]
            Text strings to embed.
        batch_size : int
            Batch size for encoding.
        show_progress : bool
            Show progress bar.

        Returns
        -------
        np.ndarray
            Array of shape (n_texts, embedding_dim).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2-normalize for cosine similarity
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_texts([query], show_progress=False)[0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Vector Store (FAISS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VectorStore:
    """
    FAISS-backed vector store for semantic search.

    Why FAISS?
        - Open-source (Meta AI), no external server required
        - Extremely fast approximate nearest-neighbor search
        - Handles millions of vectors in-memory
        - Perfect for assignment-scale datasets (5K–50K chunks)

    Alternatives: ChromaDB, Pinecone, Weaviate, Qdrant
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.chunks: list[dict] = []

    def build_index(
        self,
        embeddings: np.ndarray,
        chunks: list[dict],
    ) -> None:
        """
        Build a FAISS index from embeddings.

        Uses IndexFlatIP (inner product) since embeddings are L2-normalized,
        making inner product equivalent to cosine similarity.
        """
        assert embeddings.shape[0] == len(chunks), \
            f"Mismatch: {embeddings.shape[0]} embeddings vs {len(chunks)} chunks"

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunks = chunks

        logger.info(
            f"FAISS index built — {self.index.ntotal} vectors, "
            f"{self.dimension} dimensions"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Search for the most similar chunks to a query embedding.

        Returns
        -------
        list[dict]
            Each result has: chunk metadata + 'score' (cosine similarity)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = {**self.chunks[idx]}
            result["score"] = float(score)
            results.append(result)

        return results

    def save(self, save_dir: Path = FAISS_INDEX_DIR) -> None:
        """Save the FAISS index and chunk metadata to disk."""
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        with open(save_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"Index saved to {save_dir}")

    def load(self, save_dir: Path = FAISS_INDEX_DIR) -> None:
        """Load a previously saved FAISS index and metadata."""
        index_path = save_dir / "index.faiss"
        chunks_path = save_dir / "chunks.pkl"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Index files not found in {save_dir}. "
                "Build the index first."
            )

        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self.dimension = self.index.d
        logger.info(
            f"Loaded FAISS index — {self.index.ntotal} vectors, "
            f"{self.dimension} dimensions"
        )

    def get_stats(self) -> dict:
        """Return index statistics."""
        if self.index is None:
            return {"status": "not_built"}

        unique_papers = set(c["paper_id"] for c in self.chunks)
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "unique_papers": len(unique_papers),
            "total_chunks": len(self.chunks),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. RAG Pipeline (Orchestrator)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RAGPipeline:
    """
    End-to-end RAG pipeline: Retrieval → Context Assembly → Generation.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        llm=None,
        top_k: int = TOP_K,
        prompt_template: str = RAG_PROMPT_TEMPLATE,
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.prompt_template = prompt_template

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """Retrieve relevant chunks for a query."""
        k = top_k or self.top_k
        query_embedding = self.embedding_engine.embed_query(query)
        return self.vector_store.search(query_embedding, top_k=k)

    def build_context(self, results: list[dict]) -> str:
        """Assemble retrieved chunks into a context string."""
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: \"{r['paper_title']}\" "
                f"(Score: {r['score']:.4f})]\n{r['text']}"
            )
        return "\n\n---\n\n".join(context_parts)

    def generate(self, query: str, context: str) -> str:
        """Generate an answer using the LLM with retrieved context."""
        prompt = self.prompt_template.format(
            context=context, question=query
        )

        if self.llm is None:
            # Return raw context if no LLM is configured
            return (
                f"🔍 **Retrieved Context for:** {query}\n\n"
                f"{context}\n\n"
                "⚠️ No LLM configured. Set up OpenAI or Ollama to enable "
                "generated answers.\n"
                "See config.py for LLM_PROVIDER settings."
            )

        try:
            response = self.llm.invoke(prompt)
            # Handle both string and AIMessage responses
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Generation error: {e}\n\nRetrieved context:\n{context}"

    def query(self, question: str, top_k: int = None) -> dict:
        """
        Full RAG pipeline: retrieve → build context → generate.

        Returns
        -------
        dict
            Keys: question, answer, sources, num_sources
        """
        results = self.retrieve(question, top_k)
        context = self.build_context(results)
        answer = self.generate(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": results,
            "num_sources": len(results),
        }

    @staticmethod
    def create_llm(provider: str = LLM_PROVIDER):
        """
        Factory method to create an LLM instance.

        Supports:
            - 'openai': Uses OpenAI API (requires OPENAI_API_KEY env var)
            - 'ollama': Uses local Ollama server
        """
        if provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=OPENAI_MODEL,
                    temperature=0.3,
                )
            except Exception as e:
                logger.warning(f"OpenAI LLM init failed: {e}")
                return None

        elif provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
                return ChatOllama(
                    model=OLLAMA_MODEL,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0.3,
                )
            except Exception as e:
                logger.warning(f"Ollama LLM init failed: {e}")
                return None

        else:
            logger.warning(f"Unknown LLM provider: {provider}")
            return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_full_pipeline(
    papers_dir: Path = PAPERS_DIR,
    metadata_file: Path = METADATA_FILE,
    save_index: bool = True,
) -> RAGPipeline:
    """
    Build the complete RAG pipeline from scratch.

    Steps: Extract PDFs → Chunk → Embed → Index → Wire LLM → Return pipeline
    """
    # 1. Extract text
    extractor = PDFTextExtractor()
    documents = extractor.extract_all(papers_dir, metadata_file)

    # 2. Chunk
    chunker = TextChunker()
    chunks = chunker.chunk_documents(documents)

    # 3. Embed
    engine = EmbeddingEngine()
    texts = [c["text"] for c in chunks]
    embeddings = engine.embed_texts(texts)

    # 4. Index
    store = VectorStore(dimension=engine.dimension)
    store.build_index(embeddings, chunks)

    if save_index:
        store.save()

    # 5. LLM
    llm = RAGPipeline.create_llm()

    # 6. Assemble
    pipeline = RAGPipeline(
        embedding_engine=engine,
        vector_store=store,
        llm=llm,
    )

    logger.info("✅ Full RAG pipeline built successfully!")
    return pipeline


def load_pipeline(index_dir: Path = FAISS_INDEX_DIR) -> RAGPipeline:
    """Load a previously built pipeline from saved index."""
    engine = EmbeddingEngine()

    store = VectorStore(dimension=engine.dimension)
    store.load(index_dir)

    llm = RAGPipeline.create_llm()

    return RAGPipeline(
        embedding_engine=engine,
        vector_store=store,
        llm=llm,
    )

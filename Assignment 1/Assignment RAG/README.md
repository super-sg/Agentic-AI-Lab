# 🔬 ArXiv Research Paper RAG System

A complete **Retrieval-Augmented Generation (RAG)** pipeline that indexes AI/ML research papers from ArXiv and answers research questions using semantic search + LLM-powered synthesis.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-red?logo=streamlit)

---

## 🏗️ Architecture

```
📄 PDFs → 📝 Text Extraction → ✂️ Chunking → 🧮 Embedding → 📦 FAISS Index
                                                                    ↓
                        🤖 LLM Answer ← 📋 Context ← 🔍 Retrieval ← ❓ Query
```

| Component | Technology | Details |
|-----------|-----------|---------|
| **Data Collection** | ArXiv API | 50–100 AI/ML papers (cs.AI, cs.CL, cs.CV) |
| **PDF Processing** | pdfplumber + PyPDF2 | Dual-extractor with fallback |
| **Chunking** | RecursiveCharacterTextSplitter | 512 chars, 50 overlap |
| **Embedding** | `all-MiniLM-L6-v2` | 384-dim, L2-normalized, ~14 MB |
| **Vector Store** | FAISS (`IndexFlatIP`) | Exact cosine similarity search |
| **Generation** | OpenAI GPT-3.5 / Ollama | Configurable cloud or local LLM |
| **Web UI** | Streamlit | Interactive query interface |

> See [`docs/architecture.md`](docs/architecture.md) for detailed Mermaid diagrams and design decisions.

---

## 📁 Project Structure

```
Assignment RAG/
├── README.md                          # This file
├── requirements.txt                   # All dependencies
├── timeline.md                        # 4-week implementation timeline
├── .gitignore
├── src/
│   ├── config.py                      # Centralized configuration
│   ├── collect_papers.py              # ArXiv API data collection
│   └── rag_pipeline.py                # Core RAG pipeline module
├── notebooks/
│   └── rag_pipeline.ipynb             # 12-cell step-by-step notebook
├── app/
│   └── streamlit_app.py               # Web UI
├── docs/
│   └── architecture.md                # Architecture documentation
└── data/                              # (gitignored)
    ├── papers/                        # Downloaded PDFs
    ├── papers_metadata.json           # Paper metadata
    └── faiss_index/                   # Saved FAISS index
```

---

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd "Assignment RAG"

# Create virtual environment
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Papers

```bash
python src/collect_papers.py                            # Default: 75 papers
python src/collect_papers.py --max-papers 50             # Custom count
python src/collect_papers.py --query "transformer"       # Custom query
```

### 3. Run the Notebook

```bash
jupyter notebook notebooks/rag_pipeline.ipynb
```

Run all 12 cells sequentially to:
- Extract text from PDFs
- Chunk documents (512 chars, 50 overlap)
- Generate embeddings (all-MiniLM-L6-v2)
- Build and save the FAISS index
- Test retrieval and RAG generation
- Visualize embeddings (t-SNE) and evaluate metrics

### 4. Launch the Web UI

```bash
streamlit run app/streamlit_app.py
```

### 5. (Optional) Configure LLM

Create a `.env` file in the project root:

```env
# Option 1: OpenAI
OPENAI_API_KEY=sk-your-api-key
LLM_PROVIDER=openai

# Option 2: Local Ollama
# LLM_PROVIDER=ollama
```

If no LLM is configured, the system returns the raw retrieved passages.

---

## 🔍 Test Queries

Try these example queries to test the system:

| Category | Query |
|----------|-------|
| **Architecture** | "What are the key innovations in transformer architectures?" |
| **Comparison** | "How do CNNs compare to transformers for image tasks?" |
| **Methodology** | "What training techniques improve model generalization?" |
| **Application** | "What are the real-world applications of reinforcement learning?" |
| **Recent Work** | "What are the latest developments in large language models?" |

---

## 📊 Key Design Decisions

| Decision | Choice | Justification |
|----------|--------|---------------|
| Chunk size | 512 chars, 50 overlap | Academic paragraphs avg 150–300 words; captures 1–2 complete paragraphs without dilution |
| Embedding model | `all-MiniLM-L6-v2` | Best quality/speed ratio — 384d, 14 MB, trained on 1B+ sentence pairs |
| Vector DB | FAISS `IndexFlatIP` | Open-source, no server, exact cosine search. Perfect for 5K–50K chunks |
| Chunking method | `RecursiveCharacterTextSplitter` | Respects paragraph → sentence → word hierarchy |
| PDF extraction | pdfplumber → PyPDF2 | Dual approach handles both complex and simple layouts |

---

## 🔮 Future Improvements

- **Hybrid Search** — Combine dense (semantic) + sparse (BM25) retrieval
- **Re-ranking** — Cross-encoder re-ranker on top-k results (ms-marco-MiniLM)
- **Metadata Filtering** — Filter by category, date, or author
- **Multi-Query RAG** — Generate query variations for broader recall
- **RAGAS Evaluation** — Automated RAG evaluation metrics
- **Streaming** — Stream LLM responses for better UX
- **Citation Linking** — Link answers to specific pages in source PDFs

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|----------|
| Data Collection | `arxiv` |
| PDF Processing | `pdfplumber`, `PyPDF2` |
| Text Splitting | `langchain-text-splitters` |
| Embeddings | `sentence-transformers` |
| Vector Store | `faiss-cpu` |
| LLM | `langchain-openai`, `langchain-ollama` |
| Visualization | `matplotlib`, `seaborn`, `scikit-learn` |
| Web UI | `streamlit` |
| Notebook | `jupyter`, `ipywidgets` |

---

## 📄 License

This project is for educational/assignment purposes.

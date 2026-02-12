# 📅 Implementation Timeline — ArXiv RAG System

## 4-Week Plan

### Week 1: Foundation & Data Collection
- [x] Define project structure and architecture
- [x] Set up repository with `.gitignore`, `requirements.txt`
- [x] Create `src/config.py` — centralized configuration
- [x] Build `src/collect_papers.py` — ArXiv API download script
- [ ] Download 50–100 papers from ArXiv (cs.AI, cs.CL, cs.CV)
- [ ] Verify PDFs and metadata quality

### Week 2: Core RAG Pipeline
- [x] Build `src/rag_pipeline.py` with 5 components:
  - [x] `PDFTextExtractor` — dual extractor (pdfplumber + PyPDF2)
  - [x] `TextChunker` — RecursiveCharacterTextSplitter (512/50)
  - [x] `EmbeddingEngine` — all-MiniLM-L6-v2 (384d)
  - [x] `VectorStore` — FAISS IndexFlatIP with save/load
  - [x] `RAGPipeline` — retrieval + context assembly + generation
- [x] Create `notebooks/rag_pipeline.ipynb` (12 cells)
- [ ] Run end-to-end pipeline on downloaded papers
- [ ] Evaluate retrieval quality and tune parameters

### Week 3: UI & Visualization
- [x] Build `app/streamlit_app.py` — interactive web interface
- [ ] Add visualization charts to notebook (t-SNE, heatmaps)
- [ ] Test Streamlit app with live FAISS index
- [ ] Polish UI: error handling, loading states, responsive layout

### Week 4: Documentation & Polish
- [x] Write `docs/architecture.md` with Mermaid diagrams
- [x] Write `README.md` with setup, usage, design decisions
- [ ] Final testing: all notebook cells run end-to-end
- [ ] Record demo / take screenshots for documentation
- [ ] Code review and cleanup

---

## Deliverables Checklist

| # | Deliverable | Status |
|---|------------|--------|
| 1 | Data collection script (50+ papers) | ✅ Code complete |
| 2 | RAG pipeline module (5 components) | ✅ Code complete |
| 3 | Jupyter notebook (12 cells) | ✅ Code complete |
| 4 | Streamlit web UI | ✅ Code complete |
| 5 | Architecture documentation | ✅ Complete |
| 6 | README with professional structure | ✅ Complete |
| 7 | End-to-end pipeline verification | ⏳ Pending (needs paper download) |

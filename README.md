# Adaptive Multimodal RAG with Query Analysis and Self-Reflection

This project implements an adaptive RAG application with:
- **Frontend UI** (`frontend/`)
- **Backend API** (`backend/`) using **LangGraph + LangChain**
- **Weaviate** for multi-vector storage/retrieval over multimodal paper chunks

## Architecture
![Adaptive Multimodal RAG Architecture](https://github.com/user-attachments/assets/7af982c9-3ac0-46d0-902f-13a2778c9e30)
![Adaptive RAG Graph Flow](https://github.com/user-attachments/assets/a1d09c7e-103e-4e22-aaea-1bce706b06a7)


## Backend focus (current)
The backend now targets the CV-aligned flow:
1. Ingest research papers from:
   - uploaded PDF files, or
   - arXiv IDs/URLs (downloaded as PDF from arXiv)
2. Extract multimodal content (text, tables, and real extracted images) from PDF.
3. Build multi-vector representations using a shared CLIP embedding space across text and image patches.
4. Retrieve relevant evidence and generate grounded answers with references.

## Repository Structure
- `backend/` — Flask API, adaptive graph, Weaviate vector ops, arXiv ingestion.
- `frontend/` — React client app.

## Run Backend
```bash
cd backend
pip install -r requirements.txt
python api.py
```

## API
### Upload paper(s)
```bash
curl -X POST http://127.0.0.1:5000/upload \
  -F "files=@/path/to/paper.pdf" \
  -F "arxiv_ids=2403.14403" \
  -F "arxiv_ids=https://arxiv.org/abs/2403.14403"
```

### Ask question (grounded with references)
```bash
curl -X POST http://127.0.0.1:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution?", "paper_id": "2403.14403"}'
```

### Delete indexed paper
```bash
curl -X POST http://127.0.0.1:5000/delete \
  -H "Content-Type: application/json" \
  -d '{"paper_id": "2403.14403"}'
```


## Model Configuration
- `CHAT_MODEL` (default: `gpt-5-mini`) for routing, grading, and generation.
- `MM_EMBED_MODEL` (default: `sentence-transformers/clip-ViT-B-32`) for multimodal embeddings.

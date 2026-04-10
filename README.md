# Adaptive Multimodal RAG with Query Analysis and Self-Reflection
Developed application with front-end UI and back-end API using Adaptive RAG, which combines query analysis and active/self-correction RAG. The query analysis routes the LLM call to respond without RAG using web search or invoke single-shot and iterative RAG with self-correction. The architecture was inspired by the research paper **[Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/pdf/2403.14403)**.

## Repository Structure
- `backend/` — Flask API, LangGraph orchestration, utility chains/prompts, Pinecone vector DB operations.
- `frontend/` — React client app (Create React App-style structure).

This project implements an adaptive RAG application with:
- **Frontend UI** (`frontend/`)
- **Backend API** (`backend/`) using **LangGraph + LangChain**
- **Weaviate** for multi-vector storage/retrieval over multimodal paper chunks

## Backend focus (current)
The backend now targets the CV-aligned flow:
1. Ingest research papers from:
   - uploaded PDF files, or
   - arXiv IDs/URLs (downloaded as PDF from arXiv)
2. Extract multimodal content (text, tables, images metadata) from PDF.
3. Build multi-vector representations for each chunk.
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

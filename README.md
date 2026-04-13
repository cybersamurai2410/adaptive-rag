# Adaptive Multimodal RAG with Query Analysis and Self-Reflection

Developed web application with front-end UI and back-end API using advanced adaptive multimodal RAG architecture with query analysis and self-reflection to route LLM calls through multi-vector retrieval over research paper content.

This project implements an adaptive multimodal RAG application with:
- **Frontend UI** (`frontend/`)
- **Backend API** (`backend/`) using **LangGraph + LangChain**
- **Weaviate** for multi-vector storage/retrieval over patch-level page embeddings

## Architecture
![Adaptive Multimodal RAG Architecture](https://github.com/user-attachments/assets/7af982c9-3ac0-46d0-902f-13a2778c9e30)
![Adaptive RAG Graph Flow](https://github.com/user-attachments/assets/a1d09c7e-103e-4e22-aaea-1bce706b06a7)

## Backend overview
The backend pipeline:
1. Ingest research papers from uploaded PDFs or arXiv IDs/URLs.
2. Render each PDF page as an image and keep page text for final answer context.
3. Convert each page image into **patch-level multi-vector embeddings**.
4. Retrieve patch hits with ANN, group by page, and rerank pages with MaxSim.
5. Run adaptive RAG graph with routing, grading, rewrite loop, and grounded answer generation with references.

---

## How multimodal multi-vector embeddings work in this project

### 1) Page-first processing (ColPali style)
For each PDF:
- each page is rendered as a page image
- each page image is encoded into multiple patch vectors
- page text is retained only as answer context (not for retrieval indexing)

Each patch vector stores metadata:
- `paper_id`, `page_id`, `page`, `patch_id`, `reference`.

### 2) Embedding backend (ColPali-first)
The embedding layer uses:
- `COLPALI_MODEL=vidore/colpali-v1.2` (default)

ColPali produces **multiple vectors per input** (token/patch-level style representation) instead of a single pooled vector.

### 3) What “multi-vector” means here
Instead of one vector per page, each page stores:
- `v_1, v_2, ..., v_n` patch vectors
- each patch vector is stored as a separate Weaviate object with shared `page_id`
- raw vector values are additionally persisted in `embedding_json` for exact reranking

### 4) Two-stage retrieval
#### Stage A: ANN candidate generation
For each query vector `q_i`, Weaviate ANN retrieves nearest patch vectors.
Candidate pages are produced by grouping matches on `page_id`.

#### Stage B: Late interaction rerank (MaxSim)
For each candidate page with patch vectors `{p_j}` and query vectors `{q_i}`:

`score(page) = mean_i ( max_j dot(q_i, p_j) )`

This is the MaxSim-style late interaction used for final ranking.

### 5) Inspecting stored multivectors
You can inspect how vectors are actually stored:

```bash
curl http://127.0.0.1:5000/debug/multivector/2403.14403
```

Response includes:
- number of pages
- total patch vectors
- per-page sample with patch vector count + vector dimension

---

## How adaptive RAG architecture works in this project

The LangGraph workflow in `backend/graph_agents.py` runs as a state machine:

1. **Route question** (`paper_rag` vs `web_search`)
   - Router LLM chooses paper retrieval for paper-grounded queries.
2. **Retrieve** from Weaviate
   - Multi-vector retrieval over page patch vectors.
3. **Grade evidence relevance**
   - LLM filters weak/irrelevant retrievals.
4. **Decision**
   - If no relevant evidence: rewrite query.
   - Else: generate answer.
5. **Generate grounded answer**
   - Uses retrieved context and produces references.
6. **Self-reflection checks**
   - Hallucination grader: is answer grounded in evidence?
   - Answer grader: does answer resolve the question?
7. **Loop control**
   - If not grounded or not useful → transform query and retry.
   - If useful → end.

This creates an adaptive loop where retrieval quality and answer quality are actively checked before completion.

---

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


## Example Inputs and Outputs

### 1) Upload paper(s)
**Input**
```bash
curl -X POST http://127.0.0.1:5000/upload \
  -F "files=@/path/to/paper.pdf" \
  -F "arxiv_ids=2403.14403"
```

**Example Output**
```json
{
  "message": "Ingestion complete",
  "results": [
    {
      "source": "paper.pdf",
      "status": "ok",
      "paper_id": "paper",
      "pages": 186,
      "patch_vectors": 1398
    },
    {
      "source": "2403.14403",
      "status": "ok",
      "paper_id": "2403.14403",
      "pages": 224,
      "patch_vectors": 1712
    }
  ]
}
```

### 2) Ask grounded question
**Input**
```bash
curl -X POST http://127.0.0.1:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the routing strategy?","paper_id":"2403.14403"}'
```

**Example Output**
```json
{
  "question": "What is the routing strategy?",
  "paper_id": "2403.14403",
  "answer": "The method first classifies query complexity and routes to the most suitable retrieval/generation path ...",
  "citations": [
    "2403.14403 p.3 [page]",
    "2403.14403 p.5 [page]"
  ]
}
```

### 3) Inspect multivector layout
**Input**
```bash
curl http://127.0.0.1:5000/debug/multivector/2403.14403
```

**Example Output**
```json
{
  "paper_id": "2403.14403",
  "pages": 224,
  "total_patch_vectors": 1712,
  "samples": [
    {
      "page_id": "2403.14403:p3",
      "reference": "2403.14403 p.3 [page]",
      "page": 3,
      "patch_vectors": 128,
      "vector_dim": 128
    }
  ]
}
```

## Model Configuration
- `CHAT_MODEL` is fixed to `gpt-5` for routing, grading, and generation.
- `COLPALI_MODEL` (default: `vidore/colpali-v1.2`) sets the ColPali multimodal embedding model.

# Adaptive Multimodal RAG with Query Analysis and Self-Reflection
Developed application with front-end UI and back-end API using Adaptive RAG, which combines query analysis and active/self-correction RAG. The query analysis routes the LLM call to respond without RAG using web search or invoke single-shot and iterative RAG with self-correction. The architecture was inspired by the research paper **[Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/pdf/2403.14403)**.

## Repository Structure
- `backend/` — Flask API, LangGraph orchestration, utility chains/prompts, Pinecone vector DB operations.
- `frontend/` — React client app (Create React App-style structure).

## Architecture
![tmpn5i8_n2i](https://github.com/user-attachments/assets/7af982c9-3ac0-46d0-902f-13a2778c9e30)
![image](https://github.com/user-attachments/assets/a1d09c7e-103e-4e22-aaea-1bce706b06a7)

## Run Locally

### Backend
```bash
cd backend
python api.py
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## API Request Example
```bash
curl -X POST http://127.0.0.1:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are types of agent memory?"}'
```

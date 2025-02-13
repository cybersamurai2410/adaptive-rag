# Adaptive RAG
Adaptive RAG combines query analysis and active/self-correction RAG. The query analysis routs the LLM call to respond without RAG using web search or invoke single-shot and iterative RAG with self-correction. The architecture was inspired by the research paper **[Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/pdf/2403.14403)**

## Architecture
![image](https://github.com/user-attachments/assets/a1d09c7e-103e-4e22-aaea-1bce706b06a7)

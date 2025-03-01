# Adaptive RAG Web App
Developed application with front-end UI and back-end API using Adaptive RAG, which combines query analysis and active/self-correction RAG. The query analysis routs the LLM call to respond without RAG using web search or invoke single-shot and iterative RAG with self-correction. The architecture was inspired by the research paper **[Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/pdf/2403.14403)**

## Architecture
![tmpn5i8_n2i](https://github.com/user-attachments/assets/7af982c9-3ac0-46d0-902f-13a2778c9e30)
![image](https://github.com/user-attachments/assets/a1d09c7e-103e-4e22-aaea-1bce706b06a7)

## API Requests
```
curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d '{"question": "What are types of agent memory?"}'
```
**Ouput Logs:**
```
---ROUTE QUESTION---
---ROUTE QUESTION TO RAG---
---RETRIEVE---
"Node 'retrieve':"
'\n---\n'
---CHECK DOCUMENT RELEVANCE TO QUESTION---
---GRADE: DOCUMENT NOT RELEVANT---
---GRADE: DOCUMENT RELEVANT---
---GRADE: DOCUMENT NOT RELEVANT---
---GRADE: DOCUMENT RELEVANT---
---ASSESS GRADED DOCUMENTS---
---DECISION: GENERATE---
"Node 'grade_documents':"
'\n---\n'
---GENERATE---
---CHECK HALLUCINATIONS---
---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---
---GRADE GENERATION vs QUESTION---
---DECISION: GENERATION ADDRESSES QUESTION---
"Node 'generate':"
'\n---\n'
('The types of agent memory include short-term memory, long-term memory, and '
 'sensory memory. Short-term memory is utilized for in-context learning, while '
 'long-term memory allows for the retention and recall of information over '
 'extended periods. Sensory memory involves learning embedding representations '
 'for various raw inputs, such as text and images.')
 ```
## React UI
<img width="611" alt="image" src="https://github.com/user-attachments/assets/dfe5337d-cce8-4dfe-8702-be29bd3a139c" />

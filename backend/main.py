import requests
from pprint import pprint
from graph_agents import compile_graph
from vector_db_ops import VectorDB
import langsmith 
from dotenv import load_dotenv
load_dotenv()

def main(app, question):
    inputs = {
        "question": question
    }
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':") 
            pprint(f"Value: {value}") # question, generation, documents 
            # pprint.pprint(value.keys(), indent=2, width=80, depth=None) # Optional: print full state at each node
        pprint("\n---\n")

    # Final generation
    return value["generation"] # generate: {question, generation, documents}

if __name__ == "__main__":
    question = "How do black holes work?"
    vectordb = VectorDB()
    
    docs = [] # List of file paths and urls 
    vectordb.store_vectordb(docs)
    app = compile_graph()
    response = main(app, question)
    print(response)

    # Upload documents 
    url = "http://127.0.0.1:5000/upload"
    files = [("files", open("files/test.pdf", "rb")), ("files", open("files/test.txt", "rb"))]
    data = {"urls": ["https://example.com/article1", "https://example.com/article2"]}
    response = requests.post(url, files=files, data=data)
    print(response.json())

    # Ask question 
    url = "http://127.0.0.1:5000/ask"
    data = {"question": question}
    response = requests.post(url, json=data)
    print(response.json())

"""
Request: 
curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d '{"question": "What was the last UFC event and who won?"}'
npm start - start react app
python api.py - start flask server 
langgraph studio - langgraph dev (config from langgraph.json)
"""

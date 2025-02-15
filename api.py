from flask import Flask, request, jsonify
from flask_cors import CORS
from graph_agents import compile_graph
from dotenv import load_dotenv
import langsmith

load_dotenv()
app = Flask(__name__)  # Initialize Flask app
graph_app = compile_graph()  # Load LangGraph agentic workflow
CORS(app) # Enables React to send requests to Flask 

# api endpoint for store_vectordb

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    inputs = {"question": question}
    response = None

    for output in graph_app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}': {value}")  
        response = value["generation"]  # Get final response

    return jsonify({"question": question, "answer": response}) # Converts final response to json 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Start Flask server accessible on any network 

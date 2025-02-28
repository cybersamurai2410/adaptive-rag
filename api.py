from flask import Flask, request, jsonify
from flask_cors import CORS
from vector_db_ops import VectorDB 
from graph_agents import compile_graph
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)  # Initialize Flask app
graph_app = compile_graph()  # Load LangGraph agentic workflow
CORS(app)  # Enables React to send requests to Flask
vector_db = VectorDB()

UPLOAD_FOLDER = "uploads" # Save files locally since Flask stores temporary file objects and are not accessible when functions require file paths in parameters 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Creates "uploads/" folder if missing
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# API to Process Uploaded Files and URLs
@app.route("/upload", methods=["POST"])
def upload():
    # Get lists from UI 
    urls = request.form.getlist("urls")  
    files = request.files.getlist("files")  

    if not urls and not files:
        return jsonify({"error": "No files or URLs provided"}), 400

    saved_files = []
    try:
        # Save files before sending them to the vector database
        for file in files:
            if file.filename == "":
                continue  # Skip empty files

            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            saved_files.append(file_path)

        # Process URLs and saved files
        docs = urls + saved_files
        vector_db.store_vectordb(docs)

        return jsonify({"message": "Data successfully added to vector database"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Adaptive RAG API
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

    return jsonify({"question": question, "answer": response})  # Converts final response to JSON

# Delete document from Pinecone and JSON
@app.route("/delete", methods=["POST"])
def delete():
    data = request.get_json()
    doc_name = data.get("doc_name")

    if not doc_name:
        return jsonify({"error": "Missing document name"}), 400

    success = vector_db.delete_document_vectordb(doc_name)
    if success:
        return jsonify({"message": f"Deleted {doc_name} successfully"}), 200
    else:
        return jsonify({"error": f"File {doc_name} not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Start Flask server accessible on any network

from flask import Flask, request, jsonify
from flask_cors import CORS
from create_index import store_vectordb
from graph_agents import compile_graph
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)  # Initialize Flask app
graph_app = compile_graph()  # Load LangGraph agentic workflow
CORS(app)  # Enables React to send requests to Flask

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Creates "uploads/" folder if missing
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# API to Process Uploaded Files and URLs
@app.route("/upload", methods=["POST"])
def upload():
    urls = request.form.getlist("urls")  # Get list of URLs
    files = request.files.getlist("files")  # Get uploaded files

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
        store_vectordb(urls + saved_files)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Start Flask server accessible on any network

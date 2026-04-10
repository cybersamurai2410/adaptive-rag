import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from graph_agents import compile_graph
from vector_db_ops import VectorDB, download_arxiv_pdf

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

vector_db = VectorDB()
graph_app = compile_graph()


@app.route("/upload", methods=["POST"])
def upload():
    """
    Upload endpoint for:
    - PDF files (multipart `files`)
    - arXiv IDs/URLs (multipart `arxiv_ids`)
    """
    files = request.files.getlist("files")
    arxiv_ids = request.form.getlist("arxiv_ids")

    if not files and not arxiv_ids:
        return jsonify({"error": "Provide files and/or arxiv_ids"}), 400

    results = []

    # 1) ingest uploaded PDFs
    for file in files:
        if not file or file.filename == "":
            continue
        if not file.filename.lower().endswith(".pdf"):
            results.append({"source": file.filename, "status": "skipped", "reason": "Only PDF is supported"})
            continue

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(save_path)

        try:
            ingest_result = vector_db.store_pdf(save_path)
            results.append({"source": file.filename, "status": "ok", **ingest_result})
        except Exception as e:
            results.append({"source": file.filename, "status": "error", "error": str(e)})

    # 2) ingest arXiv IDs/URLs
    for arxiv_ref in arxiv_ids:
        try:
            pdf_path = download_arxiv_pdf(arxiv_ref, target_dir=app.config["UPLOAD_FOLDER"])
            ingest_result = vector_db.store_pdf(pdf_path)
            results.append({"source": arxiv_ref, "status": "ok", **ingest_result})
        except Exception as e:
            results.append({"source": arxiv_ref, "status": "error", "error": str(e)})

    return jsonify({"message": "Ingestion complete", "results": results}), 200


@app.route("/ask", methods=["POST"])
def ask():
    """
    Ask questions grounded in uploaded papers.
    Optional `paper_id` narrows retrieval to one paper.
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question")
    paper_id = data.get("paper_id")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    inputs = {
        "question": question,
        "paper_id": paper_id,
        "documents": [],
        "generation": "",
        "citations": [],
    }

    final = None
    for output in graph_app.stream(inputs):
        for _, value in output.items():
            final = value

    if not final:
        return jsonify({"error": "No response generated"}), 500

    return jsonify(
        {
            "question": question,
            "paper_id": paper_id,
            "answer": final.get("generation", ""),
            "citations": final.get("citations", []),
        }
    )


@app.route("/delete", methods=["POST"])
def delete():
    data = request.get_json(silent=True) or {}
    paper_id = data.get("paper_id")

    if not paper_id:
        return jsonify({"error": "Missing paper_id"}), 400

    success = vector_db.delete_document_vectordb(paper_id)
    if not success:
        return jsonify({"error": f"paper_id '{paper_id}' not found"}), 404

    return jsonify({"message": f"Deleted {paper_id} successfully"}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "adaptive-multimodal-rag-backend"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

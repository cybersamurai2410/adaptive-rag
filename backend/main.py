"""Simple local client for backend API manual testing."""

import requests

BASE_URL = "http://127.0.0.1:5000"


if __name__ == "__main__":
    # Example: ingest an arXiv paper
    upload_response = requests.post(
        f"{BASE_URL}/upload",
        data={"arxiv_ids": ["2403.14403"]},
    )
    print("UPLOAD:", upload_response.status_code, upload_response.json())

    # Example: ask grounded question
    ask_response = requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": "What is the main idea of the method and how does routing work?",
            "paper_id": "2403.14403",
        },
    )
    print("ASK:", ask_response.status_code, ask_response.json())

const API_BASE_URL = "http://127.0.0.1:5000";

export async function askQuestion(question) {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Failed to fetch answer.");
  }

  return data;
}

export async function uploadData(uploads) {
  const formData = new FormData();

  uploads.forEach((item) => {
    if (item.type === "url") {
      formData.append("arxiv_ids", item.name);
    } else if (item.type === "file") {
      formData.append("files", item.file);
    }
  });

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Failed to process data.");
  }

  return data;
}

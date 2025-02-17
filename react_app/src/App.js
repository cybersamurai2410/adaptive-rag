// import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

// export default App;

import React, { useState } from "react";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploads, setUploads] = useState([]);
  const [urlInput, setUrlInput] = useState("");
  const [processing, setProcessing] = useState(false);
  const [processMessage, setProcessMessage] = useState("");

  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setAnswer("");

    try {
      const response = await fetch("http://127.0.0.1:5000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await response.json();
      setAnswer(data.answer);
    } catch (err) {
      setAnswer("Error fetching answer. Please try again.");
    }

    setLoading(false);
  };

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    // Allow only TXT and PDF files
    const allowedTypes = ["text/plain", "application/pdf"];
    const validFiles = files.filter((file) => allowedTypes.includes(file.type));

    const fileItems = validFiles.map((file) => ({
      name: file.name,
      type: "file",
    }));

    setUploads((prev) => [...prev, ...fileItems]);
  };

  const handleUrlAdd = () => {
    if (urlInput.trim() !== "") {
      setUploads((prev) => [
        ...prev,
        { name: urlInput.trim(), type: "url" },
      ]);
      setUrlInput("");
    }
  };

// Function to send files & URLs to Flask API
const handleProcessData = async () => {
  if (uploads.length === 0) {
      setProcessMessage("No files or URLs to process.");
      return;
  }

  setProcessing(true);
  setProcessMessage("");

  // Prepare form data
  const formData = new FormData();
  uploads.forEach((item) => {
      if (item.type === "url") {
          formData.append("urls", item.name);
      } else if (item.type === "file") {
          formData.append("files", item.file);  // File object
      }
  });

  try {
      const response = await fetch("http://127.0.0.1:5000/upload", {
          method: "POST",
          body: formData,
      });

      const data = await response.json();
      if (response.ok) {
          setProcessMessage("Data successfully added to vector database!");
      } else {
          setProcessMessage("Error: " + (data.error || "Failed to process data."));
      }
  } catch (error) {
      setProcessMessage("Error: Failed to connect to server.");
  }

  setProcessing(false);
};

  return (
    <div className="App">
      <header>
        <h1>Ask a Question</h1>
      </header>

      <section className="question-section">
        <form onSubmit={handleQuestionSubmit} className="question-form">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your question here..."
            rows="4"
            className="question-input"
          />
          <button type="submit" className="submit-button">
            Submit
          </button>
        </form>

        {loading && <p className="loading">Loading...</p>}
        {answer && (
          <div className="answer-section">
            <h2>Answer:</h2>
            <p>{answer}</p>
          </div>
        )}
      </section>

      <section className="upload-section">
        <h2>Upload Files or Add URLs</h2>
        <div className="file-upload">
          <label htmlFor="fileInput">Upload TXT/PDF Files:</label>
          <input
            id="fileInput"
            type="file"
            accept=".txt,.pdf"
            onChange={handleFileUpload}
            multiple
          />
        </div>

        <div className="url-upload">
          <input
            type="text"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            placeholder="Enter a URL"
            className="url-input"
          />
          <button type="button" onClick={handleUrlAdd} className="add-url-button">
            Add URL
          </button>
        </div>
      </section>

      {uploads.length > 0 && (
        <>
        <section className="uploads-list">
          <h3>Uploaded Items:</h3>
          <ul>
            {uploads.map((item, index) => (
              <li key={index}>
                {item.type === "file" ? "📄 File: " : "🔗 URL: "}
                {item.name}
              </li>
            ))}
          </ul>
        </section>

        <div className="process-section">
          <button onClick={handleProcessData} className="process-button" disabled={processing}>
              {processing ? "Processing..." : "Process Data"}
          </button>
          {processMessage && <p className="process-message">{processMessage}</p>}
        </div>
        </>
      )}
    </div>
  );
}

export default App;

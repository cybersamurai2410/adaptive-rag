import React, { useMemo, useState } from "react";
import "./App.css";

const parseCitation = (citation, index) => {
  const raw = typeof citation === "string" ? citation.trim() : String(citation ?? "").trim();

  const pattern = /(?<paperId>(?:\d{4}\.\d{4,5}(?:v\d+)?)|(?:[a-z-]+\/\d{7}(?:v\d+)?))(?:(?:\s|,)+p\.?\s*(?<page>\d+))?(?:\s*\[(?<label>[^\]]+)\])?/i;
  const match = raw.match(pattern);

  if (!match?.groups?.paperId) {
    return {
      id: `citation-${index}`,
      raw,
      paperId: "Unknown paper",
      page: null,
      label: "source",
      display: raw || `Citation ${index + 1}`,
    };
  }

  const paperId = match.groups.paperId;
  const page = match.groups.page ? Number.parseInt(match.groups.page, 10) : null;
  const label = match.groups.label || "source";

  return {
    id: `citation-${index}`,
    raw,
    paperId,
    page: Number.isNaN(page) ? null : page,
    label,
    display: `${paperId}${page ? ` · p.${page}` : ""}`,
  };
};

function ChatPane({
  loading,
  answer,
  citations,
  activeCitation,
  onCitationClick,
}) {
  if (loading) {
    return <p className="loading">Loading...</p>;
  }

  if (!answer) {
    return null;
  }

  return (
    <div className="answer-section">
      <h2>Answer:</h2>
      <p>{answer}</p>

      {citations.length > 0 && (
        <div className="citations-section">
          <h3>Sources</h3>
          <ul className="citation-list">
            {citations.map((citation) => {
              const isActive = activeCitation?.id === citation.id;
              return (
                <li key={citation.id}>
                  <button
                    type="button"
                    className={`citation-chip ${isActive ? "active" : ""}`}
                    onClick={() => onCitationClick(citation)}
                    title={citation.raw}
                  >
                    {citation.display}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

function PaperPane({ activeCitation, activePage }) {
  const selectedText = activeCitation
    ? `${activeCitation.paperId}${activePage ? ` · page ${activePage}` : ""}`
    : "No source selected yet.";

  return (
    <section className="paper-pane">
      <h2>Paper Pane</h2>
      <p className="paper-pane-caption">
        Click a source chip above to jump to that citation.
      </p>
      <div className={`paper-preview ${activeCitation ? "highlighted" : ""}`}>
        <strong>Active citation:</strong> {selectedText}
      </div>
      {activePage && (
        <p className="page-jump-message">
          Jumped to page <strong>{activePage}</strong>. Page-level highlight enabled.
        </p>
      )}
    </section>
  );
}

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploads, setUploads] = useState([]);
  const [urlInput, setUrlInput] = useState("");
  const [processing, setProcessing] = useState(false);
  const [processMessage, setProcessMessage] = useState("");
  const [uploadFeedback, setUploadFeedback] = useState("");
  const [rawCitations, setRawCitations] = useState([]);

  const [activeCitation, setActiveCitation] = useState(null);
  const [activePage, setActivePage] = useState(null);

  const parsedCitations = useMemo(
    () => rawCitations.map((citation, index) => parseCitation(citation, index)),
    [rawCitations]
  );

  const handleCitationClick = (citation) => {
    setActiveCitation(citation);
    setActivePage(citation.page ?? null);
  };

  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setAnswer("");
    setRawCitations([]);
    setActiveCitation(null);
    setActivePage(null);

    try {
      const response = await fetch("http://127.0.0.1:5000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await response.json();
      setAnswer(data.answer || "");
      setRawCitations(Array.isArray(data.citations) ? data.citations : []);
    } catch (err) {
      setAnswer("Error fetching answer. Please try again.");
      setRawCitations([]);
    }

    setLoading(false);
  };

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    const validFiles = files.filter((file) => {
      const hasPdfMimeType = file.type === "application/pdf";
      const hasPdfExtension = file.name.toLowerCase().endsWith(".pdf");
      return hasPdfMimeType || hasPdfExtension;
    });
    const rejectedFiles = files.filter((file) => !validFiles.includes(file));

    const fileItems = validFiles.map((file) => ({
      name: file.name,
      type: "file",
      file,
    }));

    setUploads((prev) => [...prev, ...fileItems]);

    if (rejectedFiles.length > 0) {
      setUploadFeedback(
        `Skipped ${rejectedFiles.length} file(s): PDF files only are supported.`
      );
    } else {
      setUploadFeedback("");
    }
  };

  const isValidArxivInput = (input) => {
    const trimmed = input.trim();
    const arxivUrlPattern = /^(https?:\/\/)?(www\.)?arxiv\.org\/(abs|pdf)\/[a-zA-Z-]*\/?\d{4}\.\d{4,5}(v\d+)?(\.pdf)?$/i;
    const arxivIdPattern = /^(?:[a-zA-Z-]+\/)?\d{4}\.\d{4,5}(v\d+)?$/;

    return arxivUrlPattern.test(trimmed) || arxivIdPattern.test(trimmed);
  };

  const handleUrlAdd = () => {
    const trimmedInput = urlInput.trim();

    if (!trimmedInput) return;

    if (!isValidArxivInput(trimmedInput)) {
      setProcessMessage("Please enter a valid arXiv URL or arXiv ID.");
      return;
    }

    setUploads((prev) => [...prev, { name: trimmedInput, type: "url" }]);
    setProcessMessage("");
    setUrlInput("");
  };

  const handleProcessData = async () => {
    if (uploads.length === 0) {
      setProcessMessage("No files or URLs to process.");
      return;
    }

    setProcessing(true);
    setProcessMessage("");

    const formData = new FormData();
    uploads.forEach((item) => {
      if (item.type === "url") {
        formData.append("arxiv_ids", item.name);
      } else if (item.type === "file") {
        formData.append("files", item.file);
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

        <ChatPane
          loading={loading}
          answer={answer}
          citations={parsedCitations}
          activeCitation={activeCitation}
          onCitationClick={handleCitationClick}
        />
      </section>

      <PaperPane activeCitation={activeCitation} activePage={activePage} />

      <section className="upload-section">
        <h2>Upload Files or Add URLs</h2>
        <div className="file-upload">
          <label htmlFor="fileInput">Upload PDF Files:</label>
          <input
            id="fileInput"
            type="file"
            accept=".pdf,application/pdf"
            onChange={handleFileUpload}
            multiple
          />
          {uploadFeedback && <p className="process-message">{uploadFeedback}</p>}
        </div>

        <div className="url-upload">
          <input
            type="text"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            placeholder="Enter arXiv URL or arXiv ID"
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
            <button
              onClick={handleProcessData}
              className="process-button"
              disabled={processing}
            >
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

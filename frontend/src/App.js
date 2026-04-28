import React, { useMemo, useState } from "react";
import "./App.css";
import PaperPane from "./components/PaperPane";
import ChatPane from "./components/ChatPane";
import { askQuestion, uploadData } from "./api/client";

const ARXIV_ID_PATTERN = /(?:^|\/)((?:\d{4}\.\d{4,5}(?:v\d+)?)|(?:[a-z-]+\/\d{7}(?:v\d+)?))(?:\.pdf)?$/i;

const normalizeArxivId = (value) => {
  const raw = String(value ?? "").trim();
  const match = raw.match(ARXIV_ID_PATTERN);
  return match ? match[1].toLowerCase() : raw.toLowerCase();
};

const parseCitation = (citation, index) => {
  const base = typeof citation === "object" && citation !== null ? citation : { raw: citation };
  const raw = typeof base.raw === "string" ? base.raw.trim() : String(base.raw ?? citation ?? "").trim();

  const pattern = /(?<paperId>(?:\d{4}\.\d{4,5}(?:v\d+)?)|(?:[a-z-]+\/\d{7}(?:v\d+)?))(?:(?:\s|,)+p\.?\s*(?<page>\d+))?(?:\s*\[(?<label>[^\]]+)\])?/i;
  const match = raw.match(pattern);

  const parsedPaperId = base.paperId || match?.groups?.paperId || "Unknown paper";
  const parsedPage = base.page ?? (match?.groups?.page ? Number.parseInt(match.groups.page, 10) : null);

  return {
    id: base.id || `citation-${index}`,
    raw,
    paperId: parsedPaperId,
    normalizedPaperId: normalizeArxivId(parsedPaperId),
    page: Number.isNaN(parsedPage) ? null : parsedPage,
    label: base.label || match?.groups?.label || "source",
    display: base.display || `${parsedPaperId}${parsedPage ? ` · p.${parsedPage}` : ""}`,
    bbox: base.bbox || base.boundingBox || base.coordinates || null,
    snippet: typeof base.snippet === "string" ? base.snippet : "",
  };
};

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);

  const [uploads, setUploads] = useState([]);
  const [urlInput, setUrlInput] = useState("");
  const [processing, setProcessing] = useState(false);
  const [processMessage, setProcessMessage] = useState("");
  const [uploadFeedback, setUploadFeedback] = useState("");

  const [rawCitations, setRawCitations] = useState([]);
  const [activeCitation, setActiveCitation] = useState(null);

  const [selectedPaperIndex, setSelectedPaperIndex] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);

  const parsedCitations = useMemo(
    () => rawCitations.map((citation, index) => parseCitation(citation, index)),
    [rawCitations]
  );

  const handleCitationClick = (citation) => {
    setActiveCitation(citation);

    const matchingPaperIndex = uploads.findIndex((item) => {
      const normalizedName = normalizeArxivId(item.name);
      const normalizedCitation = citation.normalizedPaperId;
      return normalizedName.includes(normalizedCitation) || normalizedCitation.includes(normalizedName);
    });

    if (matchingPaperIndex >= 0) {
      setSelectedPaperIndex(matchingPaperIndex);
    }

    if (citation.page) {
      setCurrentPage(citation.page);
    }
  };

  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) return;

    setLoading(true);
    setAnswer("");
    setRawCitations([]);
    setActiveCitation(null);

    try {
      const data = await askQuestion(trimmedQuestion);
      const nextAnswer = data.answer || "";
      const nextCitations = Array.isArray(data.citations) ? data.citations : [];

      setAnswer(nextAnswer);
      setRawCitations(nextCitations);
      setChatHistory((prev) => [
        ...prev,
        {
          question: trimmedQuestion,
          answer: nextAnswer,
          sources: nextCitations,
        },
      ]);
    } catch (err) {
      setAnswer("Error fetching answer. Please try again.");
      setRawCitations([]);
    } finally {
      setLoading(false);
    }
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
      setUploadFeedback(`Skipped ${rejectedFiles.length} file(s): PDF files only are supported.`);
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

    try {
      await uploadData(uploads);
      setProcessMessage("Data successfully added to vector database!");
    } catch (error) {
      setProcessMessage(`Error: ${error.message || "Failed to process data."}`);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="App">
      <header>
        <h1>Adaptive RAG Workspace</h1>
      </header>

      <div className="app-grid">
        <ChatPane
          question={question}
          setQuestion={setQuestion}
          loading={loading}
          answer={answer}
          citations={parsedCitations}
          activeCitation={activeCitation}
          onCitationClick={handleCitationClick}
          chatHistory={chatHistory}
          onSubmit={handleQuestionSubmit}
        />

        <PaperPane
          uploads={uploads}
          selectedPaperIndex={selectedPaperIndex}
          setSelectedPaperIndex={setSelectedPaperIndex}
          currentPage={currentPage}
          setCurrentPage={setCurrentPage}
          activeCitation={activeCitation}
          urlInput={urlInput}
          setUrlInput={setUrlInput}
          handleFileUpload={handleFileUpload}
          handleUrlAdd={handleUrlAdd}
          handleProcessData={handleProcessData}
          processing={processing}
          processMessage={processMessage}
          uploadFeedback={uploadFeedback}
        />
      </div>
    </div>
  );
}

export default App;

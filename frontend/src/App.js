import React, { useState } from "react";
import "./App.css";
import PaperPane from "./components/PaperPane";
import ChatPane from "./components/ChatPane";
import { askQuestion, uploadData } from "./api/client";

function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);

  const [uploads, setUploads] = useState([]);
  const [urlInput, setUrlInput] = useState("");
  const [processing, setProcessing] = useState(false);
  const [processMessage, setProcessMessage] = useState("");
  const [uploadFeedback, setUploadFeedback] = useState("");

  const [selectedPaperIndex, setSelectedPaperIndex] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);

  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) return;

    setLoading(true);

    try {
      const data = await askQuestion(trimmedQuestion);
      setChatHistory((prev) => [
        ...prev,
        {
          question: trimmedQuestion,
          answer: data.answer || "",
          sources: Array.isArray(data.citations) ? data.citations : [],
        },
      ]);
      setQuestion("");
    } catch (error) {
      setChatHistory((prev) => [
        ...prev,
        {
          question: trimmedQuestion,
          answer: "Error fetching answer. Please try again.",
          sources: [],
        },
      ]);
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

    try {
      await uploadData(uploads);
      setProcessMessage("Data successfully added to vector database!");
    } catch (error) {
      setProcessMessage(`Error: ${error.message || "Failed to process data."}`);
    }
  };

  return (
    <div className="App">
      <header>
        <h1>Adaptive RAG Workspace</h1>
      </header>

      <main className="app-grid">
        <PaperPane
          uploads={uploads}
          selectedPaperIndex={selectedPaperIndex}
          setSelectedPaperIndex={setSelectedPaperIndex}
          currentPage={currentPage}
          setCurrentPage={setCurrentPage}
          urlInput={urlInput}
          setUrlInput={setUrlInput}
          handleFileUpload={handleFileUpload}
          handleUrlAdd={handleUrlAdd}
          handleProcessData={handleProcessData}
          processing={processing}
          processMessage={processMessage}
          uploadFeedback={uploadFeedback}
        />

        <ChatPane
          question={question}
          setQuestion={setQuestion}
          loading={loading}
          chatHistory={chatHistory}
          onSubmit={handleQuestionSubmit}
        />
      </main>
    </div>
  );
}

export default App;

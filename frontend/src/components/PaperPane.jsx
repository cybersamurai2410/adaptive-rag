import React, { useEffect, useMemo, useState } from "react";
import UploadControls from "./UploadControls";

const ARXIV_ID_PATTERN = /(?:^|\/)((?:\d{4}\.\d{4,5}(?:v\d+)?)|(?:[a-z-]+\/\d{7}(?:v\d+)?))(?:\.pdf)?$/i;

const getArxivPdfUrl = (value) => {
  const raw = String(value ?? "").trim();

  if (/^https?:\/\//i.test(raw) && raw.includes("/pdf/")) {
    return raw.endsWith(".pdf") ? raw : `${raw}.pdf`;
  }

  if (/^https?:\/\//i.test(raw) && raw.includes("/abs/")) {
    return `${raw.replace("/abs/", "/pdf/")}.pdf`;
  }

  const match = raw.match(ARXIV_ID_PATTERN);
  const arxivId = match ? match[1] : raw;
  return `https://arxiv.org/pdf/${arxivId}.pdf`;
};

function PaperPane({
  uploads,
  selectedPaperIndex,
  setSelectedPaperIndex,
  currentPage,
  setCurrentPage,
  activeCitation,
  urlInput,
  setUrlInput,
  handleFileUpload,
  handleUrlAdd,
  handleProcessData,
  processing,
  processMessage,
  uploadFeedback,
}) {
  const [localObjectUrl, setLocalObjectUrl] = useState(null);
  const selectedPaper = uploads[selectedPaperIndex];

  useEffect(() => {
    if (!selectedPaper || selectedPaper.type !== "file") {
      setLocalObjectUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedPaper.file);
    setLocalObjectUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [selectedPaper]);

  const viewerUrl = useMemo(() => {
    if (!selectedPaper) return "";

    if (selectedPaper.type === "file") {
      return localObjectUrl ? `${localObjectUrl}#page=${currentPage}` : "";
    }

    const arxivPdfUrl = getArxivPdfUrl(selectedPaper.name);
    return `${arxivPdfUrl}#page=${currentPage}`;
  }, [selectedPaper, localObjectUrl, currentPage]);

  const hasRegionHighlight = Boolean(activeCitation?.bbox);

  return (
    <section className="pane paper-pane">
      <h2>Paper Viewer</h2>

      <div className="paper-selector">
        <label htmlFor="paperSelect">Selected Paper:</label>
        <select
          id="paperSelect"
          value={selectedPaperIndex}
          onChange={(e) => {
            setSelectedPaperIndex(Number(e.target.value));
            setCurrentPage(1);
          }}
          disabled={uploads.length === 0}
        >
          {uploads.length === 0 ? (
            <option value={0}>No papers uploaded yet</option>
          ) : (
            uploads.map((item, index) => (
              <option key={`${item.name}-${index}`} value={index}>
                {item.name}
              </option>
            ))
          )}
        </select>
      </div>

      <div className="paper-viewer">
        {selectedPaper ? (
          <>
            <p>
              <strong>{selectedPaper.name}</strong>
            </p>
            {viewerUrl ? (
              <iframe
                key={`${selectedPaperIndex}-${currentPage}`}
                title={`PDF viewer - ${selectedPaper.name}`}
                src={viewerUrl}
                className="pdf-frame"
              />
            ) : (
              <p>Preparing PDF preview…</p>
            )}

            {activeCitation?.page && (
              <p className="page-jump-message">
                Jumped to page <strong>{activeCitation.page}</strong>. Page-level highlight enabled.
              </p>
            )}
            {hasRegionHighlight && (
              <p className="page-jump-message">
                Region coordinates detected; render region overlays when backend bounding boxes are wired.
              </p>
            )}
          </>
        ) : (
          <p>Add files or arXiv IDs to start browsing papers.</p>
        )}
      </div>

      <div className="page-nav">
        <button
          type="button"
          onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
          disabled={!selectedPaper || currentPage === 1}
        >
          Previous
        </button>
        <span>Page {currentPage}</span>
        <button
          type="button"
          onClick={() => setCurrentPage((prev) => prev + 1)}
          disabled={!selectedPaper}
        >
          Next
        </button>
      </div>

      <UploadControls
        urlInput={urlInput}
        setUrlInput={setUrlInput}
        handleFileUpload={handleFileUpload}
        handleUrlAdd={handleUrlAdd}
        uploads={uploads}
        handleProcessData={handleProcessData}
        processing={processing}
        processMessage={processMessage}
        uploadFeedback={uploadFeedback}
      />
    </section>
  );
}

export default PaperPane;

import React from "react";
import UploadControls from "./UploadControls";

function PaperPane({
  uploads,
  selectedPaperIndex,
  setSelectedPaperIndex,
  currentPage,
  setCurrentPage,
  urlInput,
  setUrlInput,
  handleFileUpload,
  handleUrlAdd,
  handleProcessData,
  processing,
  processMessage,
  uploadFeedback,
}) {
  const selectedPaper = uploads[selectedPaperIndex];

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
            <p>Viewer placeholder: connect your backend PDF page rendering here.</p>
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

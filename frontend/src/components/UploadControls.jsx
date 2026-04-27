import React from "react";

function UploadControls({
  urlInput,
  setUrlInput,
  handleFileUpload,
  handleUrlAdd,
  uploads,
  handleProcessData,
  processing,
  processMessage,
  uploadFeedback,
}) {
  return (
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

      {uploads.length > 0 && (
        <>
          <section className="uploads-list">
            <h3>Uploaded Items</h3>
            <ul>
              {uploads.map((item, index) => (
                <li key={`${item.name}-${index}`}>
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
    </section>
  );
}

export default UploadControls;

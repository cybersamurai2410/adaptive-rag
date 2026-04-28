import React from "react";

function ChatPane({
  question,
  setQuestion,
  loading,
  answer,
  citations,
  activeCitation,
  onCitationClick,
  onSubmit,
}) {
  return (
    <section className="pane chat-pane">
      <h2>Chat</h2>

      {loading && <p className="loading">Loading...</p>}

      {!loading && answer && (
        <div className="answer-section">
          <h3>Answer</h3>
          <p>{answer}</p>

          {citations.length > 0 && (
            <div className="citations-section">
              <h4>Sources</h4>
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
      )}

      <form onSubmit={onSubmit} className="question-form">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Enter your question here..."
          rows="4"
          className="question-input"
        />
        <button type="submit" className="submit-button" disabled={loading}>
          {loading ? "Loading..." : "Submit"}
        </button>
      </form>
    </section>
  );
}

export default ChatPane;

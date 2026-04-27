import React from "react";

function ChatPane({
  question,
  setQuestion,
  loading,
  chatHistory,
  onSubmit,
}) {
  return (
    <section className="pane chat-pane">
      <h2>Chat</h2>

      <div className="chat-history">
        {chatHistory.length === 0 ? (
          <p className="placeholder">Ask a question to start the conversation.</p>
        ) : (
          chatHistory.map((entry, index) => (
            <article key={index} className="answer-card">
              <h3>Q: {entry.question}</h3>
              <p>{entry.answer}</p>
              {entry.sources?.length > 0 && (
                <div className="source-list">
                  <h4>Sources</h4>
                  <ul>
                    {entry.sources.map((source, sourceIndex) => (
                      <li key={`${source}-${sourceIndex}`}>{source}</li>
                    ))}
                  </ul>
                </div>
              )}
            </article>
          ))
        )}
      </div>

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

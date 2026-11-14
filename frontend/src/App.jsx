import { useEffect, useRef, useState } from "react";
import { chat, predict } from "./api";
import "./App.css";

const MODALITIES = [
  {
    value: "xray",
    label: "X-Ray — pneumonia detection",
    short: "X-Ray",
    task: "Pneumonia detection",
    icon: "🫁",
    accept: ".jpg,.jpeg,.png",
    hint: "Upload a chest X-ray image (.jpg/.png).",
  },
  {
    value: "ct",
    label: "CT — lung nodule classification",
    short: "CT",
    task: "Lung nodule",
    icon: "🔬",
    accept: ".npy",
    hint: "Upload a .npy file: a 28×28×28 pre-cropped nodule patch, not a whole scan.",
  },
  {
    value: "mri",
    label: "MRI — brain tumor classification",
    short: "MRI",
    task: "Brain tumor",
    icon: "🧠",
    accept: ".jpg,.jpeg,.png",
    hint: "Upload a brain MRI image.",
  },
  {
    value: "ultrasound",
    label: "Ultrasound — breast lesion classification",
    short: "Ultrasound",
    task: "Breast lesion",
    icon: "🩺",
    accept: ".png,.jpg,.jpeg",
    hint: "Upload a breast ultrasound image.",
  },
  {
    value: "ecg",
    label: "ECG — arrhythmia classification",
    short: "ECG",
    task: "Arrhythmia",
    icon: "❤️",
    accept: ".npy",
    hint: "Upload a .npy file: a single 187-sample heartbeat waveform.",
  },
];

export default function App() {
  const [modality, setModality] = useState(MODALITIES[0].value);
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [result, setResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");

  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);

  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const messagesEndRef = useRef(null);

  const selected = MODALITIES.find((m) => m.value === modality);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, asking, result]);

  function pickFile(f) {
    if (!f) return;
    setFile(f);
    setError("");
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    pickFile(e.dataTransfer.files[0]);
  }

  async function handleAnalyze(e) {
    e.preventDefault();
    if (!file) return;
    setError("");
    setPredicting(true);
    setResult(null);
    setMessages([]);
    try {
      const data = await predict(file, modality);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setPredicting(false);
    }
  }

  async function handleAsk(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q || !result || asking) return;
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setQuestion("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setAsking(true);
    setError("");
    try {
      const data = await chat(result.session_id, q);
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setAsking(false);
    }
  }

  function handleQuestionChange(e) {
    setQuestion(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  function handleComposerKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk(e);
    }
  }

  function selectModality(value) {
    if (value === modality) return;
    setModality(value);
    setFile(null);
    setResult(null);
    setMessages([]);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function startOver() {
    setFile(null);
    setResult(null);
    setMessages([]);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            ⚕
          </span>
          <div>
            <h1>Medical Imaging AI</h1>
            <p className="brand-sub">Multi-modality diagnostic assistant</p>
          </div>
        </div>

        <p className="disclaimer">
          Research/learning project only. Not a medical device, not validated for clinical
          use, and never for real patient diagnosis or treatment decisions.
        </p>

        <nav className="modality-list">
          {MODALITIES.map((m) => (
            <button
              key={m.value}
              type="button"
              className={`modality-item ${m.value === modality ? "active" : ""}`}
              onClick={() => selectModality(m.value)}
            >
              <span className="modality-icon" aria-hidden="true">
                {m.icon}
              </span>
              <span>
                <span className="modality-name">{m.short}</span>
                <span className="modality-task">{m.task}</span>
              </span>
            </button>
          ))}
        </nav>

        <form onSubmit={handleAnalyze} className="upload-form">
          <label
            className={`dropzone ${dragOver ? "drag" : ""} ${file ? "has-file" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept={selected.accept}
              onChange={(e) => pickFile(e.target.files[0] || null)}
              hidden
            />
            {file ? (
              <>
                <strong>{file.name}</strong>
                <span>Click or drop to replace</span>
              </>
            ) : (
              <>
                <strong>Drop a file here</strong>
                <span>or click to browse</span>
              </>
            )}
          </label>
          <p className="hint">{selected.hint}</p>

          <button type="submit" className="primary-btn" disabled={!file || predicting}>
            {predicting ? (
              <span className="spinner-row">
                <span className="spinner" /> Analyzing
              </span>
            ) : (
              "Analyze"
            )}
          </button>

          {result && (
            <button type="button" className="ghost-btn" onClick={startOver}>
              Start over
            </button>
          )}
        </form>

        {error && <p className="error">{error}</p>}
      </aside>

      <main className="chat-panel">
        <header className="chat-header">
          <div>
            <h2>{selected.label}</h2>
            <p>{selected.hint}</p>
          </div>
          {result && (
            <div className="result-pill">
              <span className={`badge ${String(result.label).toLowerCase()}`}>{result.label}</span>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
                />
              </div>
              <span className="confidence-text">{(result.confidence * 100).toFixed(1)}%</span>
            </div>
          )}
        </header>

        <div className="messages">
          {!result && (
            <div className="empty-state">
              <span className="empty-icon" aria-hidden="true">
                {selected.icon}
              </span>
              <p>
                Upload a {selected.short.toLowerCase()} file in the sidebar and hit Analyze
                to get a prediction, then ask questions about it here.
              </p>
            </div>
          )}

          {result && messages.length === 0 && (
            <div className="assistant-row">
              <div className="avatar assistant-avatar">AI</div>
              <div className="bubble assistant-bubble">
                Prediction is in: <strong>{result.label}</strong> (
                {(result.confidence * 100).toFixed(1)}% confidence). Ask me anything about
                this result, for example &ldquo;Is this serious?&rdquo; or &ldquo;How
                reliable is this model?&rdquo;
              </div>
            </div>
          )}

          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="user-row">
                <div className="bubble user-bubble">{m.content}</div>
              </div>
            ) : (
              <div key={i} className="assistant-row">
                <div className="avatar assistant-avatar">AI</div>
                <div className="bubble assistant-bubble">{m.content}</div>
              </div>
            )
          )}

          {asking && (
            <div className="assistant-row">
              <div className="avatar assistant-avatar">AI</div>
              <div className="bubble assistant-bubble typing">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleAsk} className="composer">
          <textarea
            ref={textareaRef}
            value={question}
            onChange={handleQuestionChange}
            onKeyDown={handleComposerKeyDown}
            placeholder={result ? "Ask about this result..." : "Analyze a file first to start chatting"}
            disabled={!result || asking}
            rows={1}
          />
          <button
            type="submit"
            className="send-btn"
            disabled={!question.trim() || !result || asking}
            aria-label="Send"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 2 11 13" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M22 2 15 22l-4-9-9-4 20-7Z" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </form>
      </main>
    </div>
  );
}

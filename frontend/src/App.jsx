import { useRef, useState } from "react";
import { chat, predict } from "./api";
import "./App.css";

const MODALITIES = [
  { value: "xray", label: "X-Ray (pneumonia)", accept: ".jpg,.jpeg,.png", hint: "Upload a chest X-ray image." },
  { value: "ct", label: "CT (lung nodule)", accept: ".npy", hint: "Upload a .npy file: a 28x28x28 pre-cropped nodule patch, not a whole scan." },
  { value: "mri", label: "MRI (brain tumor)", accept: ".jpg,.jpeg,.png", hint: "Upload a brain MRI image." },
  { value: "ultrasound", label: "Ultrasound (breast lesion)", accept: ".png,.jpg,.jpeg", hint: "Upload a breast ultrasound image." },
  { value: "ecg", label: "ECG (arrhythmia)", accept: ".npy", hint: "Upload a .npy file: a single 187-sample heartbeat waveform." },
];

export default function App() {
  const [modality, setModality] = useState(MODALITIES[0].value);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");

  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const fileInputRef = useRef(null);

  const selected = MODALITIES.find((m) => m.value === modality);

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
    if (!q || !result) return;
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setQuestion("");
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

  function resetFile(newModality) {
    setModality(newModality);
    setFile(null);
    setResult(null);
    setMessages([]);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  return (
    <div className="page">
      <header>
        <h1>Medical Imaging AI Suite</h1>
        <p className="disclaimer">
          Learning/portfolio project. Nothing here is validated for clinical use, is a
          medical device, or should be used for real patient diagnosis or treatment decisions.
        </p>
      </header>

      <section className="card">
        <h2>1. Analyze</h2>
        <form onSubmit={handleAnalyze}>
          <label className="field">
            <span>Modality</span>
            <select value={modality} onChange={(e) => resetFile(e.target.value)}>
              {MODALITIES.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <p className="hint">{selected.hint}</p>

          <label className="field">
            <span>File</span>
            <input
              ref={fileInputRef}
              type="file"
              accept={selected.accept}
              onChange={(e) => setFile(e.target.files[0] || null)}
            />
          </label>

          <button type="submit" disabled={!file || predicting}>
            {predicting ? "Analyzing..." : "Analyze"}
          </button>
        </form>

        {result && (
          <div className="result">
            <span className={`badge ${result.label.toLowerCase()}`}>{result.label}</span>
            <span className="confidence">confidence {(result.confidence * 100).toFixed(1)}%</span>
          </div>
        )}
      </section>

      {result && (
        <section className="card">
          <h2>2. Ask about this result</h2>
          <div className="messages">
            {messages.length === 0 && <p className="hint">Ask anything about this prediction, in any language.</p>}
            {messages.map((m, i) => (
              <div key={i} className={`message ${m.role}`}>
                <strong>{m.role === "user" ? "you" : "assistant"}:</strong> {m.content}
              </div>
            ))}
            {asking && <div className="message assistant">thinking...</div>}
          </div>
          <form onSubmit={handleAsk} className="ask-row">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. Is this serious? Explain simply."
              disabled={asking}
            />
            <button type="submit" disabled={!question.trim() || asking}>
              Send
            </button>
          </form>
        </section>
      )}

      {error && <p className="error">{error}</p>}
    </div>
  );
}

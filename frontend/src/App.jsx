import { useEffect, useRef, useState } from "react";
import {
  FiActivity,
  FiChevronLeft,
  FiChevronRight,
  FiCpu,
  FiHeart,
  FiLayers,
  FiPlus,
  FiRadio,
  FiSend,
  FiX,
} from "react-icons/fi";
import { chat } from "./api";
import "./App.css";

const MODALITIES = [
  {
    value: "xray",
    label: "X-Ray — pneumonia detection",
    short: "X-Ray",
    task: "Pneumonia detection",
    icon: FiActivity,
    imageGuide: "Attach a clear chest X-ray image in JPEG, PNG, or WebP format.",
  },
  {
    value: "ct",
    label: "CT — lung nodule classification",
    short: "CT",
    task: "Lung nodule",
    icon: FiLayers,
    imageGuide: "Attach a clear CT scan slice in JPEG, PNG, or WebP format.",
  },
  {
    value: "mri",
    label: "MRI — brain tumor classification",
    short: "MRI",
    task: "Brain tumor",
    icon: FiCpu,
    imageGuide: "Attach a clear brain MRI image in JPEG, PNG, or WebP format.",
  },
  {
    value: "ultrasound",
    label: "Ultrasound — breast lesion classification",
    short: "Ultrasound",
    task: "Breast lesion",
    icon: FiRadio,
    imageGuide: "Attach a clear breast ultrasound image in JPEG, PNG, or WebP format.",
  },
  {
    value: "ecg",
    label: "ECG — arrhythmia classification",
    short: "ECG",
    task: "Arrhythmia",
    icon: FiHeart,
    imageGuide: "Attach a clear image of an ECG tracing in JPEG, PNG, or WebP format.",
  },
];

export default function App() {
  const [modality, setModality] = useState(MODALITIES[0].value);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [error, setError] = useState("");

  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [attachment, setAttachment] = useState(null);
  const [attachmentPreview, setAttachmentPreview] = useState("");
  const [chatSessionId, setChatSessionId] = useState(null);

  const imageInputRef = useRef(null);
  const textareaRef = useRef(null);
  const messagesEndRef = useRef(null);
  const attachmentPreviewUrlsRef = useRef([]);

  const selected = MODALITIES.find((m) => m.value === modality);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, asking]);

  useEffect(() => () => {
    attachmentPreviewUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  function removeAttachment() {
    if (attachmentPreview) {
      URL.revokeObjectURL(attachmentPreview);
      attachmentPreviewUrlsRef.current = attachmentPreviewUrlsRef.current.filter(
        (url) => url !== attachmentPreview,
      );
    }
    setAttachment(null);
    setAttachmentPreview("");
    if (imageInputRef.current) imageInputRef.current.value = "";
  }

  function pickAttachment(f) {
    if (!f) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(f.type)) {
      setError("Choose a JPEG, PNG, or WebP image.");
      if (imageInputRef.current) imageInputRef.current.value = "";
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError("Images must be 10 MB or smaller.");
      if (imageInputRef.current) imageInputRef.current.value = "";
      return;
    }
    removeAttachment();
    const url = URL.createObjectURL(f);
    attachmentPreviewUrlsRef.current.push(url);
    setAttachment(f);
    setAttachmentPreview(url);
    setError("");
  }

  async function askQuestion(q, image = attachment, imageUrl = attachmentPreview) {
    if ((!q && !image) || asking) return;
    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        type: "text",
        content: q,
        image: imageUrl || "",
        fileName: image?.name || "",
      },
    ]);
    setAsking(true);
    setError("");
    try {
      const data = await chat(chatSessionId, q, image);
      setChatSessionId(data.session_id);
      setMessages((prev) => [...prev, { role: "assistant", type: "text", content: data.answer }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setAsking(false);
    }
  }

  async function handleAsk(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q && !attachment) return;
    const sentAttachment = attachment;
    const sentPreview = attachmentPreview;
    setQuestion("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    removeAttachment();
    await askQuestion(q, sentAttachment, sentPreview);
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
    setMessages([]);
    setChatSessionId(null);
    setError("");
  }

  return (
    <div className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <FiActivity size={22} />
          </span>
          <div className="brand-copy">
            <h1>Medical Imaging AI</h1>
            <p className="brand-sub">Multi-modality diagnostic assistant</p>
          </div>
          <button
            type="button"
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? <FiChevronRight size={20} /> : <FiChevronLeft size={20} />}
          </button>
        </div>

        <nav className="modality-list">
          {MODALITIES.map((m) => (
            <button
              key={m.value}
              type="button"
              className={`modality-item ${m.value === modality ? "active" : ""}`}
              onClick={() => selectModality(m.value)}
              aria-label={m.label}
              title={sidebarCollapsed ? m.short : undefined}
            >
              <span className="modality-icon" aria-hidden="true">
                <m.icon size={20} />
              </span>
              <span>
                <span className="modality-name">{m.short}</span>
                <span className="modality-task">{m.task}</span>
              </span>
            </button>
          ))}
        </nav>

        <section className="modality-guide" aria-live="polite">
          <h2>{selected.short} image guide</h2>
          <p>{selected.imageGuide}</p>
        </section>

        {error && <p className="error">{error}</p>}

        <p className="sidebar-credit">
          Developed by{" "}
          <a href="https://krrkhan.com" target="_blank" rel="noreferrer">
            Khalilur Rahman Ridoy Khan
          </a>
        </p>
      </aside>

      <main className="chat-panel">
        <header className="chat-header">
          <div>
            <h2>{selected.label}</h2>
            <p>Ask a question or attach an image in the chat.</p>
          </div>
        </header>

        <div className="messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <span className="empty-icon" aria-hidden="true">
                <selected.icon size={36} />
              </span>
              <p>
                Add an image with the plus button or type a question below to start a conversation.
              </p>
            </div>
          )}

          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="user-row">
                <div className="bubble user-bubble">
                  {m.image && <img className="message-image" src={m.image} alt={m.fileName} />}
                  {m.content}
                </div>
              </div>
            ) : (
              <div key={i} className="assistant-row">
                <div className="avatar assistant-avatar" role="img" aria-label="Assistant">
                  <FiCpu size={17} aria-hidden="true" />
                </div>
                <div className="bubble assistant-bubble">{m.content}</div>
              </div>
            )
          )}

          {asking && (
            <div className="assistant-row">
              <div className="avatar assistant-avatar" role="img" aria-label="Assistant">
                <FiCpu size={17} aria-hidden="true" />
              </div>
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
          {attachmentPreview && (
            <div className="attachment-preview">
              <img src={attachmentPreview} alt={`Preview of ${attachment?.name}`} />
              <span>{attachment?.name}</span>
              <button
                type="button"
                onClick={removeAttachment}
                aria-label="Remove attached image"
                disabled={asking}
              >
                <FiX size={16} aria-hidden="true" />
              </button>
            </div>
          )}
          <div className="composer-input-row">
            <input
              ref={imageInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(e) => pickAttachment(e.target.files[0] || null)}
              hidden
            />
            <button
              type="button"
              className="attach-btn"
              onClick={() => imageInputRef.current?.click()}
              disabled={asking}
              aria-label="Add image"
              title="Add image"
            >
              <FiPlus size={20} aria-hidden="true" />
            </button>
            <textarea
              ref={textareaRef}
              value={question}
              onChange={handleQuestionChange}
              onKeyDown={handleComposerKeyDown}
              placeholder="Message Medical Imaging AI..."
              disabled={asking}
              rows={1}
            />
            <button
              type="submit"
              className="send-btn"
              disabled={(!question.trim() && !attachment) || asking}
              aria-label="Send"
            >
              <FiSend size={18} aria-hidden="true" />
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}

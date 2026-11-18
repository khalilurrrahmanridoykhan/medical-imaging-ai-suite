const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function predict(file, modality) {
  const form = new FormData();
  form.append("file", file);
  form.append("modality", modality);

  const res = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Prediction failed (${res.status})`);
  }
  return res.json();
}

export async function chat(sessionId, question, image) {
  const form = new FormData();
  if (sessionId) form.append("session_id", sessionId);
  form.append("question", question);
  if (image) form.append("image", image);

  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Chat failed (${res.status})`);
  }
  return res.json();
}

"""FastAPI backend for the React frontend: predict + chat over the 5
trained modality models.

Two endpoints:
  POST /api/predict  multipart file + modality -> {session_id, label, confidence}
  POST /api/chat      multipart {session_id?, question, image?} -> {session_id, answer}

Sessions are an in-memory dict (fine for a single-process demo; a
restart drops them). The LLM only ever sees the structured prediction,
never the raw file, same contract as chat.assistant.ChatSession.
"""

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from chat.assistant import ChatSession

app = FastAPI(title="Medical Imaging AI Suite API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict[str, ChatSession] = {}
MODALITIES = ["xray", "ct", "mri", "ultrasound", "ecg"]


@app.get("/api/health")
def health():
    return {"status": "ok", "modalities": MODALITIES}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), modality: str = Form(...)):
    if modality not in MODALITIES:
        raise HTTPException(400, f"Unknown modality '{modality}', expected one of {MODALITIES}")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        session = ChatSession(modality, tmp_path)
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}") from exc

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = session
    return {
        "session_id": session_id,
        "modality": modality,
        "label": session.prediction["label"],
        "confidence": session.prediction["confidence"],
    }


@app.post("/api/chat")
async def chat(
    question: str = Form(...),
    session_id: str | None = Form(None),
    image: UploadFile | None = File(None),
):
    session = SESSIONS.get(session_id) if session_id else None
    if session_id and session is None:
        raise HTTPException(404, "Unknown session_id, run /api/predict first")
    if session is None:
        session = ChatSession()
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = session

    image_data = None
    if image is not None:
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(415, "Choose a JPEG, PNG, or WebP image")
        image_data = await image.read(10 * 1024 * 1024 + 1)
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(413, "Images must be 10 MB or smaller")

    answer = session.ask(question, image_data)
    return {"session_id": session_id, "answer": answer}

"""FastAPI backend for the React frontend: predict + chat over the 5
trained modality models.

Two endpoints:
  POST /api/predict  multipart file + modality -> {session_id, label, confidence}
  POST /api/chat      {session_id, question} -> {answer}

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
from pydantic import BaseModel

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


class ChatRequest(BaseModel):
    session_id: str
    question: str


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
def chat(req: ChatRequest):
    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(404, "Unknown session_id, run /api/predict first")
    answer = session.ask(req.question)
    return {"answer": answer}

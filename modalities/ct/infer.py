"""Inference for the CT lung-nodule malignancy classifier.

Input contract: a .npy file containing a single pre-cropped 28x28x28
nodule-candidate voxel patch (the same format NoduleMNIST3D uses), not an
arbitrary whole CT scan or slice. Downloads trained weights from Hugging
Face Hub on first use and exposes predict(path) -> {"label", "confidence"}.
"""

import json
from pathlib import Path

import numpy as np
import torch

from modalities.ct.model import build_model

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
MODEL_PATH = WEIGHTS_DIR / "ct_model.pt"
METADATA_PATH = WEIGHTS_DIR / "ct_metadata.json"

HF_REPO = "khalilurrahmanridoykhan/medical-imaging-ai-suite-weights"

_model = None
_metadata = None


def _ensure_weights() -> None:
    if MODEL_PATH.exists() and METADATA_PATH.exists():
        return
    from huggingface_hub import hf_hub_download

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_model = hf_hub_download(repo_id=HF_REPO, filename="ct/ct_model.pt")
    downloaded_meta = hf_hub_download(repo_id=HF_REPO, filename="ct/metadata.json")
    Path(downloaded_model).replace(MODEL_PATH)
    Path(downloaded_meta).replace(METADATA_PATH)


def _load() -> None:
    global _model, _metadata
    if _model is not None:
        return
    _ensure_weights()
    _metadata = json.loads(METADATA_PATH.read_text())
    _model = build_model(num_classes=len(_metadata["classes"]))
    _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    _model.eval()


def predict(input_path: str) -> dict:
    _load()
    volume = np.load(input_path).astype("float32") / 255.0
    tensor = torch.from_numpy(volume)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)  # batch dim
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1)[0]
    idx = int(probs.argmax())
    return {"label": _metadata["classes"][idx], "confidence": float(probs[idx])}

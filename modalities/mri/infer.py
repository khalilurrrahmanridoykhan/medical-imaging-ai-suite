"""Inference for the MRI brain tumor classifier.

Downloads trained weights from Hugging Face Hub on first use, unless a
copy already exists locally in weights/. Exposes
predict(path) -> {"label": str, "confidence": float}.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from modalities.mri.model import build_model

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
MODEL_PATH = WEIGHTS_DIR / "mri_model.pt"
METADATA_PATH = WEIGHTS_DIR / "mri_metadata.json"

HF_REPO = "khalilurrahmanridoykhan/medical-imaging-ai-suite-weights"

_model = None
_metadata = None


def _ensure_weights() -> None:
    if MODEL_PATH.exists() and METADATA_PATH.exists():
        return
    from huggingface_hub import hf_hub_download

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_model = hf_hub_download(repo_id=HF_REPO, filename="mri/mri_model.pt")
    downloaded_meta = hf_hub_download(repo_id=HF_REPO, filename="mri/metadata.json")
    Path(downloaded_model).replace(MODEL_PATH)
    Path(downloaded_meta).replace(METADATA_PATH)


def _load() -> None:
    global _model, _metadata
    if _model is not None:
        return
    _ensure_weights()
    _metadata = json.loads(METADATA_PATH.read_text())
    _model = build_model(num_classes=len(_metadata["classes"]), pretrained=False)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    _model.eval()


def predict(input_path: str) -> dict:
    _load()
    size = _metadata["image_size"]
    tfm = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(_metadata["mean"], _metadata["std"]),
        ]
    )
    image = Image.open(input_path).convert("RGB")
    tensor = tfm(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1)[0]
    idx = int(probs.argmax())
    return {"label": _metadata["classes"][idx], "confidence": float(probs[idx])}

"""Registry mapping modality names to their predict() functions.

Each modality module exposes predict(input_path: str) -> dict with the
contract {"label": str, "confidence": float}. Entries are added here as
each modality's model lands (see README roadmap, phases M1-M5).
"""

from typing import Callable, Dict, List

MODALITY_REGISTRY: Dict[str, Callable[[str], dict]] = {}


def register(name: str, predict_fn: Callable[[str], dict]) -> None:
    MODALITY_REGISTRY[name] = predict_fn


def available_modalities() -> List[str]:
    return sorted(MODALITY_REGISTRY)

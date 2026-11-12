import numpy as np

from modalities.ct import infer
from modalities.ct.model import build_model


def test_predict_contract(tmp_path, monkeypatch):
    # Inject a random-weight model and fake metadata instead of downloading
    # real trained weights from Hugging Face Hub.
    monkeypatch.setattr(infer, "_load", lambda: None)
    infer._model = build_model(num_classes=2)
    infer._model.eval()
    infer._metadata = {"classes": ["benign", "malignant"], "input_shape": [1, 28, 28, 28]}

    volume_path = tmp_path / "sample.npy"
    np.save(volume_path, np.random.randint(0, 255, size=(1, 28, 28, 28), dtype=np.uint8))

    result = infer.predict(str(volume_path))

    assert set(result) == {"label", "confidence"}
    assert result["label"] in infer._metadata["classes"]
    assert 0.0 <= result["confidence"] <= 1.0

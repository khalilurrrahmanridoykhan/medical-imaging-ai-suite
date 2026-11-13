import numpy as np

from modalities.ecg import infer
from modalities.ecg.model import build_model


def test_predict_contract(tmp_path, monkeypatch):
    # Inject a random-weight model and fake metadata instead of downloading
    # real trained weights from Hugging Face Hub.
    monkeypatch.setattr(infer, "_load", lambda: None)
    infer._model = build_model(num_classes=5)
    infer._model.eval()
    infer._metadata = {"classes": ["N", "S", "V", "F", "Q"], "input_length": 187}

    signal_path = tmp_path / "sample.npy"
    np.save(signal_path, np.random.rand(187).astype("float32"))

    result = infer.predict(str(signal_path))

    assert set(result) == {"label", "confidence"}
    assert result["label"] in infer._metadata["classes"]
    assert 0.0 <= result["confidence"] <= 1.0

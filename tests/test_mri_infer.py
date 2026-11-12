from PIL import Image

from modalities.mri import infer
from modalities.mri.model import build_model


def test_predict_contract(tmp_path, monkeypatch):
    # Inject a random-weight model and fake metadata instead of downloading
    # real trained weights from Hugging Face Hub.
    monkeypatch.setattr(infer, "_load", lambda: None)
    infer._model = build_model(num_classes=4, pretrained=False)
    infer._model.eval()
    infer._metadata = {
        "classes": ["glioma", "meningioma", "notumor", "pituitary"],
        "image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(image_path)

    result = infer.predict(str(image_path))

    assert set(result) == {"label", "confidence"}
    assert result["label"] in infer._metadata["classes"]
    assert 0.0 <= result["confidence"] <= 1.0

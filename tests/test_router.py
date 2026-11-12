from router.registry import MODALITY_REGISTRY, available_modalities, register


def test_register_adds_modality():
    MODALITY_REGISTRY.clear()
    register("dummy", lambda path: {"label": "ok", "confidence": 1.0})

    assert available_modalities() == ["dummy"]
    assert MODALITY_REGISTRY["dummy"]("x")["label"] == "ok"


def test_unregistered_modality_is_absent():
    MODALITY_REGISTRY.clear()

    assert "xray" not in MODALITY_REGISTRY

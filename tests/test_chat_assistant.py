from chat.assistant import ChatSession
from router.registry import MODALITY_REGISTRY


def test_chat_session_asks_and_tracks_history(monkeypatch, tmp_path):
    MODALITY_REGISTRY.clear()
    MODALITY_REGISTRY["xray"] = lambda path: {"label": "PNEUMONIA", "confidence": 0.93}
    # Don't let run_prediction's register_all() overwrite our stub above.
    monkeypatch.setattr("chat.assistant.register_all", lambda: None)
    monkeypatch.setattr("chat.llm.chat", lambda messages: "This suggests pneumonia.")

    dummy_input = tmp_path / "sample.png"
    dummy_input.write_text("fake")

    session = ChatSession("xray", str(dummy_input))
    assert session.prediction == {"label": "PNEUMONIA", "confidence": 0.93}

    answer = session.ask("What does this mean?")

    assert answer == "This suggests pneumonia."
    assert session.history[-2] == {"role": "user", "content": "What does this mean?"}
    assert session.history[-1] == {"role": "assistant", "content": answer}


def test_unknown_modality_raises(monkeypatch, tmp_path):
    MODALITY_REGISTRY.clear()
    monkeypatch.setattr("chat.assistant.register_all", lambda: None)

    dummy_input = tmp_path / "sample.png"
    dummy_input.write_text("fake")

    try:
        ChatSession("not-a-modality", str(dummy_input))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_standalone_chat_forwards_image_to_llm(monkeypatch):
    seen = {}

    def fake_chat(messages):
        seen["messages"] = messages
        return "The image shows a sample."

    monkeypatch.setattr("chat.llm.chat", fake_chat)

    session = ChatSession()
    answer = session.ask("What is in this image?", b"image bytes")

    assert answer == "The image shows a sample."
    assert seen["messages"][-1] == {
        "role": "user",
        "content": "What is in this image?",
        "images": [b"image bytes"],
    }


def test_chat_requires_both_prediction_arguments():
    try:
        ChatSession("xray")
        assert False, "expected ValueError"
    except ValueError:
        pass

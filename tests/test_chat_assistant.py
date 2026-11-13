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

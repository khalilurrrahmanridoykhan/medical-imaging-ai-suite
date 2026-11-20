# Demo app

Phase M7: `app.py` is a Gradio app that uploads a file, picks a modality, and shows the
prediction (label + confidence), via the same `router.predict` used by the CLI and the
FastAPI backend. No chat here, only the prediction step, since the chat layer needs a
locally-running LLM (Ollama) that a free Hugging Face Space cannot provide.

Run locally: `python app/app.py` (after `pip install -r requirements.txt`).

Deploy: push this file plus `router/`, `modalities/`, and `requirements.txt` to a Hugging
Face Space (SDK: Gradio). Weights are pulled from the public Hugging Face Hub model repo on
first prediction, same as every other entry point, so nothing large needs to be pushed.

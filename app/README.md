# Demo app

Phase M7: a Gradio app that uploads a file, picks a modality, and shows the prediction
(label + confidence), via the same `router.predict` used by the CLI and the FastAPI backend.
No chat here, only the prediction step, since the chat layer needs a locally-running LLM
(Ollama) that a hosted demo cannot provide.

Two versions, same UI and prediction logic, different hosting target:

- **`app.py`** — Hugging Face Spaces target. Run locally: `python app/app.py` (after
  `pip install -r requirements.txt`). Hugging Face now requires a paid plan to host a
  Gradio/Docker Space on compute (only static Spaces stay free), so this is not currently
  deployed publicly; see the main README's "Hosting notes" for the live alternative.
- **`modal_app.py`** — the one actually deployed:
  https://khalilurrahmanridoykhan--medical-imaging-ai-suite-ui.us-east.modal.run, on
  [Modal](https://modal.com)'s free Starter plan. Deploy with `modal deploy app/modal_app.py`.
  Runs Gradio as a sticky-session Server (`@modal.sessioned()`), not a plain
  `@modal.asgi_app()` function, because Gradio's queue needs one continuously-running
  process per client across its two separate HTTP calls; a plain ASGI function can recycle
  its background queue task onto a different container between them. Model weights are
  baked into the image from the local `weights/` directory rather than pulled from Hugging
  Face Hub, since that fallback repo is not actually public under this account.

"""Deploys the Medical Imaging AI Suite's Gradio demo on Modal (modal.com).

Alternative to app.py's Hugging Face Spaces target: Hugging Face now requires
a paid plan to host a Gradio Space on compute, so this targets Modal's free
Starter plan ($30/month compute credit, scale-to-zero) instead. Same
predict-only scope as app.py: router.predict, no chat (that layer needs a
locally-running LLM this hosted demo does not have).

Trained weights (~128MB total) are baked into the image from the local
weights/ directory rather than pulled from Hugging Face Hub at runtime:
the HF_REPO path each infer.py falls back to is not actually a public
repo under this account, so a fresh container cannot download from it.

Runs Gradio as a real sticky-session Server rather than a plain ASGI
function: Gradio's queue (/gradio_api/queue/join + /queue/data) needs one
continuously-running process per client, and a plain @modal.asgi_app()
function can recycle its background queue tasks between the two separate
HTTP requests that one Gradio prediction makes, producing a 404 on the
second request. @modal.sessioned() pins a client's requests to the same
container for the session's lifetime, which keeps the queue alive.

Deploy: modal deploy app/modal_app.py
"""

import sys

import modal

app = modal.App("medical-imaging-ai-suite")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "fastapi[standard]",
        "gradio",
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "huggingface_hub",
    )
    .add_local_dir("router", remote_path="/root/router")
    .add_local_dir("modalities", remote_path="/root/modalities")
    .add_local_dir("weights", remote_path="/root/weights")
)

with image.imports():
    import gradio as gr

DISCLAIMER = (
    "Educational/portfolio project. Nothing here is validated for clinical "
    "use, is a medical device, or should ever be used for real patient "
    "diagnosis or treatment decisions."
)

MODALITY_CHOICES = {
    "X-Ray — image, Normal / Pneumonia": "xray",
    "MRI — image, Glioma / Meningioma / Pituitary / No tumor": "mri",
    "Ultrasound — image, Normal / Benign / Malignant": "ultrasound",
    "CT — .npy 28x28x28 nodule patch, Benign / Malignant": "ct",
    "ECG — .npy 187-sample heartbeat, N / S / V / F / Q": "ecg",
}


def build_demo() -> "gr.Blocks":
    sys.path.insert(0, "/root")
    from router.bootstrap import register_all
    from router.registry import MODALITY_REGISTRY

    register_all()

    def predict(modality_label: str, file) -> str:
        if file is None:
            return "Upload a file first."
        modality = MODALITY_CHOICES[modality_label]
        if modality not in MODALITY_REGISTRY:
            return f"Modality '{modality}' is not registered."
        try:
            result = MODALITY_REGISTRY[modality](file.name)
        except Exception as exc:
            return f"Could not run this file through the {modality} model: {exc}"
        return f"**{result['label']}**  \nConfidence: {result['confidence']:.1%}"

    with gr.Blocks(title="Medical Imaging AI Suite") as demo:
        gr.Markdown("# Medical Imaging AI Suite")
        gr.Markdown(DISCLAIMER)
        modality_dropdown = gr.Dropdown(
            choices=list(MODALITY_CHOICES), value=list(MODALITY_CHOICES)[0], label="Modality"
        )
        file_input = gr.File(
            label="Input file (JPG/PNG for X-Ray, MRI, Ultrasound; .npy for CT, ECG)"
        )
        run_button = gr.Button("Predict")
        output = gr.Markdown(label="Result")
        run_button.click(predict, inputs=[modality_dropdown, file_input], outputs=output)

    return demo


@app.server(image=image, port=8000, max_concurrency=1, unauthenticated=True)
@modal.sessioned()
class GradioServer:
    @modal.enter()
    def start(self):
        demo = build_demo()
        demo.launch(server_name="0.0.0.0", server_port=8000, prevent_thread_lock=True)


@app.function(image=image)
@modal.fastapi_endpoint()
def ui():
    from fastapi.responses import RedirectResponse

    server = modal.Server.from_name("medical-imaging-ai-suite", "GradioServer")
    session = server.sessions.start(idle_timeout=600)
    return RedirectResponse(url=f"{server.get_url()}/?modal_session_token={session.token}")

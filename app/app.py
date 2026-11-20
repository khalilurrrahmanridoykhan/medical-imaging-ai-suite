"""Gradio demo for the Medical Imaging AI Suite (Phase M7).

Uploads a file, picks a modality, and runs the same router used by the CLI
(router.predict) and the FastAPI backend (backend.main), so this demo and
the local web UI never drift from the same prediction logic. Designed to
be deployed on Hugging Face Spaces: model weights are pulled from the
public HF Hub repo on first use, same as every other entry point.
"""

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.bootstrap import register_all
from router.registry import MODALITY_REGISTRY

register_all()

MODALITY_CHOICES = {
    "X-Ray — image, Normal / Pneumonia": "xray",
    "MRI — image, Glioma / Meningioma / Pituitary / No tumor": "mri",
    "Ultrasound — image, Normal / Benign / Malignant": "ultrasound",
    "CT — .npy 28x28x28 nodule patch, Benign / Malignant": "ct",
    "ECG — .npy 187-sample heartbeat, N / S / V / F / Q": "ecg",
}


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
    modality_dropdown = gr.Dropdown(
        choices=list(MODALITY_CHOICES), value=list(MODALITY_CHOICES)[0], label="Modality"
    )
    file_input = gr.File(
        label="Input file (JPG/PNG for X-Ray, MRI, Ultrasound; .npy for CT, ECG)"
    )
    run_button = gr.Button("Predict")
    output = gr.Markdown(label="Result")
    run_button.click(predict, inputs=[modality_dropdown, file_input], outputs=output)

if __name__ == "__main__":
    demo.launch()

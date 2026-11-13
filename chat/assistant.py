"""Natural-language chat over a modality model's prediction.

Runs the chosen modality's classifier once on the given input, then lets
a local LLM (via Ollama) answer free-form questions about that result in
natural language. The LLM only ever sees the structured prediction
(label, confidence, this modality's known accuracy and weaknesses) and
the disclaimer, never the raw image or signal, this is a text-only LLM
explaining a result, not a vision model looking at the input itself.
"""

from router.bootstrap import register_all
from router.registry import MODALITY_REGISTRY

DISCLAIMER = (
    "This is a learning/portfolio project. Nothing here is validated for "
    "clinical use, is a medical device, or should be used for real patient "
    "diagnosis or treatment decisions."
)

MODALITY_CONTEXT = {
    "xray": (
        "Pneumonia detector (Normal vs Pneumonia). Test accuracy 84.9%, AUROC "
        "0.959. High sensitivity (99.2%) but weak specificity (61.1%): it "
        "flags over a third of healthy X-rays as pneumonia."
    ),
    "ct": (
        "Lung nodule malignancy classifier (benign vs malignant) on a small "
        "28x28x28 pre-cropped nodule patch, not a whole scan. Test accuracy "
        "81.0%, AUROC 0.854. Only 67% sensitivity for malignant nodules, "
        "misses about a third of them."
    ),
    "mri": (
        "Brain tumor classifier (glioma, meningioma, notumor, pituitary). "
        "Test accuracy 95.1%, AUROC 0.990. notumor and pituitary are 99%+ "
        "accurate, meningioma 98%, but glioma is only 82.5%, often confused "
        "with meningioma or notumor."
    ),
    "ultrasound": (
        "Breast lesion classifier (normal, benign, malignant). Test accuracy "
        "91.0%, AUROC 0.987. Malignant is caught 100% of the time in testing, "
        "but 3 of 14 normal cases were flagged as malignant (false alarms)."
    ),
    "ecg": (
        "Heartbeat arrhythmia classifier (N normal, S supraventricular, V "
        "ventricular, F fusion, Q unknown/paced). Test accuracy 86.9%, AUROC "
        "0.977. Per-class recall: N 85.9%, S 78.2%, V 87.8%, F 92.0%, Q 98.3%."
    ),
}


def run_prediction(modality: str, input_path: str) -> dict:
    register_all()
    if modality not in MODALITY_REGISTRY:
        raise ValueError(f"Unknown or untrained modality '{modality}'")
    return MODALITY_REGISTRY[modality](input_path)


def build_system_prompt(modality: str, prediction: dict) -> str:
    context = MODALITY_CONTEXT.get(modality, "No performance info available.")
    return (
        f"You are a plain-language explainer for a {modality} AI model's result.\n"
        f"Model info: {context}\n"
        f"This model's prediction on the given input: label={prediction['label']}, "
        f"confidence={prediction['confidence']:.3f}.\n"
        f"Important: {DISCLAIMER}\n"
        "Answer the user's questions about this specific result in clear, "
        "simple language, in whatever language they ask in. Be honest about "
        "the model's known accuracy and weaknesses above when relevant. You "
        "cannot see the image or signal yourself, only this prediction, so "
        "never claim to observe visual details you weren't given."
    )


class ChatSession:
    def __init__(self, modality: str, input_path: str):
        self.modality = modality
        self.prediction = run_prediction(modality, input_path)
        self.history = [
            {"role": "system", "content": build_system_prompt(modality, self.prediction)}
        ]

    def ask(self, question: str) -> str:
        from chat.llm import chat as llm_chat

        self.history.append({"role": "user", "content": question})
        answer = llm_chat(self.history)
        self.history.append({"role": "assistant", "content": answer})
        return answer

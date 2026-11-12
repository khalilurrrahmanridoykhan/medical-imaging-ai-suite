# Medical Imaging AI Suite

A multi-modality medical image AI system: one model per imaging modality (X-Ray, CT, MRI,
Ultrasound, ECG), unified behind a single router. Feed it an image and say which modality
it is, and the matching model runs and returns a result.

## Disclaimer

Every model in this repo is trained on public research datasets for learning and portfolio
purposes only. **Nothing here is validated for clinical use, is a medical device, or should
ever be used for real patient diagnosis or treatment decisions.**

## What it does

| Modality | Task | Classes |
|---|---|---|
| X-Ray | Pneumonia detection | Normal, Pneumonia |
| CT | Lung nodule classification | Benign, Malignant |
| MRI | Brain tumor classification | Glioma, Meningioma, Pituitary, No tumor |
| Ultrasound | Breast lesion classification | Normal, Benign, Malignant |
| ECG | Arrhythmia classification | 5-class heartbeat categorization |

Each modality has its own dataset, model, training script, and inference module. A
router dispatches an input file to the right model and returns a plain prediction plus
confidence score.

## Repo layout

```
modalities/
  xray/        data.py, model.py, train.py, notebook.ipynb, infer.py
  ct/          same pattern
  mri/         same pattern
  ultrasound/  same pattern
  ecg/         same pattern (input is a waveform array, not an image)
router/
  registry.py  MODALITY_REGISTRY: name -> predict(input_path) -> dict
  predict.py   CLI: python -m router.predict --input <path> --modality <name>
weights/       downloaded at runtime from Hugging Face Hub, not committed here
tests/         unit tests for each infer.py's input/output contract
docs/
  DATASETS.md  dataset sources, licenses, citations
app/           Gradio demo app (Phase M7)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Three of the five datasets are hosted on Kaggle; downloading them requires a free Kaggle
account and API token (`~/.kaggle/kaggle.json`). The CT dataset (MedMNIST `NoduleMNIST3D`)
downloads automatically with no account needed.

## Training

Each modality's model is trained in Google Colab (free GPU). `modalities/<name>/train.py`
is the single source of truth; `modalities/<name>/notebook.ipynb` is a thin wrapper that
installs dependencies and calls it. After training, weights are uploaded to a public
Hugging Face Hub model repo and pulled down at inference time, keeping large binaries out
of git.

## Usage

```bash
python -m router.predict --input sample_xray.png --modality xray
python -m router.predict --input sample_ct_slice.png --modality ct
```

Returns JSON: `{"modality": "xray", "label": "Pneumonia", "confidence": 0.93}`.

**Note:** `infer.py` downloads trained weights from Hugging Face Hub on first use. Until a
modality has actually been trained in Colab and its weights uploaded there, running its
prediction will fail with a download error, that part is a manual step outside this repo.

## Status / Roadmap

- [x] M0: Repo scaffold, LICENSE, README, GitHub setup
- [x] M1: X-Ray (pneumonia detection)
- [ ] M2: CT (lung nodule classification)
- [ ] M3: MRI (brain tumor classification)
- [ ] M4: Ultrasound (breast lesion classification)
- [ ] M5: ECG (arrhythmia classification)
- [ ] M6: Unified router + CLI
- [ ] M7: Live Gradio demo

## Datasets and licenses

See [docs/DATASETS.md](docs/DATASETS.md) for the exact source, license, and citation for
each dataset used.

## License

Apache License 2.0, see [LICENSE](LICENSE). Dataset licenses are separate and documented
in [docs/DATASETS.md](docs/DATASETS.md); trained model weights inherit the license terms
of their training data where applicable.

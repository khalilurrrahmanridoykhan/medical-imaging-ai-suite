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

Four of the five datasets are hosted on Kaggle; downloading them requires a free Kaggle
account and API token (`~/.kaggle/kaggle.json`). The CT dataset (MedMNIST `NoduleMNIST3D`)
downloads automatically with no account needed.

## Training

`modalities/<name>/train.py` is the single source of truth for training. It picks the best
available device itself (CUDA, then Apple Silicon MPS, then CPU, see `common/device.py`),
so it runs either in Google Colab (free GPU) via the thin `notebook.ipynb` wrapper, or
directly as a local script, no notebook needed, on a machine with a capable GPU (an Apple
Silicon Mac included). After training, weights are uploaded to a public Hugging Face Hub
model repo and pulled down at inference time, keeping large binaries out of git.

## Model performance (so far)

| Modality | Test accuracy | Test AUROC | Notes |
|---|---|---|---|
| X-Ray (pneumonia) | 84.9% | 0.959 | Catches 387/390 pneumonia cases (99.2% sensitivity) but only 143/234 normal cases (61.1% specificity), flags over a third of healthy X-rays as pneumonia. Val accuracy during training was ~98%; this train/test gap is a known quirk of this specific Kaggle dataset (its test split comes from a different distribution than train/val), not a bug here. |
| CT (nodule) | 81.0% | 0.854 | Catches 43/64 malignant test cases (67% sensitivity), misses the rest. A small from-scratch 3D CNN on a tiny, imbalanced dataset. |
| MRI (brain tumor) | 95.1% | 0.990 | Strongest result so far. notumor and pituitary are 99%+ correct, meningioma 98%, but glioma is only 82.5% (330/400), mostly confused with meningioma or notumor. |
| Ultrasound (breast lesion) | 91.0% | 0.987 | Malignant is caught 21/21 (100%), but 3/14 normal cases are flagged as malignant (false alarms, the safer error direction, but still a real limitation on a very small 780-image dataset). |

All of these are well short of clinical-grade performance, see the disclaimer. Remaining
modalities are not trained yet (see Status / Roadmap below).

## Usage

```bash
python -m router.predict --input sample_xray.png --modality xray
python -m router.predict --input sample_nodule_patch.npy --modality ct
```

Returns JSON: `{"modality": "xray", "label": "Pneumonia", "confidence": 0.93}`.

**CT input note:** the CT model expects a pre-cropped 28x28x28 nodule-candidate patch as a
`.npy` file (the same format NoduleMNIST3D uses), not an arbitrary whole CT scan or slice.
Extracting candidate patches from a full CT volume is a separate, much larger pipeline
that's out of scope here.

**Note:** `infer.py` downloads trained weights from Hugging Face Hub on first use, unless a
copy already exists locally in `weights/` (how X-Ray and CT were tested here, trained
locally and copied straight into `weights/`). A modality with no local weights and nothing
uploaded to HF Hub yet will fail with a download error when you try to predict with it.

## Status / Roadmap

- [x] M0: Repo scaffold, LICENSE, README, GitHub setup
- [x] M1: X-Ray (pneumonia detection), trained locally on an Apple M2 (MPS)
- [x] M2: CT (lung nodule classification), trained locally on an Apple M2 (MPS)
- [x] M3: MRI (brain tumor classification), trained locally on an Apple M2 (MPS)
- [x] M4: Ultrasound (breast lesion classification), trained locally on an Apple M2 (MPS)
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

# Weights

Trained model weights are not committed to this repo. After a modality is trained in
Google Colab (see `modalities/<name>/notebook.ipynb`), the resulting weight file is
uploaded to a public Hugging Face Hub model repo, and each `modalities/<name>/infer.py`
downloads it into this folder at first use via `huggingface_hub.hf_hub_download`.

This folder's contents are gitignored.

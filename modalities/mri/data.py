"""Download and locate the Brain Tumor MRI Dataset (Kaggle).

Source: Kaggle `masoudnickparvar/brain-tumor-mri-dataset`, CC BY 4.0 (see
docs/DATASETS.md). Requires a Kaggle API token.
"""

import subprocess
import zipfile
from pathlib import Path

DATASET_SLUG = "masoudnickparvar/brain-tumor-mri-dataset"
DATA_DIR = Path(__file__).resolve().parent / "data"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def download(dest: Path = DATA_DIR) -> Path:
    """Download and unzip the dataset into `dest`. Returns the root
    directory containing Training/ and Testing/ subfolders (each with
    glioma/meningioma/notumor/pituitary class subfolders, 1400 and 400
    images per class respectively, already balanced)."""
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "Training").is_dir() and (dest / "Testing").is_dir():
        return dest

    zip_path = dest / "brain-tumor-mri-dataset.zip"
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(dest)],
        check=True,
    )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


if __name__ == "__main__":
    root = download()
    print(f"Dataset ready at {root}")

"""Download and locate the BUSI (Breast Ultrasound Images) dataset.

Source: Kaggle `aryashah2k/breast-ultrasound-images-dataset`, a mirror of
the original Al-Dhabyani et al. 2020 BUSI dataset (see docs/DATASETS.md).
Each image has a companion `_mask*.png` segmentation file alongside it in
the same class folder; this module filters those out since this is a
classification task, not segmentation.
"""

import subprocess
import zipfile
from pathlib import Path

DATASET_SLUG = "aryashah2k/breast-ultrasound-images-dataset"
DATA_DIR = Path(__file__).resolve().parent / "data"
CLASSES = ["normal", "benign", "malignant"]


def download(dest: Path = DATA_DIR) -> Path:
    """Download and unzip the dataset into `dest`. Returns the root
    directory containing normal/, benign/, and malignant/ class folders."""
    dest.mkdir(parents=True, exist_ok=True)
    existing = dest / "Dataset_BUSI_with_GT"
    if existing.is_dir():
        return existing

    zip_path = dest / "breast-ultrasound-images-dataset.zip"
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(dest)],
        check=True,
    )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return existing


def list_samples(image_root: Path) -> list:
    """Return [(path, class_index), ...] for actual images only, skipping
    the `_mask*.png` segmentation files that live alongside them."""
    samples = []
    for class_idx, class_name in enumerate(CLASSES):
        for path in sorted((image_root / class_name).glob("*.png")):
            if "_mask" in path.stem:
                continue
            samples.append((str(path), class_idx))
    return samples


if __name__ == "__main__":
    root = download()
    samples = list_samples(root)
    print(f"Dataset ready at {root}, {len(samples)} images")

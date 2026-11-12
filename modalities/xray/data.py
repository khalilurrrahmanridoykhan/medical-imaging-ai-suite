"""Download and locate the Chest X-Ray Images (Pneumonia) dataset.

Source: Kaggle `paultimothymooney/chest-xray-pneumonia` (CC BY 4.0, see
docs/DATASETS.md). Requires a Kaggle API token at ~/.kaggle/kaggle.json or
the KAGGLE_USERNAME/KAGGLE_KEY environment variables.
"""

import subprocess
import zipfile
from pathlib import Path

DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"
DATA_DIR = Path(__file__).resolve().parent / "data"
CLASSES = ["NORMAL", "PNEUMONIA"]
SPLIT_NAMES = ("train", "val", "test")


def _find_image_root(start: Path) -> Path:
    """The dataset's zip sometimes nests chest_xray/chest_xray/..., so search
    for the directory that actually holds the three split folders."""
    candidates = [start, *start.rglob("*")]
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / s).is_dir() for s in SPLIT_NAMES):
            return candidate
    raise FileNotFoundError(f"Could not locate train/val/test folders under {start}")


def download(dest: Path = DATA_DIR) -> Path:
    """Download and unzip the dataset into `dest`. Returns the image root
    directory containing train/, val/, and test/ subfolders."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        return _find_image_root(dest)
    except FileNotFoundError:
        pass

    zip_path = dest / "chest-xray-pneumonia.zip"
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(dest)],
        check=True,
    )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return _find_image_root(dest)


def splits(image_root: Path) -> dict:
    """Return {"train": path, "val": path, "test": path}."""
    return {name: image_root / name for name in SPLIT_NAMES}


if __name__ == "__main__":
    root = download()
    print(f"Dataset ready at {root}")

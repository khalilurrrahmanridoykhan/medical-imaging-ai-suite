"""Download and locate the MIT-BIH portion of the ECG Heartbeat
Categorization Dataset (Kaggle).

Source: Kaggle `shayanfazeli/heartbeat` (Kachuee et al. 2018), derived
from the PhysioNet MIT-BIH Arrhythmia Database (see docs/DATASETS.md).
Each row is a pre-segmented single heartbeat, 187 waveform samples plus a
label, not a raw multi-beat ECG strip. The same download also contains
PTB diagnostic ECG files (ptbdb_*.csv) for a separate, unrelated binary
task; those are not used here.
"""

import subprocess
import zipfile
from pathlib import Path

import pandas as pd

DATASET_SLUG = "shayanfazeli/heartbeat"
DATA_DIR = Path(__file__).resolve().parent / "data"
CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_NAMES = {
    "N": "Normal",
    "S": "Supraventricular premature beat",
    "V": "Premature ventricular contraction",
    "F": "Fusion of ventricular and normal beat",
    "Q": "Unknown / paced beat",
}


def download(dest: Path = DATA_DIR) -> Path:
    """Download and unzip the dataset into `dest`. Returns the directory
    containing mitbih_train.csv and mitbih_test.csv."""
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "mitbih_train.csv").exists() and (dest / "mitbih_test.csv").exists():
        return dest

    zip_path = dest / "heartbeat.zip"
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(dest)],
        check=True,
    )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


def load_split(data_dir: Path, name: str):
    """name is 'train' or 'test'. Returns (signals, labels) as numpy
    arrays: signals shape (n, 187) float32, labels shape (n,) int64."""
    df = pd.read_csv(data_dir / f"mitbih_{name}.csv", header=None)
    signals = df.iloc[:, :-1].to_numpy(dtype="float32")
    labels = df.iloc[:, -1].to_numpy(dtype="int64")
    return signals, labels


if __name__ == "__main__":
    root = download()
    signals, labels = load_split(root, "train")
    print(f"Dataset ready at {root}, train shape {signals.shape}")

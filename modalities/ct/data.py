"""Load the NoduleMNIST3D dataset (MedMNIST v2).

Source: MedMNIST v2, CC BY 4.0 (see docs/DATASETS.md). Derived from
LIDC-IDRI via the LUNA16 nodule-candidate preprocessing pipeline. Each
sample is a pre-extracted 28x28x28 voxel patch centered on a nodule
candidate, not a whole CT scan or slice, so this model expects that same
kind of pre-cropped nodule patch as input, not an arbitrary CT image file.

Auto-downloads on first use via `pip install medmnist`, no account needed.
"""

from medmnist import NoduleMNIST3D

CLASSES = ["benign", "malignant"]


def load_splits(download: bool = True):
    train = NoduleMNIST3D(split="train", download=download)
    val = NoduleMNIST3D(split="val", download=download)
    test = NoduleMNIST3D(split="test", download=download)
    return train, val, test

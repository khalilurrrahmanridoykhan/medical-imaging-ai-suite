"""Shared training-device selection: CUDA > Apple Silicon MPS > CPU.

Lets each modality's train.py run on whatever accelerator is actually
available, Colab's CUDA GPU or a local Apple Silicon Mac's MPS backend,
without duplicating the same if/else in every script.
"""

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

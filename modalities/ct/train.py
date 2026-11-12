"""Train the CT lung-nodule malignancy classifier.

Single source of truth for training; modalities/ct/notebook.ipynb is a
thin Colab wrapper that installs dependencies and calls this script. Also
runs fine as a plain local script (e.g. on an Apple Silicon Mac via MPS),
since NoduleMNIST3D and the model are both lightweight.

Usage:
    python -m modalities.ct.train --epochs 15 --out modalities/ct/out
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from common.device import get_device
from modalities.ct.data import CLASSES, load_splits
from modalities.ct.model import build_model


class NormalizedVolumes(Dataset):
    """Wraps a medmnist dataset: float tensor in [0, 1], shape (1, 28, 28, 28)."""

    def __init__(self, base):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        tensor = torch.from_numpy(img).float() / 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor, int(label.reshape(-1)[0])


def class_weights(base, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, since NoduleMNIST3D is imbalanced
    (~75% benign / 25% malignant) and an unweighted loss collapses to
    always predicting the majority class."""
    counts = np.zeros(num_classes)
    for i in range(len(base)):
        counts[int(base[i][1].reshape(-1)[0])] += 1
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, device, criterion, optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, n = 0.0, 0, 0
    all_labels, all_probs = [], []
    for volumes, labels in loader:
        volumes, labels = volumes.to(device), labels.to(device)
        with torch.set_grad_enabled(is_train):
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        probs = torch.softmax(outputs, dim=1)[:, 1]
        total_loss += loss.item() * volumes.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += volumes.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return {"loss": total_loss / n, "accuracy": correct / n, "auroc": auroc}


def evaluate_confusion(model, loader, device) -> list:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for volumes, lbls in loader:
            outputs = model(volumes.to(device))
            preds.extend(outputs.argmax(1).cpu().tolist())
            labels.extend(lbls.tolist())
    return confusion_matrix(labels, preds).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="modalities/ct/out")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_base, val_base, test_base = load_splits()
    train_loader = DataLoader(NormalizedVolumes(train_base), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(NormalizedVolumes(val_base), batch_size=args.batch_size)
    test_loader = DataLoader(NormalizedVolumes(test_base), batch_size=args.batch_size)

    model = build_model(num_classes=len(CLASSES)).to(device)
    weights = class_weights(train_base, len(CLASSES)).to(device)
    print(f"Class weights (inverse frequency): {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "ct_model.pt"
    best_auroc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, device, criterion)
        print(f"epoch {epoch}: train={train_metrics} val={val_metrics}")
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), weights_path)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    test_metrics = run_epoch(model, test_loader, device, criterion)
    cm = evaluate_confusion(model, test_loader, device)
    print(f"test metrics: {test_metrics}")
    print(f"confusion matrix (rows=true, cols=pred), classes={CLASSES}: {cm}")

    metadata = {
        "classes": CLASSES,
        "input_shape": [1, 28, 28, 28],
        "test_accuracy": test_metrics["accuracy"],
        "test_auroc": test_metrics["auroc"],
        "confusion_matrix": cm,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved weights and metadata to {out_dir}")


if __name__ == "__main__":
    main()

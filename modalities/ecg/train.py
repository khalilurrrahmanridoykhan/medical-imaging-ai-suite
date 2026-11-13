"""Train the ECG heartbeat arrhythmia classifier (5-class, MIT-BIH).

Single source of truth for training; modalities/ecg/notebook.ipynb is a
thin Colab wrapper that installs dependencies and calls this script. Also
runs fine as a plain local script, e.g. on an Apple Silicon Mac via MPS.

Usage:
    python -m modalities.ecg.train --epochs 10 --out modalities/ecg/out
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from common.device import get_device
from modalities.ecg.data import CLASSES, download, load_split
from modalities.ecg.model import build_model


class SignalDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx]).unsqueeze(0)  # (1, 187)
        return signal, int(self.labels[idx])


def class_weights(labels, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights. MIT-BIH is heavily imbalanced
    (class N is ~83%, class F is under 1%); an unweighted loss collapses
    toward always predicting N."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, device, criterion, num_classes, optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, n = 0.0, 0, 0
    all_labels, all_probs = [], []
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        with torch.set_grad_enabled(is_train):
            outputs = model(signals)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        probs = torch.softmax(outputs, dim=1)
        total_loss += loss.item() * signals.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += signals.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())
    auroc = (
        roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        if len(set(all_labels)) == num_classes
        else float("nan")
    )
    return {"loss": total_loss / n, "accuracy": correct / n, "auroc": auroc}


def evaluate_confusion(model, loader, device) -> list:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for signals, lbls in loader:
            outputs = model(signals.to(device))
            preds.extend(outputs.argmax(1).cpu().tolist())
            labels.extend(lbls.tolist())
    return confusion_matrix(labels, preds).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to the folder with mitbih_train.csv/mitbih_test.csv; downloads via Kaggle if omitted.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="modalities/ecg/out")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else download()
    device = get_device()
    print(f"Using device: {device}")

    train_signals, train_labels = load_split(data_dir, "train")
    test_signals, test_labels = load_split(data_dir, "test")

    train_idx, val_idx = train_test_split(
        range(len(train_signals)), test_size=0.1, stratify=train_labels, random_state=42
    )
    train_ds = SignalDataset(train_signals[train_idx], train_labels[train_idx])
    val_ds = SignalDataset(train_signals[val_idx], train_labels[val_idx])
    test_ds = SignalDataset(test_signals, test_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = build_model(num_classes=len(CLASSES)).to(device)
    weights = class_weights(train_labels[train_idx], len(CLASSES)).to(device)
    print(f"Class weights (inverse frequency): {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "ecg_model.pt"
    best_auroc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, len(CLASSES), optimizer)
        val_metrics = run_epoch(model, val_loader, device, criterion, len(CLASSES))
        print(f"epoch {epoch}: train={train_metrics} val={val_metrics}")
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), weights_path)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    test_metrics = run_epoch(model, test_loader, device, criterion, len(CLASSES))
    cm = evaluate_confusion(model, test_loader, device)
    print(f"test metrics: {test_metrics}")
    print(f"confusion matrix (rows=true, cols=pred), classes={CLASSES}: {cm}")

    metadata = {
        "classes": CLASSES,
        "input_length": 187,
        "test_accuracy": test_metrics["accuracy"],
        "test_auroc": test_metrics["auroc"],
        "confusion_matrix": cm,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved weights and metadata to {out_dir}")


if __name__ == "__main__":
    main()

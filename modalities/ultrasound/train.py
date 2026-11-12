"""Train the breast ultrasound lesion classifier (3-class).

Single source of truth for training; modalities/ultrasound/notebook.ipynb
is a thin Colab wrapper that installs dependencies and calls this script.
Also runs fine as a plain local script, e.g. on an Apple Silicon Mac via
MPS, since BUSI is tiny (780 images total).

Usage:
    python -m modalities.ultrasound.train --epochs 15 --out modalities/ultrasound/out
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common.device import get_device
from modalities.ultrasound.data import CLASSES, download, list_samples
from modalities.ultrasound.model import build_model

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool) -> transforms.Compose:
    ops = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
    ]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class ImagePathDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def class_weights(targets, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights. BUSI is ~56% benign / 27%
    malignant / 17% normal; an unweighted loss biases toward benign."""
    counts = np.bincount(targets, minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, device, criterion, num_classes, optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, n = 0.0, 0, 0
    all_labels, all_probs = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        probs = torch.softmax(outputs, dim=1)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += images.size(0)
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
        for images, lbls in loader:
            outputs = model(images.to(device))
            preds.extend(outputs.argmax(1).cpu().tolist())
            labels.extend(lbls.tolist())
    return confusion_matrix(labels, preds).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to Dataset_BUSI_with_GT (normal/benign/malignant folders); downloads via Kaggle if omitted.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="modalities/ultrasound/out")
    args = parser.parse_args()

    image_root = Path(args.data_dir) if args.data_dir else download()
    device = get_device()
    print(f"Using device: {device}")

    # BUSI ships no train/val/test split at all (780 images total), so
    # carve out our own stratified 80/10/10 split.
    samples = list_samples(image_root)
    targets = [label for _, label in samples]
    train_samples, temp_samples, train_targets, temp_targets = train_test_split(
        samples, targets, test_size=0.2, stratify=targets, random_state=42
    )
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.5, stratify=temp_targets, random_state=42
    )
    print(f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    train_ds = ImagePathDataset(train_samples, build_transforms(train=True))
    val_ds = ImagePathDataset(val_samples, build_transforms(train=False))
    test_ds = ImagePathDataset(test_samples, build_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=2)

    model = build_model(num_classes=len(CLASSES)).to(device)
    weights = class_weights(train_targets, len(CLASSES)).to(device)
    print(f"Class weights (inverse frequency): {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "ultrasound_model.pt"
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
        "image_size": IMAGE_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "test_accuracy": test_metrics["accuracy"],
        "test_auroc": test_metrics["auroc"],
        "confusion_matrix": cm,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved weights and metadata to {out_dir}")


if __name__ == "__main__":
    main()

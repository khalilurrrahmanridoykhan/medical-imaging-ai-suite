"""1D CNN for ECG heartbeat arrhythmia classification (187-sample input).

Each MIT-BIH sample is already a short, fixed-length single-heartbeat
waveform, so a small from-scratch 1D CNN is used instead of a pretrained
backbone (no widely available pretrained model exists for this input).
"""

from torch import nn


class ECG1DCNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model(num_classes: int = 5) -> nn.Module:
    return ECG1DCNN(num_classes=num_classes)

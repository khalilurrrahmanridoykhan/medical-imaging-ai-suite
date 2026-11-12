"""3D CNN for nodule-patch malignancy classification (28x28x28 input).

NoduleMNIST3D's volumes are tiny, so a small from-scratch 3D CNN is used
instead of a pretrained backbone (no widely available 3D-CT pretrained
model exists for this input size).
"""

from torch import nn


class Nodule3DCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model(num_classes: int = 2) -> nn.Module:
    return Nodule3DCNN(num_classes=num_classes)

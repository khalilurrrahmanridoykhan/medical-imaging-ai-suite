"""Model definition for the breast ultrasound lesion classifier.

ResNet18 pretrained on ImageNet, fine-tuned with a replaced final layer
for 3-class classification (normal, benign, malignant).
"""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

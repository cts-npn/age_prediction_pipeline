"""Shared-backbone model with a classification head and a regression head."""
import torch
import torch.nn as nn
from torchvision import models


class DualHeadAgeModel(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            feat_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
        elif backbone == "mobilenet_v2":
            net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
            feat_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone '{backbone}'")

        self.backbone = net
        self.classifier_head = nn.Linear(feat_dim, num_classes)
        self.regression_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        class_logits = self.classifier_head(feats)
        reg_out = self.regression_head(feats).squeeze(1)
        return class_logits, reg_out

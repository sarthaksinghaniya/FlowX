from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class CrashSeverityModel(nn.Module):
    """
    Crash severity classifier for video input.

    Input:
        x: (B, T, C, H, W) typically (B, 16, 3, 224, 224)
    Output:
        logits: (B, 3)
    """

    def __init__(
        self,
        num_classes: int = 3,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        # Backbone: pretrained ResNet50 without final FC.
        if pretrained:
            try:
                backbone_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            except Exception:
                # Fallback if pretrained weights cannot be downloaded/loaded.
                backbone_model = resnet50(weights=None)
        else:
            backbone_model = resnet50(weights=None)

        self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])  # (N, 2048, 1, 1)
        self.feature_dim = 2048

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Temporal modeling.
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), got {tuple(x.shape)}")

        batch_size, timesteps, channels, height, width = x.shape
        if channels != 3:
            raise ValueError(f"Expected 3 input channels, got {channels}")

        # (B, T, C, H, W) -> (B*T, C, H, W)
        x = x.view(batch_size * timesteps, channels, height, width)

        # Frame-wise ResNet features: (B*T, 2048, 1, 1) -> (B*T, 2048)
        features = self.backbone(x).flatten(1)

        # (B*T, 2048) -> (B, T, 2048)
        features = features.view(batch_size, timesteps, self.feature_dim)

        # LSTM: output shape (B, T, hidden_size)
        lstm_out, _ = self.lstm(features)

        # Last timestep: (B, hidden_size)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)

        # Classification logits: (B, num_classes)
        logits = self.classifier(last_output)
        return logits


def _test_forward_pass() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CrashSeverityModel(
        num_classes=3,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)
    model.eval()

    dummy_input = torch.randn(2, 16, 3, 224, 224, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    _test_forward_pass()

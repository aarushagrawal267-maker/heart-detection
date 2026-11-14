"""The project's own 1D CNN for heartbeat classification (trained from scratch).

Architecture (default, 256-sample input):
    stage 1: 2 x [Conv1d(k=7) -> BatchNorm -> ReLU], MaxPool(2)     32 channels
    stage 2: 2 x [Conv1d(k=5) -> BatchNorm -> ReLU], MaxPool(2)     64 channels
    stage 3: 2 x [Conv1d(k=3) -> BatchNorm -> ReLU], MaxPool(2)    128 channels
    stage 4: Conv1d(k=3) -> BatchNorm -> ReLU  (Grad-CAM target), global average pool
    head:    concat RR features -> Linear(64) -> ReLU -> Dropout -> Linear(n_classes)

No pretrained weights are used anywhere: parameters start from PyTorch's default
random initialisation (seeded) and are learned only from MIT-BIH training subjects.
"""
from __future__ import annotations

import torch
from torch import nn


def conv_block(c_in: int, c_out: int, k: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, padding=k // 2, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
    )


class ECGNet(nn.Module):
    def __init__(self, n_classes: int = 4, n_rr: int = 4, use_rr: bool = True,
                 channels: tuple[int, int, int] = (32, 64, 128), dropout: float = 0.4):
        super().__init__()
        c1, c2, c3 = channels
        self.use_rr = use_rr
        self.n_rr = n_rr
        self.features = nn.Sequential(
            conv_block(1, c1, 7), conv_block(c1, c1, 7), nn.MaxPool1d(2),
            conv_block(c1, c2, 5), conv_block(c2, c2, 5), nn.MaxPool1d(2),
            conv_block(c2, c3, 3), conv_block(c3, c3, 3), nn.MaxPool1d(2),
        )
        self.last_conv = conv_block(c3, c3, 3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        head_in = c3 + (n_rr if use_rr else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.last_conv(self.features(x))

    def classify(self, fmap: torch.Tensor, rr: torch.Tensor | None) -> torch.Tensor:
        z = self.pool(fmap).flatten(1)
        if self.use_rr:
            if rr is None:
                raise ValueError("This model requires RR-interval features")
            z = torch.cat([z, rr], dim=1)
        return self.head(z)

    def forward(self, x: torch.Tensor, rr: torch.Tensor | None = None) -> torch.Tensor:
        """x: (batch, 1, length) z-scored segment; rr: (batch, n_rr) scaled RR features."""
        return self.classify(self.feature_maps(x), rr)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe_architecture(model: ECGNet, length: int) -> list[dict]:
    """Layer-by-layer summary for the documentation / Model Laboratory page."""
    rows = []
    x = torch.zeros(1, 1, length)
    with torch.no_grad():
        for layer in list(model.features) + [model.last_conv]:
            x = layer(x)
            name = "MaxPool1d(2)" if isinstance(layer, nn.MaxPool1d) else \
                f"Conv1d(k={layer[0].kernel_size[0]}) + BatchNorm + ReLU"
            rows.append({"layer": name, "output_shape": [int(x.shape[1]), int(x.shape[2])],
                         "params": sum(p.numel() for p in layer.parameters())})
    rows.append({"layer": "GlobalAvgPool", "output_shape": [int(x.shape[1])], "params": 0})
    if model.use_rr:
        rows.append({"layer": f"Concat RR features ({model.n_rr})",
                     "output_shape": [int(x.shape[1]) + model.n_rr], "params": 0})
    for layer in model.head:
        if isinstance(layer, nn.Linear):
            rows.append({"layer": f"Linear({layer.in_features}->{layer.out_features})",
                         "output_shape": [layer.out_features],
                         "params": sum(p.numel() for p in layer.parameters())})
        elif isinstance(layer, nn.Dropout):
            rows.append({"layer": f"Dropout({layer.p})", "output_shape": None, "params": 0})
    return rows

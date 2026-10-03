"""Small MLP head. Same architecture for M1 full and peptide-only."""

from __future__ import annotations

import torch
from torch import nn


class StabilityMLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, int] = (128, 64), dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

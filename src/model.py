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


class HeteroscedasticMLP(nn.Module):
    """Same trunk as StabilityMLP, with a second head for log variance.

    The variance is trained with a Gaussian negative log-likelihood, so the
    width is a function of the peptide and the contact residues rather than
    disagreement among ensemble members.
    """

    def __init__(self, n_in: int, hidden: tuple[int, int] = (128, 64), dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_in, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden[1], 1)
        self.log_var_head = nn.Linear(hidden[1], 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        mean = self.mean_head(hidden).squeeze(-1)
        log_var = self.log_var_head(hidden).squeeze(-1).clamp(-6.0, 4.0)
        return mean, log_var


def gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (0.5 * torch.exp(-log_var) * (y - mean).square() + 0.5 * log_var).mean()

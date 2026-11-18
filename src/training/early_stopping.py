"""Early stopping on validation AUROC.

Selection uses the patient-level validation split of the fold and never the test fold.
The best-scoring weights are restored before the test fold is scored, so the reported
prediction comes from the epoch chosen on validation rather than from the last epoch.
"""
from __future__ import annotations

import copy

import torch


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float("inf")
        self.best_epoch = -1
        self.best_state: dict | None = None
        self.epochs_without_improvement = 0

    def step(self, score: float, epoch: int, model: torch.nn.Module) -> bool:
        """Record an epoch. Returns True when training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

"""Fold-level training loop shared by every image experiment.

The protocol matches the tabular baselines (`src.training.tabular_cv`) exactly so that the
results are comparable on identical folds: fit on *train*, choose the stopping epoch and
the decision threshold on *val*, score *test* once. Nothing about the test fold influences
training, model selection, or thresholding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.models.single_view import set_train_mode
from src.training.early_stopping import EarlyStopping


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 8
    mixed_precision: bool = True
    num_workers: int = 4
    freeze_backbone: bool = False
    # Transfer-learning defaults, not values tuned on this cohort. A backbone learning rate
    # far below the head's is the standard recipe when adapting a large pretrained encoder to
    # a small target set, and a short linear warmup is the standard remedy for AdamW
    # instability in the first steps. Both were adopted because fine-tuning at a single
    # learning rate left 11 of 25 folds sitting on the decision boundary, having made only
    # 75 optimiser steps in total.
    backbone_lr_scale: float = 1.0
    warmup_epochs: int = 0
    # Early stopping on validation AUROC proved unusable on this cohort: with 15 validation
    # patients (4 positives, 44 pairs) the metric moves in steps of 1/44, and selecting its
    # maximum over epochs correlated with test AUROC at r = +0.06 while inflating the
    # validation estimate by +0.14. When disabled, the full epoch budget is run and the
    # final model is scored, which removes the selection step rather than refining it.
    early_stopping: bool = True
    history: list = field(default_factory=list)


def _loader(ds, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=workers > 0)


def _unpack(batch) -> tuple[list, torch.Tensor, list[str]]:
    """Split a batch into model inputs, target, identifier.

    A dataset yields `(*inputs, y, id)`. The single-view datasets yield one input tensor;
    the multi-view dataset yields two, the stacked views and a mask saying which of them
    the patient actually has. Unpacking positionally from the end keeps one fold loop for
    both rather than a second copy of the leakage-critical protocol.
    """
    *inputs, y, ids = batch
    return inputs, y, list(ids)


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader,
            device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Probabilities, labels and identifiers for every item of `loader`, in eval mode."""
    model.eval()
    probs, labels, ids = [], [], []
    for batch in loader:
        inputs, y, pid = _unpack(batch)
        inputs = [t.to(device, non_blocking=True) for t in inputs]
        logits = model(*inputs).squeeze(-1)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
        ids.extend(pid)
    return np.concatenate(probs), np.concatenate(labels), ids


@torch.no_grad()
def extract_features(model: torch.nn.Module, dataset, device: torch.device,
                     batch_size: int, workers: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Penultimate-layer representation of every item of `dataset`, under the current weights.

    Called after `train_fold` returns, so the weights are the ones that produced that fold's
    predictions -- the representation of a *fine-tuned* model, which is the thing the frozen
    era probes cannot speak for. `forward_head(..., pre_logits=True)` is timm's contract for
    "everything except the classifier", so it gives the pooled feature vector for ConvNeXt,
    Swin and ResNet alike without per-architecture surgery.

    Training is untouched: this is a forward pass in eval mode over an already-trained model.
    """
    model.eval()
    loader = _loader(dataset, batch_size, False, workers)
    features, labels, ids = [], [], []
    for batch in loader:
        inputs, y, pid = _unpack(batch)
        inputs = [t.to(device, non_blocking=True) for t in inputs]
        hidden = model.forward_features(*inputs)
        pooled = model.forward_head(hidden, pre_logits=True)
        features.append(pooled.float().cpu().numpy())
        labels.append(y.numpy())
        ids.extend(pid)
    return np.concatenate(features), np.concatenate(labels), ids


def train_fold(model: torch.nn.Module, train_ds, val_ds, test_ds, cfg: TrainConfig,
               device: torch.device, log_prefix: str = "") -> dict:
    """Train one fold and return validation and test predictions from the selected epoch."""
    train_loader = _loader(train_ds, cfg.batch_size, True, cfg.num_workers)
    val_loader = _loader(val_ds, cfg.batch_size, False, cfg.num_workers)
    test_loader = _loader(test_ds, cfg.batch_size, False, cfg.num_workers)

    # Class weighting from the training partition only.
    y_train = np.asarray(train_ds.labels, dtype=float)
    n_pos, n_neg = float(y_train.sum()), float(len(y_train) - y_train.sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.to(device)
    if cfg.backbone_lr_scale != 1.0:
        head_ids = {id(p) for p in model.get_classifier().parameters()}
        head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
        backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        groups = [{"params": head, "lr": cfg.learning_rate},
                  {"params": backbone, "lr": cfg.learning_rate * cfg.backbone_lr_scale}]
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": cfg.learning_rate}]
    optimiser = torch.optim.AdamW(groups, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(cfg.epochs - cfg.warmup_epochs, 1))
    if cfg.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimiser, start_factor=0.1, total_iters=cfg.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimiser, [warmup, cosine], milestones=[cfg.warmup_epochs])
    else:
        scheduler = cosine
    use_amp = cfg.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper = EarlyStopping(patience=cfg.patience)
    history = []

    for epoch in range(1, cfg.epochs + 1):
        set_train_mode(model, cfg.freeze_backbone)
        total, seen, stepped_any = 0.0, 0, False
        for batch in train_loader:
            inputs, y, _ = _unpack(batch)
            inputs = [t.to(device, non_blocking=True) for t in inputs]
            y = y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(*inputs).squeeze(-1), y)
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimiser)
            scaler.update()
            # GradScaler skips the update when the gradients overflow, which is normal while
            # the scale calibrates. Counting a skipped batch as an optimiser step would
            # advance the LR schedule without any weights having moved.
            stepped_any |= scaler.get_scale() >= scale_before
            total += float(loss.detach()) * len(y)
            seen += len(y)
        if stepped_any:
            scheduler.step()

        val_prob, val_y, _ = predict(model, val_loader, device)
        val_auc = roc_auc_score(val_y, val_prob) if len(np.unique(val_y)) == 2 else float("nan")
        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_auroc": val_auc})
        if log_prefix:
            print(f"{log_prefix} epoch {epoch:3d} loss {total / max(seen, 1):.4f} "
                  f"val_auroc {val_auc:.3f}", flush=True)
        if cfg.early_stopping:
            # A degenerate validation split yields NaN; treat it as no improvement.
            if stopper.step(val_auc if np.isfinite(val_auc) else -np.inf, epoch, model):
                break

    if cfg.early_stopping:
        stopper.restore(model)
    else:
        # No selection: the model scored is the one at the end of the fixed budget.
        stopper.best_epoch = cfg.epochs
        stopper.best_score = history[-1]["val_auroc"] if history else float("nan")
    val_prob, val_y, _ = predict(model, val_loader, device)
    test_prob, test_y, test_ids = predict(model, test_loader, device)
    return {
        "val_prob": val_prob, "val_y": val_y,
        "test_prob": test_prob, "test_y": test_y, "test_ids": test_ids,
        "best_epoch": stopper.best_epoch, "best_val_auroc": stopper.best_score,
        "epochs_run": len(history), "history": history,
    }

"""The pretrained image encoder every single-view experiment fine-tunes or probes.

One encoder, ConvNeXt-Tiny, fixed in advance. With 35 malignant patients a patient-level
bootstrap interval on AUROC spans roughly +/-0.10, so comparing several architectures on
this cohort would rank fold noise, and reporting the best of them would bias the estimate
upwards through selection. Additional encoders are a secondary, exploratory analysis, run
only if the pre-specified one clears the metadata and asymmetry reference baselines.

torch and timm are imported inside the functions that need them, so that a script which
only wants `DEFAULT_ENCODER` for its command line does not require either.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

DEFAULT_ENCODER = "convnext_tiny"


def fetch_pretrained_state(encoder: str = DEFAULT_ENCODER,
                           cache_path: Path | None = None) -> dict:
    """Obtain the pretrained weights once, caching them on local disk.

    Building a pretrained model inside the fold loop contacts the model hub 25 times, and a
    single network failure then aborts a run that is otherwise minutes from finishing --
    which is exactly what happened on the first attempt. The weights are fetched once here,
    written to `cache_path`, and reused for every fold, so a rerun needs no network at all.

    A consequence worth recording: the randomly initialised classifier head is also cached,
    so every fold starts from the same head. That removes a source of between-fold variance
    which is not of scientific interest; the variance that matters comes from the data
    partition, which still differs across folds and repeats.
    """
    import timm
    import torch

    if cache_path is not None and Path(cache_path).exists():
        print(f"Loading cached pretrained weights from {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    try:
        model = timm.create_model(encoder, pretrained=True, num_classes=1, in_chans=3)
    except Exception as exc:
        raise SystemExit(
            f"Could not download pretrained weights for {encoder!r}: {type(exc).__name__}: {exc}\n"
            "The model hub was unreachable. Retry when the network is available; the weights "
            "are cached on first success and no later run will need the network."
        ) from exc
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, cache_path)
        print(f"Cached pretrained weights to {cache_path}", flush=True)
    return state


def build_model_from_state(state: dict, encoder: str = DEFAULT_ENCODER, drop_rate: float = 0.0,
                           freeze_backbone: bool = False) -> torch.nn.Module:
    """Build a fold's model from cached weights, without touching the network.

    With `freeze_backbone`, only the classifier is trainable -- a linear probe. On 80
    training patients, fine-tuning 28M parameters collapsed to a near-constant output on
    14 of 25 folds, so the low-capacity condition is not a fallback but a pre-specified
    arm of the comparison.
    """
    import timm

    model = timm.create_model(encoder, pretrained=False, num_classes=1,
                              in_chans=3, drop_rate=drop_rate)
    model.load_state_dict(state)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True
    return model


def set_train_mode(model: torch.nn.Module, freeze_backbone: bool) -> None:
    """Put the model in training mode, keeping a frozen backbone in eval.

    A frozen backbone must not have dropout or stochastic depth active: those would inject
    noise into features the probe cannot adapt to, and would make the extracted
    representation differ between training and scoring.
    """
    if freeze_backbone:
        model.eval()
        model.get_classifier().train()
    else:
        model.train()


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""Training-only augmentation, shared by every image experiment.

Applied after the patient-level split and to training patients only.

**Horizontal flip is deliberately excluded.** In a thermogram of the thorax, mirroring
exchanges the left and right breast. The cohort's disease is unilateral in every patient
and the view naming encodes anatomical side (see data/README.md), so a horizontal flip
would assert that laterality carries no information -- a substantive claim about the
modality, not a neutral regularisation. It is therefore not used, and any later experiment
that enables it must report it as a change of hypothesis rather than of hyperparameter.

What remains is small rigid-body and scale variation, which models plausible differences
in patient positioning relative to a fixed camera. Photometric jitter is also excluded:
each image is already standardised individually, so brightness and contrast carry no
absolute meaning to perturb.
"""
from __future__ import annotations

import torch
from torchvision.transforms import v2


def build_train_augmentation(degrees: float = 10.0, translate: float = 0.05,
                             scale: tuple[float, float] = (0.9, 1.1)) -> torch.nn.Module:
    """Geometric augmentation only. Fill is 0, the mean of a standardised image."""
    return v2.RandomAffine(
        degrees=degrees,
        translate=(translate, translate),
        scale=scale,
        interpolation=v2.InterpolationMode.BILINEAR,
        fill=0.0,
    )


def describe(degrees: float, translate: float, scale: tuple[float, float]) -> dict:
    """Machine-readable record of what was applied, for the experiment config."""
    return {
        "random_affine_degrees": degrees,
        "random_affine_translate": translate,
        "random_affine_scale": list(scale),
        "horizontal_flip": False,
        "horizontal_flip_rationale": "mirroring exchanges left and right breast; "
                                     "disease is unilateral",
        "photometric_jitter": False,
        "photometric_jitter_rationale": "images are standardised individually; "
                                        "absolute intensity is not meaningful",
    }

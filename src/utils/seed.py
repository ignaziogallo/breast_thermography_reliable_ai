"""One seeding call for every source of randomness an experiment touches."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and, when it is usable, torch.

    torch is optional on purpose: the analysis scripts run on a workstation with no
    deep-learning stack, and they must seed identically there. The guard is deliberately
    broad, because a missing torch raises ImportError while a broken install (an absent
    CUDA library, for instance) raises OSError, and neither may stop a CPU-only analysis.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001 - see the docstring
        pass

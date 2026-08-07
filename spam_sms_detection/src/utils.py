from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    for folder in ["models", "outputs", "images", "reports"]:
        (PROJECT_ROOT / folder).mkdir(exist_ok=True)


def project_name() -> str:
    return "Spam SMS Detection"

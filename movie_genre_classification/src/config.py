from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

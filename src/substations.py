"""Load configurable substation display metadata."""

import json
from pathlib import Path

from .data_loader import CHANNELS


def load_substations(path: str | Path = "config/substations.json") -> dict:
    """Load substation metadata without changing the model's channel schema."""
    with Path(path).open(encoding="utf-8") as file:
        configs = json.load(file)
    for substation_id, config in configs.items():
        if set(config.get("parameters", {})) != set(CHANNELS):
            raise ValueError(f"{substation_id} must define every channel")
    return configs
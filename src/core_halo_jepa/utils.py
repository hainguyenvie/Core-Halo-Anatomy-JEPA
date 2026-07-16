from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def image_hw(image_size: int | list[int] | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return image_size, image_size
    return int(image_size[0]), int(image_size[1])


def cosine_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    start: float,
    end: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return start * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def linear_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    alpha = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return start + alpha * (end - start)


def dump_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "name": "core_halo",
    "seed": 42,
    "output_dir": "outputs/core_halo",
    "device": "auto",
    "data": {
        "kind": "synthetic",
        "manifest": None,
        "image_size": 128,
        "batch_size": 16,
        "num_workers": 0,
        "train_samples": 512,
        "calibration_samples": 128,
        "test_samples": 256,
        "test_anomaly_fraction": 0.5,
        "min_brain_fraction": 0.05,
        "strict_healthy_train": True,
    },
    "model": {
        "in_channels": 1,
        "patch_size": 8,
        "embed_dim": 96,
        "encoder_depth": 3,
        "predictor_depth": 2,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "core_size": 2,
        "halo_size": 1,
        "context_radius": 5,
        "include_contralateral": True,
        "anatomy_mode": None,
        "contralateral_size": 2,
    },
    "train": {
        "epochs": 20,
        "lr": 3.0e-4,
        "min_lr": 1.0e-6,
        "weight_decay": 0.05,
        "warmup_epochs": 2,
        "ema_decay_start": 0.99,
        "ema_decay_end": 0.9995,
        "targets_per_image": 2,
        "cosine_loss_weight": 0.1,
        "grad_clip": 1.0,
        "amp": True,
        "log_every": 20,
    },
    "score": {
        "geometry_stride": 1,
        "geometry_batch_size": 64,
        "variance_floor": 1.0e-4,
        "shrinkage": 0.1,
        "threshold_percentile": 99.5,
        "smoothing_sigma": 0.0,
        "image_score_percentile": 99.0,
        "save_examples": 12,
    },
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = _deep_update(copy.deepcopy(DEFAULT_CONFIG), loaded)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, received: {item!r}")
        key, raw_value = item.split("=", 1)
        _set_dotted(config, key, yaml.safe_load(raw_value))
    validate_config(config)
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    data = config["data"]
    model = config["model"]
    image_size = data["image_size"]
    if isinstance(image_size, int):
        height = width = image_size
    else:
        height, width = image_size
    patch_size = int(model["patch_size"])
    if height % patch_size or width % patch_size:
        raise ValueError("image_size must be divisible by model.patch_size")
    if int(model["embed_dim"]) % int(model["num_heads"]):
        raise ValueError("model.embed_dim must be divisible by model.num_heads")
    if int(model["embed_dim"]) % 4:
        raise ValueError("model.embed_dim must be divisible by 4 for 2-D positional encoding")
    if int(model["core_size"]) < 1:
        raise ValueError("model.core_size must be >= 1")
    if int(model["halo_size"]) < 0:
        raise ValueError("model.halo_size must be >= 0")
    if int(model["context_radius"]) <= int(model["halo_size"]):
        raise ValueError("context_radius must be larger than halo_size")
    anatomy_mode = model.get("anatomy_mode")
    valid_modes = {None, "none", "mirror", "random_same_subject", "cross_subject_mirror"}
    if anatomy_mode not in valid_modes:
        raise ValueError(f"model.anatomy_mode must be one of {sorted(map(str, valid_modes))}")
    if data["kind"] not in {"synthetic", "slice_manifest"}:
        raise ValueError("data.kind must be 'synthetic' or 'slice_manifest'")
    if data["kind"] == "slice_manifest" and not data.get("manifest"):
        raise ValueError("data.manifest is required for slice_manifest data")


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False)

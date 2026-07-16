from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset

from .utils import image_hw


def robust_normalize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.all():
        image = np.where(finite, image, 0.0)
    foreground = image != 0
    if foreground.sum() < 16:
        foreground = np.ones_like(image, dtype=bool)
    values = image[foreground]
    low, high = np.percentile(values, [1.0, 99.0])
    clipped = np.clip(image, low, high)
    mean = float(clipped[foreground].mean())
    std = float(clipped[foreground].std())
    normalized = (clipped - mean) / max(std, 1.0e-6)
    normalized = np.clip(normalized, -4.0, 4.0) / 4.0
    if not np.all(foreground):
        normalized[~foreground] = 0.0
    return normalized.astype(np.float32)


def _resize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()[None, None]
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()[None, None]
    image_tensor = F.interpolate(image_tensor, size=size, mode="bilinear", align_corners=False)
    mask_tensor = F.interpolate(mask_tensor, size=size, mode="nearest")
    return image_tensor[0, 0].numpy(), (mask_tensor[0, 0].numpy() > 0.5).astype(np.uint8)


class SyntheticBrainDataset(Dataset):
    """Deterministic symmetric pseudo-MRI with held-out unilateral lesions.

    This is a mechanism test, not a claim of clinical performance. It gives the
    three context rules the same training subjects and known lesion masks.
    """

    def __init__(
        self,
        size: int,
        image_size: int | list[int],
        seed: int,
        anomaly_fraction: float,
        split: str,
    ) -> None:
        self.size = int(size)
        self.height, self.width = image_hw(image_size)
        self.seed = int(seed)
        self.anomaly_fraction = float(anomaly_fraction)
        self.split = split

    def __len__(self) -> int:
        return self.size

    def _healthy_brain(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        height, width = self.height, self.width
        yy, xx = np.mgrid[0:height, 0:width]
        center_y = height * (0.50 + rng.normal(0, 0.008))
        center_x = width * 0.50
        radius_y = height * rng.uniform(0.37, 0.42)
        radius_x = width * rng.uniform(0.31, 0.36)
        normalized_radius = ((yy - center_y) / radius_y) ** 2 + ((xx - center_x) / radius_x) ** 2
        brain = normalized_radius <= 1.0

        radial = np.clip(1.0 - normalized_radius, 0.0, 1.0)
        image = 0.22 + 0.55 * radial + 0.12 * np.cos(3.0 * np.pi * normalized_radius)
        image *= brain

        # Paired tissue nuclei enforce homologous left/right structure.
        for _ in range(3):
            offset_x = width * rng.uniform(0.09, 0.22)
            offset_y = height * rng.uniform(-0.18, 0.18)
            sigma_x = width * rng.uniform(0.025, 0.055)
            sigma_y = height * rng.uniform(0.025, 0.060)
            amplitude = rng.uniform(-0.18, 0.18)
            for sign in (-1.0, 1.0):
                blob = np.exp(
                    -0.5
                    * (
                        ((xx - (center_x + sign * offset_x)) / sigma_x) ** 2
                        + ((yy - (center_y + offset_y)) / sigma_y) ** 2
                    )
                )
                image += amplitude * blob * brain

        # Symmetric ventricles.
        vent_y = center_y - 0.03 * height
        for sign in (-1.0, 1.0):
            vent_x = center_x + sign * 0.07 * width
            vent = (
                ((xx - vent_x) / (0.035 * width)) ** 2 + ((yy - vent_y) / (0.09 * height)) ** 2
            ) <= 1
            image[vent] *= rng.uniform(0.35, 0.55)

        bias = 1.0 + rng.uniform(-0.12, 0.12) * (xx - center_x) / max(width, 1)
        image = gaussian_filter(image * bias, sigma=rng.uniform(0.45, 0.9))
        image += rng.normal(0.0, 0.012, image.shape) * brain
        image[~brain] = 0.0
        return image.astype(np.float32), brain

    def _inject_lesion(
        self,
        image: np.ndarray,
        brain: np.ndarray,
        rng: np.random.Generator,
        lesion_size: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape
        yy, xx = np.mgrid[0:height, 0:width]
        scale = {"small": 0.018, "medium": 0.032, "large": 0.055}[lesion_size]
        sigma_y = height * scale * rng.uniform(0.8, 1.25)
        sigma_x = width * scale * rng.uniform(0.8, 1.25)
        side = rng.choice([-1.0, 1.0])
        center_x = width * (0.5 + side * rng.uniform(0.10, 0.24))
        center_y = height * rng.uniform(0.30, 0.70)
        gaussian = np.exp(
            -0.5 * (((xx - center_x) / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2)
        )
        mask = (gaussian > np.exp(-2.0)) & brain
        polarity = rng.choice([-1.0, 1.0])
        amplitude = polarity * rng.uniform(0.35, 0.65)
        lesion_image = image + amplitude * gaussian * brain
        lesion_image[~brain] = 0.0
        return lesion_image.astype(np.float32), mask.astype(np.uint8)

    def __getitem__(self, index: int) -> dict[str, Any]:
        split_offset = {"train": 0, "calibration": 100_000, "test": 200_000}[self.split]
        rng = np.random.default_rng(self.seed + split_offset + index)
        image, brain = self._healthy_brain(rng)
        is_anomaly = self.split == "test" and rng.random() < self.anomaly_fraction
        lesion_size = "healthy"
        mask = np.zeros_like(image, dtype=np.uint8)
        if is_anomaly:
            lesion_size = ("small", "medium", "large")[index % 3]
            image, mask = self._inject_lesion(image, brain, rng, lesion_size)
        image = robust_normalize(image)
        return {
            "image": torch.from_numpy(image)[None],
            "mask": torch.from_numpy(mask.astype(np.float32))[None],
            "sample_id": f"synthetic-{self.split}-{index:05d}",
            "lesion_size": lesion_size,
        }


def _load_array(path: Path, key: str | None = None) -> np.ndarray:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        return np.load(path, allow_pickle=False)
    if suffixes.endswith(".npz"):
        archive = np.load(path, allow_pickle=False)
        if key and key in archive:
            return archive[key]
        if len(archive.files) == 1:
            return archive[archive.files[0]]
        raise KeyError(f"{path} has multiple arrays; expected key {key!r}")
    raise ValueError(f"Unsupported slice format for {path}; use .npy or .npz")


class SliceManifestDataset(Dataset):
    """2-D slice dataset described by a portable CSV manifest.

    Required columns: image, split. Optional columns: mask, sample_id,
    patient_id, lesion_size. Paths may be absolute or relative to the manifest.
    """

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        image_size: int | list[int],
        strict_healthy_train: bool = True,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.split = split
        self.size = image_hw(image_size)
        self.strict_healthy_train = strict_healthy_train
        with self.manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        split_aliases = {"calibration", "val", "validation"} if split == "calibration" else {split}
        self.rows = [row for row in rows if row.get("split", "").lower() in split_aliases]
        if not self.rows:
            raise ValueError(f"No rows for split {split!r} in {self.manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self._path(row["image"])
        if image_path.suffix.lower() == ".npz":
            image = _load_array(image_path, "image")
        else:
            image = _load_array(image_path)
        if row.get("mask"):
            mask_path = self._path(row["mask"])
            mask = _load_array(mask_path, "mask" if mask_path.suffix.lower() == ".npz" else None)
        elif image_path.suffix.lower() == ".npz":
            archive = np.load(image_path, allow_pickle=False)
            mask = archive["mask"] if "mask" in archive else np.zeros_like(image)
        else:
            mask = np.zeros_like(image)
        image = np.squeeze(np.asarray(image))
        mask = np.squeeze(np.asarray(mask))
        if image.ndim != 2 or mask.ndim != 2:
            raise ValueError(f"Expected 2-D image/mask in row {index}: {image_path}")
        image, mask = _resize_pair(image, mask, self.size)
        image = robust_normalize(image)
        if self.split == "train" and self.strict_healthy_train and mask.any():
            raise ValueError(f"Training row contains a positive lesion mask: {image_path}")
        return {
            "image": torch.from_numpy(image)[None],
            "mask": torch.from_numpy(mask.astype(np.float32))[None],
            "sample_id": row.get("sample_id") or row.get("patient_id") or image_path.stem,
            "lesion_size": row.get("lesion_size") or "unknown",
        }


def build_dataloaders(config: dict, seed: int) -> dict[str, DataLoader]:
    kind = config["kind"]
    if kind == "synthetic":
        datasets: dict[str, Dataset] = {
            "train": SyntheticBrainDataset(
                config["train_samples"], config["image_size"], seed, 0.0, "train"
            ),
            "calibration": SyntheticBrainDataset(
                config["calibration_samples"],
                config["image_size"],
                seed,
                0.0,
                "calibration",
            ),
            "test": SyntheticBrainDataset(
                config["test_samples"],
                config["image_size"],
                seed,
                config["test_anomaly_fraction"],
                "test",
            ),
        }
    else:
        datasets = {
            split: SliceManifestDataset(
                config["manifest"],
                split,
                config["image_size"],
                strict_healthy_train=bool(config["strict_healthy_train"]),
            )
            for split in ("train", "calibration", "test")
        }
    generator = torch.Generator().manual_seed(seed)
    common = dict(
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config["num_workers"]) > 0,
    )
    return {
        "train": DataLoader(datasets["train"], shuffle=True, generator=generator, **common),
        "calibration": DataLoader(datasets["calibration"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from .geometry import enumerate_geometries, patch_foreground
from .model import CoreHaloJEPA


@dataclass
class ResidualCalibrator:
    """Coordinate-wise diagonal Mahalanobis calibration on healthy residuals."""

    mean: torch.Tensor
    variance: torch.Tensor
    count: torch.Tensor
    variance_floor: float

    @classmethod
    def empty(cls, token_count: int, dimension: int, variance_floor: float) -> ResidualCalibrator:
        return cls(
            mean=torch.zeros(token_count, dimension, dtype=torch.float64),
            variance=torch.zeros(token_count, dimension, dtype=torch.float64),
            count=torch.zeros(token_count, dtype=torch.float64),
            variance_floor=variance_floor,
        )

    def update(self, residual: torch.Tensor, token_indices: torch.Tensor) -> None:
        residual = (
            residual.detach().to(device="cpu", dtype=torch.float64).reshape(-1, residual.shape[-1])
        )
        token_indices = token_indices.detach().to(device="cpu").reshape(-1)
        ones = torch.ones_like(token_indices, dtype=torch.float64)
        self.mean.index_add_(0, token_indices, residual)
        self.variance.index_add_(0, token_indices, residual.square())
        self.count.index_add_(0, token_indices, ones)

    def finalize(self, shrinkage: float) -> None:
        observed = self.count > 0
        total_count = self.count.sum().clamp_min(1.0)
        global_mean = self.mean.sum(dim=0) / total_count
        global_second = self.variance.sum(dim=0) / total_count
        global_variance = (global_second - global_mean.square()).clamp_min(self.variance_floor)

        counts = self.count.clamp_min(1.0).unsqueeze(-1)
        coordinate_mean = self.mean / counts
        coordinate_variance = (self.variance / counts - coordinate_mean.square()).clamp_min(0.0)
        coordinate_mean[~observed] = global_mean
        coordinate_variance[~observed] = global_variance
        coordinate_variance = (
            (1.0 - shrinkage) * coordinate_variance + shrinkage * global_variance
        ).clamp_min(self.variance_floor)
        self.mean = coordinate_mean.float()
        self.variance = coordinate_variance.float()
        self.count = self.count.float()

    def score(self, residual: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=residual.device, dtype=residual.dtype)[token_indices]
        variance = self.variance.to(device=residual.device, dtype=residual.dtype)[token_indices]
        return ((residual - mean).square() / variance).mean(dim=-1)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.cpu(),
            "variance": self.variance.cpu(),
            "count": self.count.cpu(),
            "variance_floor": self.variance_floor,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> ResidualCalibrator:
        return cls(
            mean=state["mean"].cpu(),
            variance=state["variance"].cpu(),
            count=state["count"].cpu(),
            variance_floor=float(state["variance_floor"]),
        )


def geometry_kwargs(model_config: dict, score_config: dict) -> dict:
    return {
        "core_size": int(model_config["core_size"]),
        "halo_size": int(model_config["halo_size"]),
        "context_radius": int(model_config["context_radius"]),
        "include_contralateral": bool(model_config["include_contralateral"]),
        "contralateral_size": int(model_config["contralateral_size"]),
        "stride": int(score_config["geometry_stride"]),
    }


@torch.inference_mode()
def fit_calibrator(
    model: CoreHaloJEPA,
    loader: Iterable,
    model_config: dict,
    score_config: dict,
    device: torch.device,
) -> ResidualCalibrator:
    model.eval()
    calibrator: ResidualCalibrator | None = None
    chunk_size = int(score_config["geometry_batch_size"])
    kwargs = geometry_kwargs(model_config, score_config)
    for batch in tqdm(loader, desc="Calibrating healthy residuals", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        foreground = patch_foreground(images, model.patch_size).cpu()
        grid_size = foreground.shape[-2:]
        if calibrator is None:
            calibrator = ResidualCalibrator.empty(
                token_count=grid_size[0] * grid_size[1],
                dimension=int(model_config["embed_dim"]),
                variance_floor=float(score_config["variance_floor"]),
            )
        for image_index in range(images.shape[0]):
            geometries = enumerate_geometries(
                grid_size=grid_size,
                foreground=foreground[image_index],
                **kwargs,
            )
            source = images[image_index : image_index + 1]
            for start in range(0, len(geometries), chunk_size):
                chunk = geometries[start : start + chunk_size]
                mapping = torch.zeros(len(chunk), dtype=torch.long, device=device)
                output = model(source, chunk, mapping)
                calibrator.update(output.residual, output.target_indices)
    if calibrator is None:
        raise RuntimeError("Calibration loader was empty")
    calibrator.finalize(float(score_config["shrinkage"]))
    return calibrator


@torch.inference_mode()
def score_image(
    model: CoreHaloJEPA,
    image: torch.Tensor,
    calibrator: ResidualCalibrator,
    model_config: dict,
    score_config: dict,
) -> torch.Tensor:
    """Return one pixel-level anomaly map with the same HxW as the input."""

    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.shape[0] != 1:
        raise ValueError("score_image accepts exactly one image")
    device = next(model.parameters()).device
    image = image.to(device)
    foreground = patch_foreground(image, model.patch_size)[0]
    grid_size = tuple(int(value) for value in foreground.shape)
    geometries = enumerate_geometries(
        grid_size=grid_size,
        foreground=foreground.cpu(),
        **geometry_kwargs(model_config, score_config),
    )
    token_count = grid_size[0] * grid_size[1]
    score_sum = torch.zeros(token_count, device=device)
    score_count = torch.zeros(token_count, device=device)
    chunk_size = int(score_config["geometry_batch_size"])
    for start in range(0, len(geometries), chunk_size):
        chunk = geometries[start : start + chunk_size]
        mapping = torch.zeros(len(chunk), dtype=torch.long, device=device)
        output = model(image, chunk, mapping)
        token_scores = calibrator.score(output.residual, output.target_indices)
        flat_indices = output.target_indices.reshape(-1)
        score_sum.index_add_(0, flat_indices, token_scores.reshape(-1))
        score_count.index_add_(0, flat_indices, torch.ones_like(token_scores).reshape(-1))
    token_map = (score_sum / score_count.clamp_min(1.0)).reshape(grid_size)
    token_map = token_map * foreground.to(token_map.dtype)
    pixel_map = F.interpolate(
        token_map[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    sigma = float(score_config.get("smoothing_sigma", 0.0))
    if sigma > 0:
        smoothed = gaussian_filter(pixel_map.detach().cpu().numpy(), sigma=sigma)
        pixel_map = torch.from_numpy(np.asarray(smoothed)).to(pixel_map)
    return pixel_map.detach().cpu()

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .config import save_resolved_config
from .data import build_dataloaders
from .geometry import patch_foreground, sample_geometries
from .metrics import compute_metrics, threshold_from_healthy
from .model import CoreHaloJEPA, JepaOutput, build_model
from .scoring import ResidualCalibrator, fit_calibrator, score_image
from .utils import (
    atomic_torch_save,
    cosine_with_warmup,
    dump_json,
    linear_schedule,
    resolve_device,
    seed_everything,
)


def jepa_loss(output: JepaOutput, cosine_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    regression = F.smooth_l1_loss(output.prediction, output.target)
    cosine = 1.0 - F.cosine_similarity(output.prediction, output.target, dim=-1).mean()
    loss = regression + cosine_weight * cosine
    diagnostics = {
        "loss": float(loss.detach()),
        "regression_loss": float(regression.detach()),
        "cosine_loss": float(cosine.detach()),
        "target_std": float(output.target.std(dim=(0, 1)).mean().detach()),
        "prediction_std": float(output.prediction.std(dim=(0, 1)).mean().detach()),
    }
    return loss, diagnostics


def _geometry_args(model_config: dict) -> dict[str, Any]:
    return {
        "core_size": int(model_config["core_size"]),
        "halo_size": int(model_config["halo_size"]),
        "context_radius": int(model_config["context_radius"]),
        "include_contralateral": bool(model_config["include_contralateral"]),
        "contralateral_size": int(model_config["contralateral_size"]),
    }


@torch.inference_mode()
def latent_validation_loss(
    model: CoreHaloJEPA,
    loader,
    config: dict,
    device: torch.device,
    max_batches: int = 8,
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(int(config["seed"]) + 17)
    values = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        foreground = patch_foreground(images, model.patch_size).cpu()
        geometries, image_indices = sample_geometries(
            foreground,
            count_per_image=1,
            generator=generator,
            **_geometry_args(config["model"]),
        )
        output = model(images, geometries, image_indices)
        loss, _ = jepa_loss(output, float(config["train"]["cosine_loss_weight"]))
        values.append(float(loss))
    model.train()
    return float(np.mean(values)) if values else float("nan")


def checkpoint_payload(
    model: CoreHaloJEPA,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: dict,
    epoch: int,
    global_step: int,
    best_validation: float,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "epoch": epoch,
        "global_step": global_step,
        "best_validation": best_validation,
    }


def run_training(config: dict, resume: str | Path | None = None) -> Path:
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    loaders = build_dataloaders(config["data"], int(config["seed"]))
    model = build_model(config["model"]).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    else:  # PyTorch 2.1 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    global_step = 0
    best_validation = float("inf")
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_validation = float(state.get("best_validation", best_validation))

    epochs = int(config["train"]["epochs"])
    total_steps = max(1, epochs * len(loaders["train"]))
    warmup_steps = int(config["train"]["warmup_epochs"]) * len(loaders["train"])
    geometry_generator = torch.Generator().manual_seed(int(config["seed"]) + 1)
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        running: dict[str, float] = {}
        progress = tqdm(loaders["train"], desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_index, batch in enumerate(progress):
            images = batch["image"].to(device, non_blocking=True)
            if bool(config["data"]["strict_healthy_train"]) and bool(batch["mask"].any()):
                raise ValueError("Training batch contains lesion-positive pixels")
            foreground = patch_foreground(images, model.patch_size).cpu()
            geometries, image_indices = sample_geometries(
                foreground,
                count_per_image=int(config["train"]["targets_per_image"]),
                generator=geometry_generator,
                **_geometry_args(config["model"]),
            )
            learning_rate = cosine_with_warmup(
                global_step,
                total_steps,
                warmup_steps,
                float(config["train"]["lr"]),
                float(config["train"]["min_lr"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(images, geometries, image_indices)
                loss, diagnostics = jepa_loss(output, float(config["train"]["cosine_loss_weight"]))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, float(config["train"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            ema_decay = linear_schedule(
                global_step,
                total_steps,
                float(config["train"]["ema_decay_start"]),
                float(config["train"]["ema_decay_end"]),
            )
            model.update_target_encoder(ema_decay)
            global_step += 1
            diagnostics.update({"lr": learning_rate, "ema_decay": ema_decay})
            for key, value in diagnostics.items():
                running[key] = running.get(key, 0.0) + value
            if batch_index % int(config["train"]["log_every"]) == 0:
                progress.set_postfix(loss=f"{diagnostics['loss']:.4f}", lr=f"{learning_rate:.2e}")

        epoch_metrics = {key: value / max(len(progress), 1) for key, value in running.items()}
        epoch_metrics["epoch"] = float(epoch + 1)
        epoch_metrics["validation_loss"] = latent_validation_loss(
            model, loaders["calibration"], config, device
        )
        history.append(epoch_metrics)
        dump_json(history, output_dir / "train_history.json")
        payload = checkpoint_payload(
            model, optimizer, scaler, config, epoch, global_step, best_validation
        )
        atomic_torch_save(payload, output_dir / "last.pt")
        if epoch_metrics["validation_loss"] < best_validation:
            best_validation = epoch_metrics["validation_loss"]
            payload["best_validation"] = best_validation
            atomic_torch_save(payload, output_dir / "best.pt")

    dump_json(
        {
            "device": str(device),
            "seconds": time.time() - started,
            "epochs": epochs,
            "global_step": global_step,
            "best_validation_loss": best_validation,
        },
        output_dir / "training_summary.json",
    )
    return output_dir / "best.pt"


@torch.inference_mode()
def _collect_maps(
    model: CoreHaloJEPA,
    loader,
    calibrator: ResidualCalibrator,
    config: dict,
    description: str,
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str], list[str]
]:
    scores: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    brains: list[np.ndarray] = []
    images_out: list[np.ndarray] = []
    lesion_sizes: list[str] = []
    sample_ids: list[str] = []
    for batch in tqdm(loader, desc=description, leave=False):
        images = batch["image"]
        for index in range(images.shape[0]):
            image = images[index]
            score = score_image(
                model,
                image,
                calibrator,
                config["model"],
                config["score"],
            ).numpy()
            image_np = image[0].numpy()
            scores.append(score)
            masks.append(batch["mask"][index, 0].numpy().astype(np.uint8))
            brains.append((np.abs(image_np) > 1.0e-6).astype(np.uint8))
            images_out.append(image_np)
            lesion_sizes.append(str(batch["lesion_size"][index]))
            sample_ids.append(str(batch["sample_id"][index]))
    return scores, masks, brains, images_out, lesion_sizes, sample_ids


def load_model_checkpoint(
    checkpoint: str | Path, device: torch.device
) -> tuple[CoreHaloJEPA, dict]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "config" not in state:
        raise KeyError("Checkpoint does not contain its resolved config")
    model = build_model(state["config"]["model"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, state


def run_evaluation(config: dict, checkpoint: str | Path) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_dataloaders(config["data"], int(config["seed"]))
    model, _ = load_model_checkpoint(checkpoint, device)
    calibrator = fit_calibrator(
        model,
        loaders["calibration"],
        config["model"],
        config["score"],
        device,
    )
    atomic_torch_save(calibrator.state_dict(), output_dir / "calibrator.pt")

    calibration = _collect_maps(
        model, loaders["calibration"], calibrator, config, "Scoring calibration set"
    )
    threshold = threshold_from_healthy(
        calibration[0],
        calibration[2],
        float(config["score"]["threshold_percentile"]),
    )
    test = _collect_maps(model, loaders["test"], calibrator, config, "Scoring test set")
    metrics = compute_metrics(
        score_maps=test[0],
        lesion_masks=test[1],
        brain_masks=test[2],
        lesion_sizes=test[4],
        threshold=threshold,
        image_score_percentile=float(config["score"]["image_score_percentile"]),
    )
    metrics.update(
        {
            "experiment": config["name"],
            "seed": int(config["seed"]),
            "checkpoint": str(Path(checkpoint).resolve()),
            "device": str(device),
        }
    )
    dump_json(metrics, output_dir / "metrics.json")

    example_count = min(int(config["score"].get("save_examples", 0)), len(test[0]))
    if example_count:
        np.savez_compressed(
            output_dir / "examples.npz",
            images=np.stack(test[3][:example_count]),
            masks=np.stack(test[1][:example_count]),
            scores=np.stack(test[0][:example_count]),
            sample_ids=np.asarray(test[5][:example_count]),
        )
    return metrics

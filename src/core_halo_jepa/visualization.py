from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _gray(image: np.ndarray) -> np.ndarray:
    foreground = np.abs(image) > 1.0e-6
    values = image[foreground] if foreground.any() else image.reshape(-1)
    low, high = np.percentile(values, [1.0, 99.0])
    scaled = np.clip((image - low) / max(high - low, 1.0e-6), 0.0, 1.0)
    return np.repeat((scaled * 255).astype(np.uint8)[..., None], 3, axis=-1)


def _heatmap(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = score[score > 0]
    high = np.percentile(positive, 99.0) if positive.size else 1.0
    value = np.clip(score / max(float(high), 1.0e-6), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8), value


def _labelled(array: np.ndarray, label: str, band: int = 20) -> Image.Image:
    height, width = array.shape[:2]
    canvas = Image.new("RGB", (width, height + band), "white")
    canvas.paste(Image.fromarray(array), (0, band))
    ImageDraw.Draw(canvas).text((4, 3), label, fill="black")
    return canvas


def save_qualitative_grid(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    scores: list[np.ndarray],
    sample_ids: list[str],
    path: str | Path,
    max_examples: int,
) -> None:
    """Write a compact MRI | ground truth | anomaly-score grid as PNG."""

    if max_examples <= 0 or not images:
        return
    anomalous = [index for index, mask in enumerate(masks) if mask.any()]
    healthy = [index for index, mask in enumerate(masks) if not mask.any()]
    selected = (anomalous + healthy)[:max_examples]
    rows = []
    for index in selected:
        gray = _gray(images[index])
        mask_overlay = gray.copy().astype(np.float32)
        positive = masks[index].astype(bool)
        mask_overlay[positive] = 0.35 * mask_overlay[positive] + 0.65 * np.asarray(
            [255.0, 32.0, 32.0]
        )
        heat, strength = _heatmap(scores[index])
        score_overlay = (
            gray.astype(np.float32) * (1.0 - 0.70 * strength[..., None])
            + heat.astype(np.float32) * (0.70 * strength[..., None])
        ).astype(np.uint8)
        panels = [
            _labelled(gray, f"MRI: {sample_ids[index]}"),
            _labelled(mask_overlay.astype(np.uint8), "Ground truth"),
            _labelled(score_overlay, "Anomaly score"),
        ]
        row = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
        offset = 0
        for panel in panels:
            row.paste(panel, (offset, 0))
            offset += panel.width
        rows.append(row)
    canvas = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    offset = 0
    for row in rows:
        canvas.paste(row, (0, offset))
        offset += row.height
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def dice_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    prediction = scores >= threshold
    labels = labels.astype(bool)
    denominator = prediction.sum() + labels.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, labels).sum() / denominator)


def best_dice(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if labels.sum() == 0:
        return float("nan"), float("nan")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    dice = 2.0 * precision * recall / np.maximum(precision + recall, 1.0e-12)
    index = int(np.nanargmax(dice))
    threshold = float(thresholds[min(index, len(thresholds) - 1)]) if len(thresholds) else 0.0
    return float(dice[index]), threshold


def threshold_from_healthy(
    score_maps: list[np.ndarray],
    brain_masks: list[np.ndarray],
    percentile: float,
) -> float:
    values = [
        score[brain.astype(bool)] for score, brain in zip(score_maps, brain_masks, strict=True)
    ]
    values = [item for item in values if item.size]
    if not values:
        raise ValueError("No foreground values were available for threshold calibration")
    return float(np.percentile(np.concatenate(values), percentile))


def compute_metrics(
    score_maps: list[np.ndarray],
    lesion_masks: list[np.ndarray],
    brain_masks: list[np.ndarray],
    lesion_sizes: list[str],
    threshold: float,
    image_score_percentile: float,
) -> dict[str, float | dict[str, float]]:
    if not (len(score_maps) == len(lesion_masks) == len(brain_masks) == len(lesion_sizes)):
        raise ValueError("Metric inputs must have equal lengths")

    pixel_labels = []
    pixel_scores = []
    for score, lesion, brain in zip(score_maps, lesion_masks, brain_masks, strict=True):
        roi = brain.astype(bool)
        pixel_labels.append(lesion[roi].astype(np.uint8))
        pixel_scores.append(score[roi].astype(np.float64))
    labels = np.concatenate(pixel_labels)
    scores = np.concatenate(pixel_scores)
    oracle_dice, oracle_threshold = best_dice(labels, scores)

    image_labels = np.asarray([int(mask.any()) for mask in lesion_masks])
    image_scores = np.asarray(
        [
            np.percentile(score[brain.astype(bool)], image_score_percentile)
            if brain.any()
            else float(score.max())
            for score, brain in zip(score_maps, brain_masks, strict=True)
        ]
    )
    healthy = image_labels == 0
    healthy_parts = [
        score[brain.astype(bool)]
        for score, brain, is_healthy in zip(score_maps, brain_masks, healthy, strict=True)
        if is_healthy and brain.any()
    ]
    healthy_pixels = np.concatenate(healthy_parts) if healthy_parts else np.asarray([])

    stratified: dict[str, list[float]] = defaultdict(list)
    for score, lesion, brain, size in zip(
        score_maps, lesion_masks, brain_masks, lesion_sizes, strict=True
    ):
        if not lesion.any():
            continue
        roi = brain.astype(bool)
        stratified[size].append(dice_at_threshold(lesion[roi], score[roi], threshold))

    return {
        "pixel_auroc": _safe_auc(labels, scores),
        "pixel_auprc": _safe_auprc(labels, scores),
        "dice_calibrated": dice_at_threshold(labels, scores, threshold),
        "dice_oracle": oracle_dice,
        "oracle_threshold": oracle_threshold,
        "calibrated_threshold": float(threshold),
        "healthy_pixel_fpr": (
            float((healthy_pixels >= threshold).mean()) if healthy_pixels.size else float("nan")
        ),
        "image_auroc": _safe_auc(image_labels, image_scores),
        "dice_by_lesion_size": {
            key: float(np.mean(values)) for key, values in sorted(stratified.items())
        },
        "n_images": int(len(score_maps)),
        "n_anomalous_images": int(image_labels.sum()),
    }


def compute_per_image_metrics(
    score_maps: list[np.ndarray],
    lesion_masks: list[np.ndarray],
    brain_masks: list[np.ndarray],
    lesion_sizes: list[str],
    sample_ids: list[str],
    threshold: float,
    image_score_percentile: float,
) -> list[dict]:
    """Return paired, sample-addressable metrics for bootstrap comparisons."""

    if not (
        len(score_maps)
        == len(lesion_masks)
        == len(brain_masks)
        == len(lesion_sizes)
        == len(sample_ids)
    ):
        raise ValueError("Per-image metric inputs must have equal lengths")
    rows = []
    for score, lesion, brain, size, sample_id in zip(
        score_maps,
        lesion_masks,
        brain_masks,
        lesion_sizes,
        sample_ids,
        strict=True,
    ):
        roi = brain.astype(bool)
        labels = lesion[roi].astype(np.uint8)
        values = score[roi].astype(np.float64)
        prediction = values >= threshold
        is_anomaly = bool(labels.any())
        true_positive = int(np.logical_and(prediction, labels.astype(bool)).sum())
        recall = float(true_positive / labels.sum()) if is_anomaly else None
        auprc = _safe_auprc(labels, values) if is_anomaly else None
        auroc = _safe_auc(labels, values) if is_anomaly and np.unique(labels).size == 2 else None
        rows.append(
            {
                "sample_id": sample_id,
                "lesion_size": size,
                "is_anomaly": is_anomaly,
                "brain_pixels": int(roi.sum()),
                "lesion_pixels": int(labels.sum()),
                "image_score": (
                    float(np.percentile(values, image_score_percentile))
                    if values.size
                    else float(score.max())
                ),
                "pixel_auprc": float(auprc) if auprc is not None else None,
                "pixel_auroc": float(auroc) if auroc is not None else None,
                "dice_calibrated": (
                    dice_at_threshold(labels, values, threshold) if is_anomaly else None
                ),
                "lesion_recall": recall,
                "healthy_pixel_fpr": float(prediction.mean()) if not is_anomaly else None,
            }
        )
    return rows

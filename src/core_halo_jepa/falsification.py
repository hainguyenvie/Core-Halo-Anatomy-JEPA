from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import validate_config
from .engine import run_evaluation, run_training
from .utils import dump_json

METRIC_PATHS = [
    "pixel_auroc",
    "pixel_auprc",
    "dice_calibrated",
    "healthy_pixel_fpr",
    "image_auroc",
    "dice_by_lesion_size.small",
    "dice_by_lesion_size.medium",
    "dice_by_lesion_size.large",
]


def _nested(payload: dict[str, Any], dotted: str) -> float:
    value: Any = payload
    for key in dotted.split("."):
        value = value[key]
    return float(value)


def _variant(
    base: dict[str, Any],
    name: str,
    context_radius: int,
    halo_size: int,
    anatomy_mode: str,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["name"] = name
    config["model"].update(
        {
            "context_radius": context_radius,
            "halo_size": halo_size,
            "anatomy_mode": anatomy_mode,
            "include_contralateral": anatomy_mode != "none",
        }
    )
    return config


def build_variants(base: dict[str, Any], suite: str) -> list[dict[str, Any]]:
    if suite not in {"all", "context", "halo", "anatomy"}:
        raise ValueError("suite must be all, context, halo, or anatomy")
    variants: dict[str, dict[str, Any]] = {}

    def add(config: dict[str, Any]) -> None:
        variants.setdefault(config["name"], config)

    if suite in {"all", "context"}:
        for radius in (1, 2, 3, 5, 7):
            add(_variant(base, f"ctx_r{radius}", radius, 0, "none"))
    if suite in {"all", "halo"}:
        add(_variant(base, "ctx_r5", 5, 0, "none"))
        add(_variant(base, "halo_h1", 5, 1, "none"))
        add(_variant(base, "halo_h2", 5, 2, "none"))
    if suite in {"all", "anatomy"}:
        add(_variant(base, "halo_h1", 5, 1, "none"))
        add(_variant(base, "anat_mirror", 5, 1, "mirror"))
        add(_variant(base, "anat_random", 5, 1, "random_same_subject"))
        add(_variant(base, "anat_cross_subject", 5, 1, "cross_subject_mirror"))
    return list(variants.values())


def _smoke_config(config: dict[str, Any]) -> None:
    config["data"].update(
        {
            "image_size": 64,
            "train_samples": 8,
            "calibration_samples": 4,
            "test_samples": 12,
            "batch_size": 2,
        }
    )
    config["model"].update(
        {
            "embed_dim": 32,
            "encoder_depth": 1,
            "predictor_depth": 1,
            "num_heads": 4,
        }
    )
    config["train"].update({"epochs": 1, "warmup_epochs": 0, "targets_per_image": 1})
    config["score"].update({"geometry_stride": 2, "geometry_batch_size": 32, "save_examples": 3})


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["experiment"])].append(run)
    aggregate: dict[str, dict[str, float]] = {}
    for name, items in grouped.items():
        aggregate[name] = {}
        for path in METRIC_PATHS:
            try:
                values = np.asarray([_nested(item, path) for item in items], dtype=float)
            except KeyError:
                continue
            aggregate[name][f"{path}.mean"] = float(np.nanmean(values))
            aggregate[name][f"{path}.std"] = float(np.nanstd(values))
    return aggregate


def _bootstrap_mean_ci(
    differences: np.ndarray,
    samples: int,
    seed: int = 2026,
) -> tuple[float, float, float]:
    if differences.size == 0:
        return float("nan"), float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, size=(samples, differences.size))
    distribution = differences[indices].mean(axis=1)
    return (
        float(differences.mean()),
        float(np.percentile(distribution, 2.5)),
        float(np.percentile(distribution, 97.5)),
    )


def _comparison(
    left: str,
    right: str,
    runs: list[dict[str, Any]],
    per_image: dict[tuple[str, int], list[dict[str, Any]]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    left_runs = {int(run["seed"]): run for run in runs if run["experiment"] == left}
    right_runs = {int(run["seed"]): run for run in runs if run["experiment"] == right}
    seeds = sorted(set(left_runs) & set(right_runs))
    seed_differences = np.asarray(
        [left_runs[seed]["pixel_auprc"] - right_runs[seed]["pixel_auprc"] for seed in seeds]
    )

    paired: dict[str, dict[str, float | int]] = {}
    for metric, lesion_size in (
        ("pixel_auprc", None),
        ("dice_calibrated", None),
        ("dice_calibrated", "small"),
    ):
        differences = []
        for seed in seeds:
            left_rows = {row["sample_id"]: row for row in per_image[(left, seed)]}
            right_rows = {row["sample_id"]: row for row in per_image[(right, seed)]}
            for sample_id in sorted(set(left_rows) & set(right_rows)):
                left_row = left_rows[sample_id]
                right_row = right_rows[sample_id]
                if lesion_size and left_row["lesion_size"] != lesion_size:
                    continue
                left_value = left_row.get(metric)
                right_value = right_row.get(metric)
                if left_value is None or right_value is None:
                    continue
                differences.append(float(left_value) - float(right_value))
        values = np.asarray(differences, dtype=float)
        mean, low, high = _bootstrap_mean_ci(values, bootstrap_samples)
        key = f"{lesion_size}_{metric}" if lesion_size else metric
        paired[key] = {
            "mean_difference": mean,
            "ci95_low": low,
            "ci95_high": high,
            "n": len(values),
        }

    required_wins = math.ceil(len(seeds) * 2 / 3)
    seed_mean = float(seed_differences.mean()) if seed_differences.size else float("nan")
    seed_wins = int((seed_differences > 0).sum())
    primary = paired["pixel_auprc"]
    if len(seeds) < 3:
        verdict = "insufficient_seeds"
    elif seed_mean >= 0.001 and seed_wins >= required_wins and float(primary["ci95_low"]) > 0:
        verdict = "supported"
    elif (
        seed_mean <= -0.001
        and seed_wins <= len(seeds) - required_wins
        and float(primary["ci95_high"]) < 0
    ):
        verdict = "contradicted"
    elif abs(seed_mean) < 0.001:
        verdict = "negligible"
    else:
        verdict = "inconclusive"
    return {
        "left": left,
        "right": right,
        "global_pixel_auprc_mean_difference": seed_mean,
        "seed_differences": seed_differences.tolist(),
        "seed_wins": seed_wins,
        "n_seeds": len(seeds),
        "paired": paired,
        "verdict": verdict,
    }


def _joint_verdict(items: list[dict[str, Any]]) -> str:
    if not items:
        return "not_run"
    verdicts = {str(item["verdict"]) for item in items}
    if "insufficient_seeds" in verdicts:
        return "insufficient_seeds"
    if verdicts == {"supported"}:
        return "supported"
    if "contradicted" in verdicts:
        return "contradicted"
    if verdicts == {"negligible"}:
        return "negligible"
    return "inconclusive"


def _component_verdicts(comparisons: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {(item["left"], item["right"]): item for item in comparisons}
    components: dict[str, dict[str, Any]] = {}
    for key, pair in (
        ("wide_context_r5_vs_r2", ("ctx_r5", "ctx_r2")),
        ("halo_h1_vs_r5", ("halo_h1", "ctx_r5")),
    ):
        if pair in indexed:
            components[key] = {
                "verdict": indexed[pair]["verdict"],
                "required_comparisons": [f"{pair[0]}>{pair[1]}"],
            }
    anatomy_pairs = [
        ("anat_mirror", "halo_h1"),
        ("anat_mirror", "anat_random"),
        ("anat_mirror", "anat_cross_subject"),
    ]
    anatomy_items = [indexed[pair] for pair in anatomy_pairs if pair in indexed]
    if len(anatomy_items) == len(anatomy_pairs):
        components["mirror_anatomy_joint"] = {
            "verdict": _joint_verdict(anatomy_items),
            "required_comparisons": [f"{left}>{right}" for left, right in anatomy_pairs],
        }
    return components


def _comparison_pairs(names: set[str]) -> list[tuple[str, str]]:
    candidates = [
        ("ctx_r1", "ctx_r2"),
        ("ctx_r3", "ctx_r2"),
        ("ctx_r5", "ctx_r2"),
        ("ctx_r7", "ctx_r2"),
        ("halo_h1", "ctx_r5"),
        ("halo_h2", "ctx_r5"),
        ("anat_mirror", "halo_h1"),
        ("anat_mirror", "anat_random"),
        ("anat_mirror", "anat_cross_subject"),
    ]
    return [(left, right) for left, right in candidates if left in names and right in names]


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# JEPA falsification results",
        "",
        "## Variant ranking",
        "",
        "| Variant | Pixel AUPRC | Calibrated Dice | Small-lesion Dice | Healthy FPR |",
        "|---|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        summary["aggregate"].items(),
        key=lambda item: item[1].get("pixel_auprc.mean", float("-inf")),
        reverse=True,
    )
    for name, values in ranked:
        lines.append(
            f"| {name} | {values.get('pixel_auprc.mean', float('nan')):.4f} | "
            f"{values.get('dice_calibrated.mean', float('nan')):.4f} | "
            f"{values.get('dice_by_lesion_size.small.mean', float('nan')):.4f} | "
            f"{values.get('healthy_pixel_fpr.mean', float('nan')):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired tests",
            "",
            "| Comparison | Global ΔAUPRC | Seed wins | Per-image ΔAUPRC [95% CI] | Verdict |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in summary["comparisons"]:
        primary = item["paired"]["pixel_auprc"]
        lines.append(
            f"| {item['left']} − {item['right']} | "
            f"{item['global_pixel_auprc_mean_difference']:+.5f} | "
            f"{item['seed_wins']}/{item['n_seeds']} | "
            f"{primary['mean_difference']:+.5f} "
            f"[{primary['ci95_low']:+.5f}, {primary['ci95_high']:+.5f}] | "
            f"**{item['verdict']}** |"
        )
    if summary["component_verdicts"]:
        lines.extend(
            [
                "",
                "## Component decisions",
                "",
                "| Component | Required comparisons | Verdict |",
                "|---|---|---|",
            ]
        )
        for name, item in summary["component_verdicts"].items():
            required = ", ".join(item["required_comparisons"])
            lines.append(f"| {name} | {required} | **{item['verdict']}** |")
    lines.extend(
        [
            "",
            "`supported` requires a non-trivial global gain (≥0.001), wins in at least two-thirds of seeds, and a positive paired bootstrap lower bound.",
            "Runs with fewer than three seeds are labeled `insufficient_seeds` and are plumbing checks only.",
        ]
    )
    return "\n".join(lines) + "\n"


def _assert_resume_compatible(config: dict[str, Any], run_dir: Path) -> None:
    resolved_path = run_dir / "resolved_config.yaml"
    if not resolved_path.exists():
        raise RuntimeError(
            f"Cannot safely reuse artifacts in {run_dir}: resolved_config.yaml is missing. "
            "Use --rerun or choose a new --output-root."
        )
    existing = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    expected = {key: value for key, value in config.items() if not key.startswith("_")}
    if existing != expected:
        raise RuntimeError(
            f"Refusing to mix incompatible artifacts in {run_dir}. The requested configuration "
            "does not match resolved_config.yaml; use --rerun or choose a new --output-root."
        )


def run_falsification(
    base_config: dict[str, Any],
    seeds: list[int],
    output_root: str | Path,
    suite: str,
    smoke: bool,
    skip_existing: bool,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    variants = build_variants(base_config, suite)
    runs: list[dict[str, Any]] = []
    per_image: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for variant in variants:
        for seed in seeds:
            config = copy.deepcopy(variant)
            config["seed"] = seed
            config["model"]["anatomy_seed"] = seed
            config["output_dir"] = str(output_root / config["name"] / f"seed_{seed}")
            if smoke:
                _smoke_config(config)
            validate_config(config)
            run_dir = Path(config["output_dir"])
            metrics_path = run_dir / "metrics.json"
            per_image_path = run_dir / "per_image_metrics.json"
            checkpoint = run_dir / "best.pt"
            existing_artifacts = (
                checkpoint.exists() or metrics_path.exists() or per_image_path.exists()
            )
            if skip_existing and existing_artifacts:
                _assert_resume_compatible(config, run_dir)
            if skip_existing and metrics_path.exists() and per_image_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                if not (skip_existing and checkpoint.exists()):
                    checkpoint = run_training(config)
                metrics = run_evaluation(config, checkpoint)
            rows = json.loads(per_image_path.read_text(encoding="utf-8"))
            runs.append(metrics)
            per_image[(config["name"], seed)] = rows

    names = {variant["name"] for variant in variants}
    comparisons = [
        _comparison(left, right, runs, per_image, bootstrap_samples)
        for left, right in _comparison_pairs(names)
    ]
    component_verdicts = _component_verdicts(comparisons)
    summary = {
        "suite": suite,
        "smoke": smoke,
        "seeds": seeds,
        "bootstrap_samples": bootstrap_samples,
        "variants": [variant["name"] for variant in variants],
        "runs": runs,
        "aggregate": _aggregate(runs),
        "comparisons": comparisons,
        "component_verdicts": component_verdicts,
    }
    dump_json(summary, output_root / "falsification_summary.json")
    (output_root / "falsification_summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary

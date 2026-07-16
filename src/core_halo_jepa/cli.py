from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .engine import run_evaluation, run_training
from .utils import dump_json


def _metric(metrics: dict[str, Any], dotted: str) -> float:
    value: Any = metrics
    for key in dotted.split("."):
        value = value[key]
    return float(value)


def _markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Core-Halo Anatomy JEPA PoC results",
        "",
        "| Experiment | Pixel AUROC | Pixel AUPRC | Calibrated Dice | Small-lesion Dice | Healthy FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary["aggregate"].items():
        small = values.get("dice_by_lesion_size.small.mean", float("nan"))
        lines.append(
            f"| {name} | {values['pixel_auroc.mean']:.4f} | "
            f"{values['pixel_auprc.mean']:.4f} | {values['dice_calibrated.mean']:.4f} | "
            f"{small:.4f} | {values['healthy_pixel_fpr.mean']:.4f} |"
        )
    lines.extend(["", "## Pre-registered directional checks", ""])
    for check, passed in summary["hypothesis_checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] {check}")
    lines.extend(
        [
            "",
            "A failed check is informative: it falsifies that component under this setup and should not be hidden.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_poc(
    config_paths: list[str],
    seeds: list[int],
    output_root: str | Path,
    smoke: bool,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for config_path in config_paths:
        base = load_config(config_path)
        for seed in seeds:
            config = copy.deepcopy(base)
            config["seed"] = seed
            config["output_dir"] = str(output_root / config["name"] / f"seed_{seed}")
            if smoke:
                config["data"].update(
                    {
                        "image_size": 64,
                        "train_samples": 8,
                        "calibration_samples": 4,
                        "test_samples": 6,
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
                config["score"].update({"geometry_stride": 2, "geometry_batch_size": 32})
            checkpoint = run_training(config)
            metrics = run_evaluation(config, checkpoint)
            results.append(metrics)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["experiment"])].append(result)
    metric_paths = [
        "pixel_auroc",
        "pixel_auprc",
        "dice_calibrated",
        "healthy_pixel_fpr",
        "image_auroc",
        "dice_by_lesion_size.small",
        "dice_by_lesion_size.medium",
        "dice_by_lesion_size.large",
    ]
    aggregate: dict[str, dict[str, float]] = {}
    for name, runs in grouped.items():
        aggregate[name] = {}
        for path in metric_paths:
            try:
                values = np.asarray([_metric(run, path) for run in runs], dtype=float)
            except KeyError:
                continue
            aggregate[name][f"{path}.mean"] = float(np.nanmean(values))
            aggregate[name][f"{path}.std"] = float(np.nanstd(values))

    def greater(left: str, right: str, metric: str) -> bool:
        if left not in aggregate or right not in aggregate:
            return False
        return aggregate[left].get(f"{metric}.mean", -np.inf) > aggregate[right].get(
            f"{metric}.mean", np.inf
        )

    checks = {
        "Wider target-free context improves pixel AUPRC over local context": greater(
            "wide_context", "local_context", "pixel_auprc"
        ),
        "Adding a halo improves pixel AUPRC over the same wide context": greater(
            "core_halo", "wide_context", "pixel_auprc"
        ),
        "Contralateral anatomy improves pixel AUPRC over Core-Halo alone": greater(
            "core_halo_anatomy", "core_halo", "pixel_auprc"
        ),
        "Full model improves small-lesion Dice over local context": greater(
            "core_halo_anatomy", "local_context", "dice_by_lesion_size.small"
        ),
    }
    summary = {
        "smoke": smoke,
        "seeds": seeds,
        "runs": results,
        "aggregate": aggregate,
        "hypothesis_checks": checks,
    }
    dump_json(summary, output_root / "poc_summary.json")
    (output_root / "poc_summary.md").write_text(_markdown_summary(summary), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="core-halo-jepa",
        description="Train and falsify Core-Halo Anatomy JEPA experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train one JEPA configuration")
    train.add_argument("--config", required=True)
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    train.add_argument("--resume")

    evaluate = subparsers.add_parser("evaluate", help="Calibrate and evaluate a checkpoint")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    poc = subparsers.add_parser("run-poc", help="Run the pre-registered ablation matrix")
    poc.add_argument("--configs", nargs="+", required=True)
    poc.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    poc.add_argument("--output-root", default="outputs/poc")
    poc.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        config = load_config(args.config, args.set)
        checkpoint = run_training(config, args.resume)
        print(checkpoint)
    elif args.command == "evaluate":
        config = load_config(args.config, args.set)
        metrics = run_evaluation(config, args.checkpoint)
        print(json.dumps(metrics, indent=2))
    else:
        summary = run_poc(args.configs, args.seeds, args.output_root, args.smoke)
        print(json.dumps(summary["hypothesis_checks"], indent=2))


if __name__ == "__main__":
    main()

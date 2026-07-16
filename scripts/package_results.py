#!/usr/bin/env python3
"""Copy reviewable experiment outputs without checkpoints or raw arrays."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT_FILES = {"falsification_summary.json", "falsification_summary.md"}
RUN_FILES = {
    "metrics.json",
    "per_image_metrics.json",
    "qualitative_grid.png",
    "resolved_config.yaml",
    "train_history.json",
    "training_summary.json",
}


def package_results(input_root: Path, output_root: Path) -> list[Path]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError("Output must not be the input directory or a child of it")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Choose a new review directory."
        )

    sources = [path for name in ROOT_FILES if (path := input_root / name).is_file()]
    sources.extend(
        path for path in input_root.rglob("*") if path.is_file() and path.name in RUN_FILES
    )
    if not any(path.name == "falsification_summary.json" for path in sources):
        raise FileNotFoundError(f"No falsification_summary.json found in {input_root}")

    copied = []
    for source in sorted(set(sources)):
        destination = output_root / source.relative_to(input_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package small, reviewable falsification outputs for Git."
    )
    parser.add_argument("--input", default="outputs/falsification", type=Path)
    parser.add_argument("--output", default="results/falsification", type=Path)
    args = parser.parse_args()
    copied = package_results(args.input, args.output)
    total_bytes = sum(path.stat().st_size for path in copied)
    print(f"Copied {len(copied)} files ({total_bytes / 1024**2:.2f} MiB) to {args.output}")


if __name__ == "__main__":
    main()

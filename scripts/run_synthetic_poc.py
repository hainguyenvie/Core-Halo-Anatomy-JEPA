#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from core_halo_jepa.cli import run_poc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four-way synthetic hypothesis test.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-root", default="outputs/poc")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configs = [
        root / "configs/synthetic/local_context.yaml",
        root / "configs/synthetic/wide_context.yaml",
        root / "configs/synthetic/core_halo.yaml",
        root / "configs/synthetic/core_halo_anatomy.yaml",
    ]
    run_poc([str(path) for path in configs], args.seeds, args.output_root, args.smoke)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Copy reviewable experiment outputs without checkpoints or raw arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

from core_halo_jepa.result_packaging import package_results


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

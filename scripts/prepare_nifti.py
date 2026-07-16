#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from core_halo_jepa.data import _resize_pair


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def lesion_size(mask: np.ndarray, brain: np.ndarray) -> str:
    if not mask.any():
        return "healthy"
    fraction = float(mask.sum() / max(brain.sum(), 1))
    if fraction <= 0.005:
        return "small"
    if fraction <= 0.02:
        return "medium"
    return "large"


def canonical_array(path: Path, channel: int) -> np.ndarray:
    image = nib.as_closest_canonical(nib.load(path))
    array = np.asarray(image.dataobj)
    if array.ndim == 4:
        array = array[..., channel]
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D volume at {path}, found shape {array.shape}")
    return np.asarray(array)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert patient-level NIfTI volumes to leakage-safe 2-D slice manifests."
    )
    parser.add_argument("--volumes-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--image-size", type=int, default=240)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--min-brain-fraction", type=float, default=0.05)
    args = parser.parse_args()

    source_csv = Path(args.volumes_csv).resolve()
    source_root = source_csv.parent
    output_dir = Path(args.output_dir).resolve()
    slices_dir = output_dir / "arrays"
    slices_dir.mkdir(parents=True, exist_ok=True)
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image", "split", "patient_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Input CSV must contain columns: {sorted(required)}")

    output_rows = []
    seen_patients: dict[str, str] = {}
    for row in rows:
        patient_id = row["patient_id"]
        split = row["split"].lower()
        previous = seen_patients.setdefault(patient_id, split)
        if previous != split:
            raise ValueError(f"Patient {patient_id!r} occurs in both {previous!r} and {split!r}")
        image_path = resolve(source_root, row["image"])
        image = canonical_array(image_path, args.channel)
        if row.get("mask"):
            mask_path = resolve(source_root, row["mask"])
            mask = canonical_array(mask_path, 0) > 0
            if mask.shape != image.shape:
                raise ValueError(
                    f"Image/mask grid mismatch for {patient_id}: {image.shape} vs {mask.shape}. "
                    "Register/resample the mask before this script."
                )
        else:
            mask = np.zeros_like(image, dtype=bool)

        for slice_index in range(image.shape[args.axis]):
            image_slice = np.take(image, slice_index, axis=args.axis)
            mask_slice = np.take(mask, slice_index, axis=args.axis)
            brain = np.isfinite(image_slice) & (image_slice != 0)
            if brain.mean() < args.min_brain_fraction and not mask_slice.any():
                continue
            image_slice, mask_slice = _resize_pair(
                image_slice.astype(np.float32),
                mask_slice.astype(np.uint8),
                (args.image_size, args.image_size),
            )
            brain_slice = image_slice != 0
            sample_id = f"{patient_id}_axis{args.axis}_{slice_index:04d}"
            destination = slices_dir / f"{sample_id}.npz"
            np.savez_compressed(
                destination,
                image=image_slice.astype(np.float32),
                mask=mask_slice.astype(np.uint8),
            )
            output_rows.append(
                {
                    "image": destination.relative_to(output_dir).as_posix(),
                    "mask": "",
                    "split": split,
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "lesion_size": lesion_size(mask_slice, brain_slice),
                }
            )

    manifest = output_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "mask", "split", "sample_id", "patient_id", "lesion_size"],
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} slices to {manifest}")


if __name__ == "__main__":
    main()

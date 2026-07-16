import json
from pathlib import Path

import pytest

from scripts.package_results import package_results


def test_package_results_excludes_large_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "outputs"
    run = source / "anat_mirror" / "seed_0"
    run.mkdir(parents=True)
    (source / "falsification_summary.json").write_text("{}", encoding="utf-8")
    (source / "falsification_summary.md").write_text("# Results", encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps({"pixel_auprc": 0.5}), encoding="utf-8")
    (run / "best.pt").write_bytes(b"large checkpoint")
    (run / "examples.npz").write_bytes(b"raw arrays")

    destination = tmp_path / "review"
    copied = package_results(source, destination)

    assert len(copied) == 3
    assert (destination / "anat_mirror" / "seed_0" / "metrics.json").exists()
    assert not (destination / "anat_mirror" / "seed_0" / "best.pt").exists()
    assert not (destination / "anat_mirror" / "seed_0" / "examples.npz").exists()


def test_package_results_refuses_nonempty_destination(tmp_path: Path) -> None:
    source = tmp_path / "outputs"
    source.mkdir()
    (source / "falsification_summary.json").write_text("{}", encoding="utf-8")
    destination = tmp_path / "review"
    destination.mkdir()
    (destination / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError):
        package_results(source, destination)

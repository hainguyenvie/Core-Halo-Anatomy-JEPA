import copy
from pathlib import Path

import pytest

from core_halo_jepa.config import DEFAULT_CONFIG, save_resolved_config
from core_halo_jepa.falsification import (
    _assert_resume_compatible,
    _comparison,
    build_variants,
)


def test_all_suite_has_unique_causal_controls() -> None:
    variants = build_variants(copy.deepcopy(DEFAULT_CONFIG), "all")
    names = [variant["name"] for variant in variants]
    assert len(names) == len(set(names)) == 10
    assert {"ctx_r1", "ctx_r2", "ctx_r3", "ctx_r5", "ctx_r7"}.issubset(names)
    assert {"halo_h1", "halo_h2"}.issubset(names)
    assert {"anat_mirror", "anat_random", "anat_cross_subject"}.issubset(names)


def test_anatomy_variants_change_only_anatomy_mode() -> None:
    variants = {
        item["name"]: item for item in build_variants(copy.deepcopy(DEFAULT_CONFIG), "anatomy")
    }
    for name in ("anat_mirror", "anat_random", "anat_cross_subject"):
        assert variants[name]["model"]["context_radius"] == 5
        assert variants[name]["model"]["halo_size"] == 1
        assert variants[name]["model"]["include_contralateral"]
    assert variants["anat_mirror"]["model"]["anatomy_mode"] == "mirror"
    assert variants["anat_random"]["model"]["anatomy_mode"] == "random_same_subject"
    assert variants["anat_cross_subject"]["model"]["anatomy_mode"] == "cross_subject_mirror"


def test_one_seed_comparison_is_never_evidence() -> None:
    runs = [
        {"experiment": "left", "seed": 0, "pixel_auprc": 0.8},
        {"experiment": "right", "seed": 0, "pixel_auprc": 0.2},
    ]
    per_image = {
        ("left", 0): [{"sample_id": "case", "lesion_size": "small", "pixel_auprc": 0.8}],
        ("right", 0): [{"sample_id": "case", "lesion_size": "small", "pixel_auprc": 0.2}],
    }
    result = _comparison("left", "right", runs, per_image, bootstrap_samples=20)
    assert result["verdict"] == "insufficient_seeds"


def test_resume_rejects_changed_configuration(tmp_path: Path) -> None:
    config = copy.deepcopy(DEFAULT_CONFIG)
    save_resolved_config(config, tmp_path / "resolved_config.yaml")
    _assert_resume_compatible(config, tmp_path)

    changed = copy.deepcopy(config)
    changed["train"]["epochs"] += 1
    with pytest.raises(RuntimeError, match="incompatible artifacts"):
        _assert_resume_compatible(changed, tmp_path)

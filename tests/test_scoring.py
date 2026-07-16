import numpy as np
import torch

from core_halo_jepa.metrics import compute_per_image_metrics
from core_halo_jepa.scoring import ResidualCalibrator


def test_coordinate_calibrator_prefers_healthy_residuals() -> None:
    calibrator = ResidualCalibrator.empty(token_count=3, dimension=2, variance_floor=1.0e-3)
    healthy = torch.tensor([[[0.0, 0.1]], [[0.1, 0.0]], [[-0.1, 0.0]]])
    indices = torch.tensor([[1], [1], [1]])
    calibrator.update(healthy, indices)
    calibrator.finalize(shrinkage=0.1)
    inlier = calibrator.score(torch.tensor([[[0.0, 0.0]]]), torch.tensor([[1]])).item()
    outlier = calibrator.score(torch.tensor([[[2.0, 2.0]]]), torch.tensor([[1]])).item()
    assert outlier > inlier


def test_per_image_metrics_preserve_sample_identity() -> None:
    scores = [np.asarray([[0.0, 0.9], [0.1, 0.8]], dtype=float)]
    lesions = [np.asarray([[0, 1], [0, 1]], dtype=np.uint8)]
    brains = [np.ones((2, 2), dtype=np.uint8)]
    rows = compute_per_image_metrics(
        scores,
        lesions,
        brains,
        ["small"],
        ["case-1"],
        threshold=0.5,
        image_score_percentile=99.0,
    )
    assert rows[0]["sample_id"] == "case-1"
    assert rows[0]["dice_calibrated"] == 1.0
    assert rows[0]["pixel_auprc"] == 1.0

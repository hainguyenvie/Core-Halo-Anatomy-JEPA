import torch

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

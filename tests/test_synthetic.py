import torch

from core_halo_jepa.data import SyntheticBrainDataset


def test_synthetic_data_are_deterministic_and_train_is_healthy() -> None:
    dataset = SyntheticBrainDataset(4, 64, seed=7, anomaly_fraction=0.0, split="train")
    first = dataset[1]
    repeated = dataset[1]
    assert torch.equal(first["image"], repeated["image"])
    assert first["mask"].sum() == 0


def test_synthetic_test_contains_lesion_mask() -> None:
    dataset = SyntheticBrainDataset(4, 64, seed=7, anomaly_fraction=1.0, split="test")
    assert all(dataset[index]["mask"].sum() > 0 for index in range(len(dataset)))

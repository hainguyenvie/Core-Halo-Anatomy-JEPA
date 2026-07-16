import torch

from core_halo_jepa.geometry import make_geometry
from core_halo_jepa.model import CoreHaloJEPA


def tiny_config() -> dict:
    return {
        "in_channels": 1,
        "patch_size": 8,
        "embed_dim": 32,
        "encoder_depth": 1,
        "predictor_depth": 1,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
    }


def test_model_shapes_and_frozen_target() -> None:
    model = CoreHaloJEPA(tiny_config())
    images = torch.randn(2, 1, 64, 64)
    geometries = [
        make_geometry(2, 1, (8, 8), 2, 1, 3, True, 2),
        make_geometry(4, 4, (8, 8), 2, 1, 3, True, 2),
        make_geometry(1, 5, (8, 8), 2, 1, 3, True, 2),
    ]
    output = model(images, geometries, torch.tensor([0, 1, 1]))
    assert output.prediction.shape == output.target.shape == (3, 4, 32)
    assert not output.target.requires_grad
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())


def test_ema_update_moves_target_toward_context() -> None:
    model = CoreHaloJEPA(tiny_config())
    with torch.no_grad():
        context_parameter = next(model.context_encoder.parameters())
        target_parameter = next(model.target_encoder.parameters())
        context_parameter.add_(1.0)
        distance_before = (target_parameter - context_parameter).abs().mean()
        model.update_target_encoder(0.5)
        distance_after = (target_parameter - context_parameter).abs().mean()
    assert distance_after < distance_before

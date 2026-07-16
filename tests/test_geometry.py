import torch

from core_halo_jepa.geometry import (
    CONTRALATERAL_CONTEXT,
    enumerate_geometries,
    make_geometry,
    pack_geometries,
)


def test_core_and_halo_never_enter_context() -> None:
    geometry = make_geometry(
        top=5,
        left=3,
        grid_size=(16, 16),
        core_size=2,
        halo_size=1,
        context_radius=5,
        include_contralateral=True,
        contralateral_size=2,
    )
    target = set(geometry.target_indices.tolist())
    context = set(geometry.context_indices.tolist())
    assert target.isdisjoint(context)
    for row in range(4, 8):
        for col in range(2, 6):
            assert row * 16 + col not in context
    assert CONTRALATERAL_CONTEXT in geometry.context_types.tolist()


def test_midline_contralateral_context_cannot_reintroduce_target() -> None:
    geometry = make_geometry(4, 3, (10, 10), 4, 1, 3, True, 4)
    assert set(geometry.target_indices.tolist()).isdisjoint(geometry.context_indices.tolist())


def test_pack_geometries_marks_padding() -> None:
    edge = make_geometry(0, 0, (8, 8), 2, 1, 3, False, 2)
    center = make_geometry(3, 3, (8, 8), 2, 1, 3, False, 2)
    target, context, types, padding = pack_geometries([edge, center], "cpu")
    assert target.shape == (2, 4)
    assert context.shape == types.shape == padding.shape
    assert padding[0].sum() > 0
    assert padding[1].sum() == 0


def test_foreground_filter_keeps_only_relevant_cores() -> None:
    foreground = torch.zeros(8, 8, dtype=torch.bool)
    foreground[4, 4] = True
    geometries = enumerate_geometries((8, 8), 2, 1, 3, False, 2, foreground=foreground)
    assert geometries
    assert all(
        geometry.top <= 4 < geometry.top + 2 and geometry.left <= 4 < geometry.left + 2
        for geometry in geometries
    )


def test_anatomy_controls_are_token_count_matched_and_leak_free() -> None:
    kwargs = dict(
        top=5,
        left=2,
        grid_size=(16, 16),
        core_size=2,
        halo_size=1,
        context_radius=5,
        include_contralateral=True,
        contralateral_size=2,
    )
    mirror = make_geometry(**kwargs, anatomy_mode="mirror")
    random = make_geometry(**kwargs, anatomy_mode="random_same_subject", anatomy_seed=11)
    cross = make_geometry(**kwargs, anatomy_mode="cross_subject_mirror")
    assert mirror.anatomy_token_count == random.anatomy_token_count == cross.anatomy_token_count
    assert mirror.anatomy_token_count > 0
    mirror_anatomy = mirror.context_indices[mirror.context_types == CONTRALATERAL_CONTEXT]
    random_anatomy = random.context_indices[random.context_types == CONTRALATERAL_CONTEXT]
    assert not torch.equal(mirror_anatomy, random_anatomy)
    for geometry in (mirror, random, cross):
        assert set(geometry.target_indices.tolist()).isdisjoint(geometry.context_indices.tolist())


def test_anatomy_tokens_remain_explicit_when_wide_context_already_sees_coordinate() -> None:
    geometry = make_geometry(4, 1, (12, 12), 2, 1, 7, True, 2, anatomy_mode="mirror")
    local = geometry.context_indices[geometry.context_types == 0]
    anatomy = geometry.context_indices[geometry.context_types == CONTRALATERAL_CONTEXT]
    assert set(local.tolist()) & set(anatomy.tolist())

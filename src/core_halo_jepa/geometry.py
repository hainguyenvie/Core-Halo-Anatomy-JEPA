from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

LOCAL_CONTEXT = 0
CONTRALATERAL_CONTEXT = 1
ANATOMY_MODES = {"none", "mirror", "random_same_subject", "cross_subject_mirror"}


@dataclass(frozen=True)
class Geometry:
    """Token indices defining one target core and its target-free context."""

    target_indices: torch.Tensor
    context_indices: torch.Tensor
    context_types: torch.Tensor
    top: int
    left: int

    @property
    def anatomy_token_count(self) -> int:
        return int((self.context_types == CONTRALATERAL_CONTEXT).sum())

    def to(self, device: torch.device | str) -> Geometry:
        return Geometry(
            target_indices=self.target_indices.to(device),
            context_indices=self.context_indices.to(device),
            context_types=self.context_types.to(device),
            top=self.top,
            left=self.left,
        )


def _flat(row: int, col: int, grid_width: int) -> int:
    return row * grid_width + col


def _inside_box(
    row: int,
    col: int,
    top: int,
    left: int,
    height: int,
    width: int,
) -> bool:
    return top <= row < top + height and left <= col < left + width


def resolve_anatomy_mode(include_contralateral: bool, anatomy_mode: str | None) -> str:
    if anatomy_mode is None:
        return "mirror" if include_contralateral else "none"
    if anatomy_mode not in ANATOMY_MODES:
        raise ValueError(
            f"Unknown anatomy_mode={anatomy_mode!r}; expected one of {sorted(ANATOMY_MODES)}"
        )
    return anatomy_mode


def _mirror_block(
    top: int,
    left: int,
    grid_size: tuple[int, int],
    core_size: int,
    contralateral_size: int,
    excluded_box: tuple[int, int, int],
) -> list[int]:
    grid_height, grid_width = grid_size
    excluded_top, excluded_left, excluded_size = excluded_box
    target_center_row = top + (core_size - 1) / 2
    target_center_col = left + (core_size - 1) / 2
    mirror_center_col = (grid_width - 1) - target_center_col
    contra_top = round(target_center_row - (contralateral_size - 1) / 2)
    contra_left = round(mirror_center_col - (contralateral_size - 1) / 2)
    indices = []
    for row in range(contra_top, contra_top + contralateral_size):
        for col in range(contra_left, contra_left + contralateral_size):
            if not (0 <= row < grid_height and 0 <= col < grid_width):
                continue
            if _inside_box(
                row,
                col,
                excluded_top,
                excluded_left,
                excluded_size,
                excluded_size,
            ):
                continue
            indices.append(_flat(row, col, grid_width))
    return indices


def _matched_random_indices(
    count: int,
    top: int,
    left: int,
    grid_size: tuple[int, int],
    excluded_box: tuple[int, int, int],
    forbidden: set[int],
    anatomy_seed: int,
) -> list[int]:
    """Choose a deterministic token-count-matched non-homologous control."""

    if count == 0:
        return []
    grid_height, grid_width = grid_size
    excluded_top, excluded_left, excluded_size = excluded_box
    candidates = []
    for row in range(grid_height):
        for col in range(grid_width):
            index = _flat(row, col, grid_width)
            if index in forbidden:
                continue
            if _inside_box(
                row,
                col,
                excluded_top,
                excluded_left,
                excluded_size,
                excluded_size,
            ):
                continue
            candidates.append(index)
    if len(candidates) < count:
        raise ValueError("Not enough non-target tokens for the matched random anatomy control")
    mixed_seed = (
        int(anatomy_seed) * 1_000_003 + (top + 1) * 97_409 + (left + 1) * 65_537
    ) & 0x7FFF_FFFF
    generator = torch.Generator().manual_seed(mixed_seed)
    order = torch.randperm(len(candidates), generator=generator)[:count].tolist()
    return [candidates[index] for index in order]


def make_geometry(
    top: int,
    left: int,
    grid_size: tuple[int, int],
    core_size: int,
    halo_size: int,
    context_radius: int,
    include_contralateral: bool,
    contralateral_size: int,
    anatomy_mode: str | None = None,
    anatomy_seed: int = 0,
) -> Geometry:
    """Build a Core-Halo geometry on a 2-D patch grid.

    The target core and its halo are never passed to the context encoder. The
    contralateral block is mirrored across the left-right (column) axis, which
    assumes consistently oriented/registered axial slices.
    """

    grid_height, grid_width = grid_size
    if not (0 <= top <= grid_height - core_size):
        raise ValueError(f"Invalid target top={top} for grid height {grid_height}")
    if not (0 <= left <= grid_width - core_size):
        raise ValueError(f"Invalid target left={left} for grid width {grid_width}")

    target = [
        _flat(row, col, grid_width)
        for row in range(top, top + core_size)
        for col in range(left, left + core_size)
    ]

    excluded_top = top - halo_size
    excluded_left = left - halo_size
    excluded_size = core_size + 2 * halo_size
    local_top = max(0, top - context_radius)
    local_left = max(0, left - context_radius)
    local_bottom = min(grid_height, top + core_size + context_radius)
    local_right = min(grid_width, left + core_size + context_radius)

    context_indices: list[int] = []
    context_types: list[int] = []
    for row in range(local_top, local_bottom):
        for col in range(local_left, local_right):
            if _inside_box(
                row,
                col,
                excluded_top,
                excluded_left,
                excluded_size,
                excluded_size,
            ):
                continue
            context_indices.append(_flat(row, col, grid_width))
            context_types.append(LOCAL_CONTEXT)

    mode = resolve_anatomy_mode(include_contralateral, anatomy_mode)
    if mode != "none":
        mirror = _mirror_block(
            top,
            left,
            grid_size,
            core_size,
            contralateral_size,
            (excluded_top, excluded_left, excluded_size),
        )
        if mode == "random_same_subject":
            anatomy_indices = _matched_random_indices(
                count=len(mirror),
                top=top,
                left=left,
                grid_size=grid_size,
                excluded_box=(excluded_top, excluded_left, excluded_size),
                forbidden=set(mirror),
                anatomy_seed=anatomy_seed,
            )
        else:
            anatomy_indices = mirror
        # Keep anatomy tokens explicit even when their coordinates are already
        # in the wide local window. This makes mirror/random/cross-subject
        # controls token-count matched instead of silently deduplicating them.
        context_indices.extend(anatomy_indices)
        context_types.extend([CONTRALATERAL_CONTEXT] * len(anatomy_indices))

    if not context_indices:
        raise ValueError("Geometry has no visible context; enlarge context_radius")
    target_tensor = torch.tensor(target, dtype=torch.long)
    context_tensor = torch.tensor(context_indices, dtype=torch.long)
    type_tensor = torch.tensor(context_types, dtype=torch.long)
    if set(target_tensor.tolist()) & set(context_tensor.tolist()):
        raise AssertionError("Target leakage: target and context indices overlap")
    return Geometry(target_tensor, context_tensor, type_tensor, top, left)


def patch_foreground(
    images: torch.Tensor, patch_size: int, threshold: float = 1.0e-6
) -> torch.Tensor:
    """Return a [B, Gh, Gw] foreground mask without learning from lesion labels."""

    if images.ndim != 4:
        raise ValueError("images must have shape [B, C, H, W]")
    magnitude = images.abs().mean(dim=1, keepdim=True)
    occupancy = F.avg_pool2d(
        (magnitude > threshold).float(), kernel_size=patch_size, stride=patch_size
    )
    return occupancy[:, 0] > 0.05


def enumerate_geometries(
    grid_size: tuple[int, int],
    core_size: int,
    halo_size: int,
    context_radius: int,
    include_contralateral: bool,
    contralateral_size: int,
    stride: int = 1,
    foreground: torch.Tensor | None = None,
    anatomy_mode: str | None = None,
    anatomy_seed: int = 0,
) -> list[Geometry]:
    grid_height, grid_width = grid_size
    geometries: list[Geometry] = []
    for top in range(0, grid_height - core_size + 1, stride):
        for left in range(0, grid_width - core_size + 1, stride):
            if foreground is not None:
                core_foreground = foreground[top : top + core_size, left : left + core_size]
                if not bool(core_foreground.any()):
                    continue
            geometries.append(
                make_geometry(
                    top=top,
                    left=left,
                    grid_size=grid_size,
                    core_size=core_size,
                    halo_size=halo_size,
                    context_radius=context_radius,
                    include_contralateral=include_contralateral,
                    contralateral_size=contralateral_size,
                    anatomy_mode=anatomy_mode,
                    anatomy_seed=anatomy_seed,
                )
            )
    return geometries


def sample_geometries(
    foreground: torch.Tensor,
    count_per_image: int,
    core_size: int,
    halo_size: int,
    context_radius: int,
    include_contralateral: bool,
    contralateral_size: int,
    generator: torch.Generator | None = None,
    anatomy_mode: str | None = None,
    anatomy_seed: int = 0,
) -> tuple[list[Geometry], torch.Tensor]:
    """Sample target cores, preferring cores that intersect foreground.

    Returns geometries and the source image index for each geometry.
    """

    if foreground.ndim != 3:
        raise ValueError("foreground must have shape [B, Gh, Gw]")
    batch_size, grid_height, grid_width = foreground.shape
    geometries: list[Geometry] = []
    image_indices: list[int] = []
    all_positions = [
        (top, left)
        for top in range(grid_height - core_size + 1)
        for left in range(grid_width - core_size + 1)
    ]
    for batch_index in range(batch_size):
        candidates = []
        for top, left in all_positions:
            core = foreground[batch_index, top : top + core_size, left : left + core_size]
            if bool(core.any()):
                candidates.append((top, left))
        if not candidates:
            candidates = all_positions
        draws = torch.randint(
            low=0,
            high=len(candidates),
            size=(count_per_image,),
            generator=generator,
        )
        for draw in draws.tolist():
            top, left = candidates[draw]
            geometries.append(
                make_geometry(
                    top,
                    left,
                    (grid_height, grid_width),
                    core_size,
                    halo_size,
                    context_radius,
                    include_contralateral,
                    contralateral_size,
                    anatomy_mode,
                    anatomy_seed,
                )
            )
            image_indices.append(batch_index)
    return geometries, torch.tensor(image_indices, dtype=torch.long)


def pack_geometries(
    geometries: list[Geometry], device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length context sets for batched transformer attention."""

    if not geometries:
        raise ValueError("At least one geometry is required")
    target_length = geometries[0].target_indices.numel()
    if any(item.target_indices.numel() != target_length for item in geometries):
        raise ValueError("All geometries in a batch must use the same core size")
    max_context = max(item.context_indices.numel() for item in geometries)
    batch_size = len(geometries)
    target = torch.empty((batch_size, target_length), dtype=torch.long, device=device)
    context = torch.zeros((batch_size, max_context), dtype=torch.long, device=device)
    context_types = torch.zeros((batch_size, max_context), dtype=torch.long, device=device)
    padding = torch.ones((batch_size, max_context), dtype=torch.bool, device=device)
    for row, item in enumerate(geometries):
        length = item.context_indices.numel()
        target[row] = item.target_indices.to(device)
        context[row, :length] = item.context_indices.to(device)
        context_types[row, :length] = item.context_types.to(device)
        padding[row, :length] = False
    return target, context, context_types, padding

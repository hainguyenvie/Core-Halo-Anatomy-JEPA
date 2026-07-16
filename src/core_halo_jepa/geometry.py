from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

LOCAL_CONTEXT = 0
CONTRALATERAL_CONTEXT = 1


@dataclass(frozen=True)
class Geometry:
    """Token indices defining one target core and its target-free context."""

    target_indices: torch.Tensor
    context_indices: torch.Tensor
    context_types: torch.Tensor
    top: int
    left: int

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


def make_geometry(
    top: int,
    left: int,
    grid_size: tuple[int, int],
    core_size: int,
    halo_size: int,
    context_radius: int,
    include_contralateral: bool,
    contralateral_size: int,
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

    context: dict[int, int] = {}
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
            context[_flat(row, col, grid_width)] = LOCAL_CONTEXT

    if include_contralateral:
        target_center_row = top + (core_size - 1) / 2
        target_center_col = left + (core_size - 1) / 2
        mirror_center_col = (grid_width - 1) - target_center_col
        contra_top = round(target_center_row - (contralateral_size - 1) / 2)
        contra_left = round(mirror_center_col - (contralateral_size - 1) / 2)
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
                index = _flat(row, col, grid_width)
                context.setdefault(index, CONTRALATERAL_CONTEXT)

    if not context:
        raise ValueError("Geometry has no visible context; enlarge context_radius")
    target_tensor = torch.tensor(target, dtype=torch.long)
    context_tensor = torch.tensor(list(context), dtype=torch.long)
    type_tensor = torch.tensor(list(context.values()), dtype=torch.long)
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

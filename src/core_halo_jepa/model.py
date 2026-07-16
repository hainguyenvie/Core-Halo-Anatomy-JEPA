from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .geometry import Geometry, pack_geometries


def sincos_2d(
    grid_height: int,
    grid_width: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Deterministic 2-D sine/cosine positions in row-major order."""

    if dimension % 4:
        raise ValueError("Position dimension must be divisible by 4")
    quarter = dimension // 4
    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / max(quarter, 1)))
    rows = torch.arange(grid_height, device=device, dtype=torch.float32)
    cols = torch.arange(grid_width, device=device, dtype=torch.float32)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    row_phase = row_grid.reshape(-1, 1) * omega.reshape(1, -1)
    col_phase = col_grid.reshape(-1, 1) * omega.reshape(1, -1)
    encoding = torch.cat(
        [row_phase.sin(), row_phase.cos(), col_phase.sin(), col_phase.cos()], dim=-1
    )
    return encoding.to(dtype=dtype)


def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expanded = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return tokens.gather(dim=1, index=expanded)


class PatchTransformerEncoder(nn.Module):
    """Non-overlapping patch encoder; omitted tokens cannot leak through convolutions."""

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def patch_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        features = self.patch_embed(images)
        grid_size = (features.shape[-2], features.shape[-1])
        tokens = features.flatten(2).transpose(1, 2)
        position = sincos_2d(*grid_size, self.embed_dim, tokens.device, tokens.dtype)
        return tokens + position.unsqueeze(0), grid_size

    def forward_visible(
        self,
        images: torch.Tensor,
        indices: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        tokens, grid_size = self.patch_tokens(images)
        visible = gather_tokens(tokens, indices)
        return self.encode_visible_tokens(visible, padding_mask), grid_size

    def encode_visible_tokens(
        self,
        visible: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        visible = self.blocks(visible, src_key_padding_mask=padding_mask)
        return self.norm(visible)

    def forward_full(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        tokens, grid_size = self.patch_tokens(images)
        tokens = self.blocks(tokens)
        return self.norm(tokens), grid_size


class LatentPredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerDecoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        target_indices: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        position = sincos_2d(
            *grid_size,
            self.embed_dim,
            memory.device,
            memory.dtype,
        )
        target_position = position[target_indices]
        queries = self.query.expand(target_indices.shape[0], target_indices.shape[1], -1)
        queries = queries + target_position
        prediction = self.blocks(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.projection(self.norm(prediction))


@dataclass
class JepaOutput:
    prediction: torch.Tensor
    target: torch.Tensor
    target_indices: torch.Tensor

    @property
    def residual(self) -> torch.Tensor:
        return self.prediction - self.target


class CoreHaloJEPA(nn.Module):
    """EMA-target JEPA with a target-free halo and optional homologous context."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        encoder_args = dict(
            in_channels=int(config["in_channels"]),
            patch_size=int(config["patch_size"]),
            embed_dim=int(config["embed_dim"]),
            depth=int(config["encoder_depth"]),
            num_heads=int(config["num_heads"]),
            mlp_ratio=float(config["mlp_ratio"]),
            dropout=float(config["dropout"]),
        )
        self.context_encoder = PatchTransformerEncoder(**encoder_args)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.context_type_embedding = nn.Embedding(2, int(config["embed_dim"]))
        nn.init.trunc_normal_(self.context_type_embedding.weight, std=0.02)
        self.predictor = LatentPredictor(
            embed_dim=int(config["embed_dim"]),
            depth=int(config["predictor_depth"]),
            num_heads=int(config["num_heads"]),
            mlp_ratio=float(config["mlp_ratio"]),
            dropout=float(config["dropout"]),
        )
        self.patch_size = int(config["patch_size"])

    def train(self, mode: bool = True) -> CoreHaloJEPA:
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self, decay: float) -> None:
        for target, context in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters(), strict=True
        ):
            target.data.mul_(decay).add_(context.data, alpha=1.0 - decay)

    def forward(
        self,
        images: torch.Tensor,
        geometries: list[Geometry],
        image_indices: torch.Tensor | None = None,
        donor_indices: torch.Tensor | None = None,
    ) -> JepaOutput:
        if image_indices is None:
            if images.shape[0] != len(geometries):
                raise ValueError("One geometry is required for each image in the batch")
            image_indices = torch.arange(images.shape[0], device=images.device)
        else:
            image_indices = image_indices.to(images.device)
            if image_indices.numel() != len(geometries):
                raise ValueError("image_indices must map every geometry to one source image")
            if int(image_indices.max()) >= images.shape[0] or int(image_indices.min()) < 0:
                raise ValueError("image_indices contains an out-of-range source image")
        if donor_indices is not None:
            donor_indices = donor_indices.to(images.device)
            if donor_indices.shape != image_indices.shape:
                raise ValueError("donor_indices must map every geometry to one donor image")
            if int(donor_indices.max()) >= images.shape[0] or int(donor_indices.min()) < 0:
                raise ValueError("donor_indices contains an out-of-range source image")
        target_indices, context_indices, context_types, padding = pack_geometries(
            geometries, images.device
        )
        all_context_tokens, grid_size = self.context_encoder.patch_tokens(images)
        context_sources = image_indices[:, None].expand_as(context_indices)
        if donor_indices is not None:
            context_sources = torch.where(
                context_types == 1,
                donor_indices[:, None].expand_as(context_indices),
                context_sources,
            )
        context = all_context_tokens[context_sources, context_indices]
        context = self.context_encoder.encode_visible_tokens(context, padding)
        context = context + self.context_type_embedding(context_types)
        prediction = self.predictor(context, padding, target_indices, grid_size)
        with torch.no_grad():
            all_targets, target_grid_size = self.target_encoder.forward_full(images)
            if target_grid_size != grid_size:
                raise AssertionError("Context and target grids do not match")
            target = gather_tokens(all_targets[image_indices], target_indices)
        return JepaOutput(prediction, target.detach(), target_indices)


def build_model(config: dict) -> CoreHaloJEPA:
    return CoreHaloJEPA(config)

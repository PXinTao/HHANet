# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPADConvOffsetStage(nn.Module):
    """Content-Position Aware Dynamic Conv (CPADConv) with external offset map.

    Unifies population-level anatomical position priors with instance-level
    deformation fields via dynamic convolution for content-position dual awareness.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stage: int = 1,
        posi_channels: int = 16,
        posi_grid_size: int = 16,
        offset_scale: float = 0.5,
        bias: bool = True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = kernel_size
        self.stage = stage
        self.offset_scale = offset_scale

        self.posi_map = nn.Parameter(torch.ones(1, posi_channels, posi_grid_size, posi_grid_size))

        adapt = []
        for _ in range(max(0, stage - 1)):
            adapt.append(nn.Conv2d(2, 2, 3, stride=2, padding=1, groups=2, bias=False))
        self.offset_adapt = nn.Sequential(*adapt) if adapt else nn.Identity()

        hidden = max(out_ch // 2, 16)
        self.weight_gen = nn.Sequential(
            nn.Conv2d(posi_channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch * (kernel_size * kernel_size), 1, bias=True),
        )

        self.channel_adapter = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

        # init: near-identity center weight
        with torch.no_grad():
            last = self.weight_gen[-1]
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                k2 = self.k * self.k
                bias_vec = torch.zeros(out_ch * k2)
                center = (self.k // 2) * self.k + (self.k // 2)
                for c in range(out_ch):
                    bias_vec[c * k2 + center] = 0.1
                last.bias.copy_(bias_vec)

    @staticmethod
    def _make_base_grid(B: int, H: int, W: int, device) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, steps=H, device=device)
        xs = torch.linspace(-1.0, 1.0, steps=W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij') if 'indexing' in torch.meshgrid.__code__.co_varnames else torch.meshgrid(ys, xs)
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        return grid

    def _norm_offset(self, offset: torch.Tensor, H: int, W: int) -> torch.Tensor:
        dx = offset[:, 0:1] * (2.0 / max(W - 1, 1)) * self.offset_scale
        dy = offset[:, 1:2] * (2.0 / max(H - 1, 1)) * self.offset_scale
        return torch.cat([dx, dy], dim=1)

    def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        device = x.device

        off = self.offset_adapt(offset)
        if off.shape[-2:] != (H, W):
            off = F.adaptive_avg_pool2d(off, (H, W))

        base_grid = self._make_base_grid(B, H, W, device)
        off_norm = self._norm_offset(off, H, W)
        grid = base_grid + off_norm.permute(0, 2, 3, 1)

        up_posi = F.grid_sample(
            self.posi_map.repeat(B, 1, 1, 1),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )

        weights = self.weight_gen(up_posi)  # (B, Cout*k*k, H, W)
        x_adapt = self.channel_adapter(x)  # (B, Cout, H, W)

        patches = F.unfold(x_adapt, kernel_size=self.k, padding=self.k // 2)  # (B, Cout*k*k, H*W)
        patches = patches.view(B, self.out_ch, self.k * self.k, H, W)
        weights = weights.view(B, self.out_ch, self.k * self.k, H, W)

        out = (weights * patches).sum(dim=2)
        if self.use_bias:
            out = out + self.bias
        return out


class StaticConvStage(nn.Module):
    """Ablation replacement for CPADConv: plain 3x3 conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=bias)

    def forward(self, x: torch.Tensor, offset: torch.Tensor | None = None) -> torch.Tensor:
        return self.conv(x)

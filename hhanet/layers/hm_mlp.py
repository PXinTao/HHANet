# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tok_mlp import TokenDWConv


class HMSwiGLUMLP(nn.Module):
    """Hierarchically Modulated SwiGLU MLP (HM-MLP).

    Linear-complexity global semantic aggregation with cross-stage FiLM
    calibration to prevent bottleneck-layer representation degradation.

    Args:
        dim: Input/output feature dimension.
        hidden: Hidden dimension (defaults to dim * 2).
        cond_ch: Channel count of the conditioning feature map (from prior stage).
        drop: Dropout rate.
        use_dw: Whether to use depthwise conv for light spatial mixing.
        use_film: If False, gamma=0, beta=0 (ablation of cross-stage modulation).
    """

    def __init__(self, dim: int, hidden: int | None = None, cond_ch: int | None = None, drop: float = 0.0, use_dw: bool = True, use_film: bool = True):
        super().__init__()
        if cond_ch is None or cond_ch <= 0:
            raise ValueError("HMSwiGLUMLP: cond_ch must be set (>0)")
        self.use_film = use_film
        hidden = hidden or dim * 2
        self.hidden = hidden

        self.fc1 = nn.Linear(dim, hidden * 2)
        self.dw = TokenDWConv(hidden) if use_dw else nn.Identity()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

        self.cond_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cond_ch, 2 * hidden, 1),
        )

    @staticmethod
    def swiglu(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.silu(u) * v

    def forward(self, x: torch.Tensor, H: int, W: int, cond_map: torch.Tensor) -> torch.Tensor:
        # 1) FiLM params
        if self.use_film:
            gb = self.cond_head(cond_map).flatten(2).transpose(1, 2).contiguous()  # (B,1,2H)
            gamma, beta = gb.chunk(2, dim=-1)  # (B,1,H),(B,1,H)
            gamma = torch.tanh(gamma)
        else:
            B = x.shape[0]
            device = x.device
            gamma = torch.zeros((B, 1, self.hidden), device=device, dtype=x.dtype)
            beta = torch.zeros((B, 1, self.hidden), device=device, dtype=x.dtype)

        # 2) SwiGLU + (optional) FiLM
        u, v = self.fc1(x).chunk(2, dim=-1)
        u = u * (1.0 + gamma) + beta
        y = self.swiglu(u, v)

        # 3) light spatial mixing + proj
        y = self.dw(y, H, W)
        y = self.drop(self.fc2(y))
        return y


class HMSwiGLUBlock(nn.Module):
    """Residual block wrapper for HMSwiGLUMLP."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.0, norm_layer=nn.LayerNorm, cond_ch: int | None = None, use_dw: bool = True, use_film: bool = True):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mlp = HMSwiGLUMLP(dim, hidden=int(dim * mlp_ratio), cond_ch=cond_ch, drop=drop, use_dw=use_dw, use_film=use_film)

    def forward(self, x: torch.Tensor, H: int, W: int, cond_map: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x), H, W, cond_map)


class PlainSwiGLUBlock(nn.Module):
    """Ablation block: Stage-5 without conditioning (no FiLM, no cond_map usage)."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.0, norm_layer=nn.LayerNorm, use_dw: bool = True):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.dw = TokenDWConv(hidden) if use_dw else nn.Identity()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int, cond_map: torch.Tensor | None = None) -> torch.Tensor:
        y = self.norm(x)
        u, v = self.fc1(y).chunk(2, dim=-1)
        y = F.silu(u) * v
        y = self.dw(y, H, W)
        y = self.drop(self.fc2(y))
        return x + y

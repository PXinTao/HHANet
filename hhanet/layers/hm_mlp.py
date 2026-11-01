from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tok_mlp import TokenDWConv


class HMSwiGLUMLP(nn.Module):
    """Input-conditioned SwiGLU with cross-stage calibration and depthwise spatial mixing."""

    def __init__(
        self,
        dim: int,
        hidden: int | None = None,
        cond_ch: int | None = None,
        drop: float = 0.0,
    ):
        super().__init__()
        if cond_ch is None or cond_ch <= 0:
            raise ValueError("HMSwiGLUMLP: cond_ch must be set (>0)")
        hidden = hidden or dim * 2
        self.hidden = hidden
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.dw = TokenDWConv(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
        self.cond_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(cond_ch, 2 * hidden, 1)
        )

    @staticmethod
    def swiglu(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.silu(u) * v

    def forward(
        self, x: torch.Tensor, H: int, W: int, cond_map: torch.Tensor
    ) -> torch.Tensor:
        gb = self.cond_head(cond_map).flatten(2).transpose(1, 2).contiguous()
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        u, v = self.fc1(x).chunk(2, dim=-1)
        u = u * (1.0 + gamma) + beta
        y = self.swiglu(u, v)
        y = self.dw(y, H, W)
        y = self.drop(self.fc2(y))
        return y


class HMSwiGLUBlock(nn.Module):
    """Residual block wrapper for HMSwiGLUMLP."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        norm_layer=nn.LayerNorm,
        cond_ch: int | None = None,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mlp = HMSwiGLUMLP(
            dim, hidden=int(dim * mlp_ratio), cond_ch=cond_ch, drop=drop
        )

    def forward(
        self, x: torch.Tensor, H: int, W: int, cond_map: torch.Tensor
    ) -> torch.Tensor:
        return x + self.mlp(self.norm(x), H, W, cond_map)

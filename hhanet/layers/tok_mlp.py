# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.factor import safe_factor_hw


class TokenDWConv(nn.Module):
    """Depthwise 3x3 conv applied on token sequence by reshaping to (H, W)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        H, W = safe_factor_hw(N, H, W)
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dw(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ShiftMLP(nn.Module):
    """Shift-MLP (Stage-4) with safe H/W inference."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
        shift_size: int = 5,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = TokenDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.shift_size = shift_size
        self.pad = shift_size // 2

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        H, W = safe_factor_hw(N, H, W)

        # shift along H
        xn = x.transpose(1, 2).contiguous().view(B, C, H, W)
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, s, 2) for x_c, s in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W).reshape(B, C, H * W).transpose(1, 2).contiguous()

        x = self.fc1(x_s)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        # shift along W
        xn = x.transpose(1, 2).contiguous().view(B, x.shape[-1], H, W)
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, s, 3) for x_c, s in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W).reshape(B, x.shape[-1], H * W).transpose(1, 2).contiguous()

        x = self.fc2(x_s)
        x = self.drop(x)
        return x


class ShiftedBlock(nn.Module):
    """Residual block wrapper for ShiftMLP."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = ShiftMLP(in_features=dim, hidden_features=hidden, drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return x + self.mlp(self.norm(x), H, W)


class PlainMLPBlock(nn.Module):
    """Ablation block: Token-MLP without spatial shift (no explicit mixing)."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return x + y


class OverlapPatchEmbed(nn.Module):
    """Conv-based overlapping patch embedding."""

    def __init__(self, img_size: int, patch_size: int = 3, stride: int = 2, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x, H, W

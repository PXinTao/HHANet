# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.blocks import DWConvBlock
from ..layers.cpadconv import CPADConvOffsetStage, StaticConvStage
from ..layers.hm_mlp import HMSwiGLUBlock, PlainSwiGLUBlock
from ..layers.tok_mlp import OverlapPatchEmbed, PlainMLPBlock, ShiftedBlock


class HHAEncoder(nn.Module):
    """HHANet encoder (S1-S5) with optional ablation switches.

    Stages:
        S1/S2: CPADConv (or plain conv for ablation)
        S3: DWConvBlock
        S4: ShiftedBlock (or PlainMLPBlock for ablation)
        S5: HMSwiGLUBlock (FiLM) (or FiLM-off / PlainSwiGLUBlock)

    Args:
        use_cpad: enable CPADConv in S1/S2
        use_shift: enable shift mixing in S4
        use_film: enable FiLM conditioning in S5
        use_stage5_hm: if False, S5 becomes plain SwiGLU (no cond at all)
    """

    def __init__(
        self,
        in_ch: int = 3,
        dims: tuple[int, int, int, int, int] = (64, 128, 256, 320, 512),
        depths: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1),
        img_size: int = 224,
        cpad_kernel: int = 3,
        cpad_offset_scale: float = 0.5,
        mlp_ratio_stage4: float = 2.0,
        drop_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        # ablation switches
        use_cpad: bool = True,
        use_shift: bool = True,
        use_film: bool = True,
        use_stage5_hm: bool = True,
    ):
        super().__init__()
        self.dims = dims
        self.depths = depths
        self.use_cpad = use_cpad
        self.use_shift = use_shift
        self.use_film = use_film
        self.use_stage5_hm = use_stage5_hm

        C1, C2, C3, C4, C5 = dims
        d1, d2, d3, d4, d5 = depths

        # ------------------- Stage 1 -------------------
        self.s1_blocks = nn.ModuleList()
        if use_cpad:
            self.s1_blocks.append(
                CPADConvOffsetStage(
                    in_ch,
                    C1,
                    kernel_size=cpad_kernel,
                    stage=1,
                    offset_scale=cpad_offset_scale,
                    posi_grid_size=max(4, img_size // 2),
                )
            )
            for _ in range(max(0, d1 - 1)):
                self.s1_blocks.append(
                    CPADConvOffsetStage(
                        C1,
                        C1,
                        kernel_size=cpad_kernel,
                        stage=1,
                        offset_scale=cpad_offset_scale,
                        posi_grid_size=max(4, img_size // 2),
                    )
                )
        else:
            self.s1_blocks.append(StaticConvStage(in_ch, C1, kernel_size=3, bias=False))
            for _ in range(max(0, d1 - 1)):
                self.s1_blocks.append(StaticConvStage(C1, C1, kernel_size=3, bias=False))
        self.bn1 = nn.BatchNorm2d(C1)

        # ------------------- Stage 2 -------------------
        self.s2_blocks = nn.ModuleList()
        if use_cpad:
            self.s2_blocks.append(
                CPADConvOffsetStage(
                    C1,
                    C2,
                    kernel_size=cpad_kernel,
                    stage=2,
                    offset_scale=cpad_offset_scale,
                    posi_grid_size=max(4, img_size // 4),
                )
            )
            for _ in range(max(0, d2 - 1)):
                self.s2_blocks.append(
                    CPADConvOffsetStage(
                        C2,
                        C2,
                        kernel_size=cpad_kernel,
                        stage=2,
                        offset_scale=cpad_offset_scale,
                        posi_grid_size=max(4, img_size // 4),
                    )
                )
        else:
            self.s2_blocks.append(StaticConvStage(C1, C2, kernel_size=3, bias=False))
            for _ in range(max(0, d2 - 1)):
                self.s2_blocks.append(StaticConvStage(C2, C2, kernel_size=3, bias=False))
        self.bn2 = nn.BatchNorm2d(C2)

        # ------------------- Stage 3 -------------------
        s3_layers = [DWConvBlock(C2, C3)]
        for _ in range(max(0, d3 - 1)):
            s3_layers.append(DWConvBlock(C3, C3))
        self.s3 = nn.Sequential(*s3_layers)
        self.bn3 = nn.BatchNorm2d(C3)

        # ------------------- Stage 4 -------------------
        self.patch_s4 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=C3, embed_dim=C4)
        self.s4_blocks = nn.ModuleList()
        blk4 = ShiftedBlock if use_shift else PlainMLPBlock
        for _ in range(max(1, d4)):
            self.s4_blocks.append(blk4(dim=C4, mlp_ratio=mlp_ratio_stage4, drop=drop_rate, norm_layer=norm_layer))
        self.norm_s4 = norm_layer(C4)

        # ------------------- Stage 5 -------------------
        self.patch_s5 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=C4, embed_dim=C5)
        self.s5_blocks = nn.ModuleList()
        if use_stage5_hm:
            for _ in range(max(1, d5)):
                self.s5_blocks.append(
                    HMSwiGLUBlock(dim=C5, mlp_ratio=2.0, drop=drop_rate, norm_layer=norm_layer, cond_ch=C4, use_dw=True, use_film=use_film)
                )
        else:
            for _ in range(max(1, d5)):
                self.s5_blocks.append(PlainSwiGLUBlock(dim=C5, mlp_ratio=2.0, drop=drop_rate, norm_layer=norm_layer, use_dw=True))
        self.norm_s5 = norm_layer(C5)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor, offset: torch.Tensor) -> dict[str, torch.Tensor]:
        B, _, H, W = x.shape

        # Stage 1
        y = x
        for blk in self.s1_blocks:
            # CPAD needs offset, StaticConv ignores
            y = blk(y, offset)
        x1 = F.relu(self.bn1(y), inplace=True)
        x1_ds = self.pool(x1)  # H/2

        # Stage 2
        y = x1_ds
        for blk in self.s2_blocks:
            y = blk(y, offset)
        x2 = F.relu(self.bn2(y), inplace=True)
        x2_ds = self.pool(x2)  # H/4

        # Stage 3
        x3 = F.relu(self.bn3(self.s3(x2_ds)), inplace=True)
        x3_ds = self.pool(x3)  # H/8

        # Stage 4
        t4, H4, W4 = self.patch_s4(x3_ds)  # H/8 -> H/16
        for blk in self.s4_blocks:
            t4 = blk(t4, H4, W4)
        t4 = self.norm_s4(t4)
        s4 = t4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()

        # Stage 5
        t5, H5, W5 = self.patch_s5(s4)  # H/16 -> H/32
        if self.use_stage5_hm:
            for blk in self.s5_blocks:
                t5 = blk(t5, H5, W5, cond_map=s4)
        else:
            for blk in self.s5_blocks:
                t5 = blk(t5, H5, W5, cond_map=None)
        t5 = self.norm_s5(t5)
        s5 = t5.reshape(B, H5, W5, -1).permute(0, 3, 1, 2).contiguous()

        return {
            's1': x1_ds,
            's2': x2_ds,
            's3': x3_ds,
            's4': s4,
            's5': s5,
        }

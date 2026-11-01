from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.blocks import DWConvBlock
from ..layers.cpadconv import CPADConvOffsetStage
from ..layers.hm_mlp import HMSwiGLUBlock
from ..layers.tok_mlp import OverlapPatchEmbed, ShiftedBlock


class HHAEncoder(nn.Module):
    """Five-stage HHANet encoder: CPADConv, depthwise transition, and HM-MLP."""

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
    ):
        super().__init__()
        self.dims = dims
        self.depths = depths
        C1, C2, C3, C4, C5 = dims
        d1, d2, d3, d4, d5 = depths
        self.s1_blocks = nn.ModuleList()
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
        self.bn1 = nn.BatchNorm2d(C1)
        self.s2_blocks = nn.ModuleList()
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
        self.bn2 = nn.BatchNorm2d(C2)
        s3_layers = [DWConvBlock(C2, C3)]
        for _ in range(max(0, d3 - 1)):
            s3_layers.append(DWConvBlock(C3, C3))
        self.s3 = nn.Sequential(*s3_layers)
        self.bn3 = nn.BatchNorm2d(C3)
        self.patch_s4 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2, in_chans=C3, embed_dim=C4
        )
        self.s4_blocks = nn.ModuleList()
        blk4 = ShiftedBlock
        for _ in range(max(1, d4)):
            self.s4_blocks.append(
                blk4(
                    dim=C4,
                    mlp_ratio=mlp_ratio_stage4,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                )
            )
        self.norm_s4 = norm_layer(C4)
        self.patch_s5 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2, in_chans=C4, embed_dim=C5
        )
        self.s5_blocks = nn.ModuleList()
        for _ in range(max(1, d5)):
            self.s5_blocks.append(
                HMSwiGLUBlock(
                    dim=C5,
                    mlp_ratio=2.0,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                    cond_ch=C4,
                )
            )
        self.norm_s5 = norm_layer(C5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor, offset: torch.Tensor) -> dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        y = x
        for blk in self.s1_blocks:
            y = blk(y, offset)
        x1 = F.relu(self.bn1(y), inplace=True)
        x1_ds = self.pool(x1)
        y = x1_ds
        for blk in self.s2_blocks:
            y = blk(y, offset)
        x2 = F.relu(self.bn2(y), inplace=True)
        x2_ds = self.pool(x2)
        x3 = F.relu(self.bn3(self.s3(x2_ds)), inplace=True)
        x3_ds = self.pool(x3)
        t4, H4, W4 = self.patch_s4(x3_ds)
        for blk in self.s4_blocks:
            t4 = blk(t4, H4, W4)
        t4 = self.norm_s4(t4)
        s4 = t4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        t5, H5, W5 = self.patch_s5(s4)
        for blk in self.s5_blocks:
            t5 = blk(t5, H5, W5, cond_map=s4)
        t5 = self.norm_s5(t5)
        s5 = t5.reshape(B, H5, W5, -1).permute(0, 3, 1, 2).contiguous()
        return {"s1": x1_ds, "s2": x2_ds, "s3": x3_ds, "s4": s4, "s5": s5}

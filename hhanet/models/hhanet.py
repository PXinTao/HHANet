from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.blocks import UpBlock
from ..layers.gca import GCAAlign
from .encoder import HHAEncoder


class HHANet(nn.Module):
    """HHANet with CPADConv, HM-MLP, and prototype-based cross-level alignment."""

    def __init__(
        self,
        num_classes: int,
        in_ch: int = 3,
        dims: tuple[int, int, int, int, int] = (16, 32, 64, 64, 128),
        depths: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1),
        img_size: int = 256,
        gca_d_model: tuple[int, int, int] = (64, 48, 32),
        gca_num_proto: tuple[int, int, int] = (32, 24, 16),
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_class = num_classes
        self.in_ch = in_ch
        self.encoder = HHAEncoder(
            in_ch=in_ch,
            dims=dims,
            depths=depths,
            img_size=img_size,
            drop_rate=drop_rate,
        )
        C1, C2, C3, C4, C5 = dims
        self.offset = nn.Conv2d(
            in_channels=in_ch, out_channels=2, kernel_size=7, padding=3
        )
        self.head = nn.Conv2d(C1, num_classes, 1)
        d52 = gca_d_model[0] if len(gca_d_model) > 0 else 256
        d41 = gca_d_model[1] if len(gca_d_model) > 1 else 128
        k52 = gca_num_proto[0] if len(gca_num_proto) > 0 else 32
        k41 = gca_num_proto[1] if len(gca_num_proto) > 1 else 16
        self.gca_52 = GCAAlign(
            dim_S=C2, dim_C=C5, d_model=d52, num_proto=k52, dropout=drop_rate
        )
        self.gca_41 = GCAAlign(
            dim_S=C1, dim_C=C4, d_model=d41, num_proto=k41, dropout=drop_rate
        )
        self.dec_s5_s3 = UpBlock(C5, C3, C3)
        self.dec_s3_s2 = UpBlock(C3, C2, C2)
        self.dec_s2_s1 = UpBlock(C2, C1, C1)
        self.dec_s1_out = UpBlock(C1, 0, C1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1 and self.in_ch == 3:
            x = x.repeat(1, 3, 1, 1)
        if self.offset is not None:
            offset = self.offset(x)
        else:
            offset = torch.zeros(
                (x.shape[0], 2, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype
            )
        feats = self.encoder(x, offset)
        s1, s2, s3, s4, s5 = (
            feats["s1"],
            feats["s2"],
            feats["s3"],
            feats["s4"],
            feats["s5"],
        )
        s2_aligned, _ = self.gca_52(s2, s5)
        s1_aligned, _ = self.gca_41(s1, s4)
        y = self.dec_s5_s3(s5, s3, scale=4)
        y = self.dec_s3_s2(y, s2_aligned, scale=2)
        y = self.dec_s2_s1(y, s1_aligned, scale=2)
        y = self.dec_s1_out(y, None, scale=2)
        out = self.head(y)
        if self.num_class == 1:
            return out.sigmoid()
        return out

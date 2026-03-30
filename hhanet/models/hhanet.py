# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.blocks import UpBlock
from ..layers.gca import GCAAlign
from .encoder import HHAEncoder


class HHANet(nn.Module):
    """HHANet: Hierarchical Heterogeneous Alignment Network for medical image segmentation.

    align_variant:
        - "chain": gca_54(s5->s4), gca_42(s4->s2), gca_21(s2->s1) + chain decoder
        - "s3_trunk": gca_52(s5->s2), gca_41(s4->s1) + s3-based slim decoder
    """

    def __init__(
        self,
        num_classes: int,
        in_ch: int = 3,
        dims: tuple[int, int, int, int, int] = (16, 32, 64, 64, 128),
        depths: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1),
        img_size: int = 256,
        # encoder switches
        use_cpad: bool = True,
        use_shift: bool = True,
        use_film: bool = True,
        use_stage5_hm: bool = True,
        # gca config
        gca_d_model: tuple[int, int, int] = (64, 48, 32),
        gca_num_proto: tuple[int, int, int] = (32, 24, 16),
        use_gca: bool = True,
        use_gca_align: bool = True,
        align_variant: str = "s3_trunk",
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_class = num_classes
        self.in_ch = in_ch
        self.align_variant = align_variant
        self.use_gca = use_gca
        self.use_cpad = use_cpad

        # encoder
        self.encoder = HHAEncoder(
            in_ch=in_ch,
            dims=dims,
            depths=depths,
            img_size=img_size,
            drop_rate=drop_rate,
            use_cpad=use_cpad,
            use_shift=use_shift,
            use_film=use_film,
            use_stage5_hm=use_stage5_hm,
        )

        C1, C2, C3, C4, C5 = dims

        # offset predictor only needed when CPAD is enabled
        self.offset = nn.Conv2d(in_channels=in_ch, out_channels=2, kernel_size=7, padding=3) if use_cpad else None

        # head
        self.head = nn.Conv2d(C1, num_classes, 1)

        # -------------------- variants --------------------
        if align_variant == "chain":
            d54 = gca_d_model[0] if len(gca_d_model) > 0 else 256
            d42 = gca_d_model[1] if len(gca_d_model) > 1 else 192
            d21 = gca_d_model[2] if len(gca_d_model) > 2 else 128
            k54 = gca_num_proto[0] if len(gca_num_proto) > 0 else 32
            k42 = gca_num_proto[1] if len(gca_num_proto) > 1 else 24
            k21 = gca_num_proto[2] if len(gca_num_proto) > 2 else 16

            if use_gca:
                self.gca_54 = GCAAlign(dim_S=C4, dim_C=C5, d_model=d54, num_proto=k54, use_gca=use_gca_align, dropout=drop_rate)
                self.gca_42 = GCAAlign(dim_S=C2, dim_C=C4, d_model=d42, num_proto=k42, use_gca=use_gca_align, dropout=drop_rate)
                self.gca_21 = GCAAlign(dim_S=C1, dim_C=C2, d_model=d21, num_proto=k21, use_gca=use_gca_align, dropout=drop_rate)
            else:
                self.gca_54 = None
                self.gca_42 = None
                self.gca_21 = None

            self.dec5 = UpBlock(C5, C4, C4)
            self.dec4 = UpBlock(C4, C3, C3)
            self.dec3 = UpBlock(C3, C2, C2)
            self.dec2 = UpBlock(C2, C1, C1)
            self.dec1 = UpBlock(C1, 0, C1)

        elif align_variant == "s3_trunk":
            d52 = gca_d_model[0] if len(gca_d_model) > 0 else 256
            d41 = gca_d_model[1] if len(gca_d_model) > 1 else 128
            k52 = gca_num_proto[0] if len(gca_num_proto) > 0 else 32
            k41 = gca_num_proto[1] if len(gca_num_proto) > 1 else 16

            if use_gca:
                self.gca_52 = GCAAlign(dim_S=C2, dim_C=C5, d_model=d52, num_proto=k52, use_gca=use_gca_align, dropout=drop_rate)
                self.gca_41 = GCAAlign(dim_S=C1, dim_C=C4, d_model=d41, num_proto=k41, use_gca=use_gca_align, dropout=drop_rate)
            else:
                self.gca_52 = None
                self.gca_41 = None

            self.dec_s5_s3 = UpBlock(C5, C3, C3)
            self.dec_s3_s2 = UpBlock(C3, C2, C2)
            self.dec_s2_s1 = UpBlock(C2, C1, C1)
            self.dec_s1_out = UpBlock(C1, 0, C1)

        else:
            raise ValueError(f"Unknown align_variant: {align_variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # single-channel -> 3-channel (for ultrasound / grayscale)
        if x.size(1) == 1 and self.in_ch == 3:
            x = x.repeat(1, 3, 1, 1)

        # offset
        if self.offset is not None:
            offset = self.offset(x)
        else:
            offset = torch.zeros((x.shape[0], 2, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)

        feats = self.encoder(x, offset)
        s1, s2, s3, s4, s5 = feats['s1'], feats['s2'], feats['s3'], feats['s4'], feats['s5']

        # -------------------- decode --------------------
        if self.align_variant == "chain":
            if self.use_gca:
                s4_aligned, _ = self.gca_54(s4, s5)
                s2_aligned, _ = self.gca_42(s2, s4_aligned)
                s1_aligned, _ = self.gca_21(s1, s2_aligned)
            else:
                s4_aligned, s2_aligned, s1_aligned = s4, s2, s1

            y = self.dec5(s5, s4_aligned, scale=2)
            y = self.dec4(y, s3, scale=2)
            y = self.dec3(y, s2_aligned, scale=2)
            y = self.dec2(y, s1_aligned, scale=2)
            y = self.dec1(y, None, scale=2)

        else:  # s3_trunk
            if self.use_gca:
                s2_aligned, _ = self.gca_52(s2, s5)
                s1_aligned, _ = self.gca_41(s1, s4)
            else:
                s2_aligned, s1_aligned = s2, s1

            y = self.dec_s5_s3(s5, s3, scale=4)
            y = self.dec_s3_s2(y, s2_aligned, scale=2)
            y = self.dec_s2_s1(y, s1_aligned, scale=2)
            y = self.dec_s1_out(y, None, scale=2)

        out = self.head(y)
        if self.num_class == 1:
            return out.sigmoid()
        return out

# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from .hhanet import HHANet


# -------------------- Variant presets --------------------
_VARIANTS: dict[str, dict] = {
    "tiny": dict(
        dims=(16, 32, 64, 64, 128),
        depths=(1, 1, 1, 1, 1),
    ),
    "base": dict(
        dims=(32, 64, 128, 256, 512),
        depths=(2, 3, 4, 3, 2),
    ),
}


@dataclass
class HHANetConfig:
    # backbone
    dims: tuple[int, int, int, int, int] = (16, 32, 64, 64, 128)
    depths: tuple[int, int, int, int, int] = (1, 1, 1, 1, 1)
    img_size: int = 256

    # encoder switches
    use_cpad: bool = True
    use_shift: bool = True
    use_film: bool = True
    use_stage5_hm: bool = True

    # gca
    gca_d_model: tuple[int, int, int] = (64, 48, 32)
    gca_num_proto: tuple[int, int, int] = (32, 24, 16)
    use_gca: bool = True
    use_gca_align: bool = True
    align_variant: str = "s3_trunk"

    # misc
    drop_rate: float = 0.0


# -------------------- Legacy key mapping --------------------
_LEGACY_KEY_RULES: list[tuple[str, str]] = [
    ("gcr", "gca"),
    ("cond_", "hm_"),
]


def _remap_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename keys from the old HappyNet checkpoint to HHANet naming."""
    new_sd: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in _LEGACY_KEY_RULES:
            new_k = new_k.replace(old, new)
        new_sd[new_k] = v
    return new_sd


# -------------------- Builder --------------------
def build_hhanet(
    variant: str = "tiny",
    num_classes: int = 1,
    in_ch: int = 3,
    cfg: HHANetConfig | None = None,
    pretrained: str | None = None,
    legacy_ckpt: bool = False,
    **kwargs,
) -> HHANet:
    """Build an HHANet model.

    Args:
        variant: 'tiny' or 'base'.
        num_classes: number of output classes.
        in_ch: input channels.
        cfg: optional config override (takes precedence over variant preset).
        pretrained: path to a checkpoint file.
        legacy_ckpt: if True, remap old HappyNet key names before loading.
        **kwargs: override individual config fields.
    """
    if cfg is None:
        preset = _VARIANTS.get(variant.lower(), {})
        cfg = HHANetConfig(**{k: v for k, v in preset.items() if hasattr(HHANetConfig, k)})

    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise ValueError(f"Unknown config field: {k}")

    model = HHANet(
        num_classes=num_classes,
        in_ch=in_ch,
        dims=cfg.dims,
        depths=cfg.depths,
        img_size=cfg.img_size,
        use_cpad=cfg.use_cpad,
        use_shift=cfg.use_shift,
        use_film=cfg.use_film,
        use_stage5_hm=cfg.use_stage5_hm,
        gca_d_model=cfg.gca_d_model,
        gca_num_proto=cfg.gca_num_proto,
        use_gca=cfg.use_gca,
        use_gca_align=cfg.use_gca_align,
        align_variant=cfg.align_variant,
        drop_rate=cfg.drop_rate,
    )

    if pretrained is not None:
        ckpt = torch.load(pretrained, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        if legacy_ckpt:
            sd = _remap_legacy_state_dict(sd)
        model.load_state_dict(sd, strict=False)

    return model

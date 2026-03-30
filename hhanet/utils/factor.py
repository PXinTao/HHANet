# -*- coding: utf-8 -*-

from __future__ import annotations


def safe_factor_hw(N: int, hintH: int = 0, hintW: int = 0) -> tuple[int, int]:
    """Infer a valid (H, W) factorization for token length N.

    - Prefer matching provided hints to avoid shape mismatch.
    - Guarantees to return a pair that multiplies to N.
    """
    if hintH and N % hintH == 0:
        return hintH, N // hintH
    if hintW and N % hintW == 0:
        return N // hintW, hintW

    r = int(N ** 0.5)
    for h in range(r, 0, -1):
        if N % h == 0:
            return h, N // h
    return 1, N

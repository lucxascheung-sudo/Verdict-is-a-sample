"""The cheap tier: mean absolute difference (MAD).

Deterministic, no model, no randomness. Measures *how much* two images differ,
not *whether* the difference is correct. See README for that failure mode.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

SIDE = 384


def _load(path: str, side: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((side, side), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32)


def mad(path_a: str, path_b: str, side: int = SIDE) -> float:
    """Mean absolute pixel difference of two images resized to `side` x `side`.

    Returns 0.0 for byte-identical images. Roughly: <8 means nothing really
    changed, >40 means a substantial change. Those cutoffs are heuristics for
    this instrument, not thresholds with any general meaning.
    """
    return float(np.abs(_load(path_a, side) - _load(path_b, side)).mean())


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("usage: python mad.py IMAGE_A IMAGE_B")
    print(f"MAD = {mad(sys.argv[1], sys.argv[2]):.2f}")

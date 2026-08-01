"""Dynamic-region masking for fidelity metrics.

Fidelity metrics (PSNR/SSIM/LPIPS) are dominated by static background pixels; a
model can score well while getting every *moving* pixel wrong. SPEC §4 requires
every fidelity metric to also report a variant restricted to moving regions.
The masking logic lives here, not inside individual metrics (CLAUDE.md), so the
metrics only apply a mask they are handed.

The mask is derived from the **reference** (ground-truth) frames by temporal
differencing: a pixel is "dynamic" at step ``t`` when it changed appreciably
between reference frames ``t-1`` and ``t``. This is deterministic and needs no
tracker; tracker-based masks can replace it later behind the same interface.
"""

from __future__ import annotations

import numpy as np

__all__ = ["dynamic_region_mask", "DEFAULT_MOTION_THRESHOLD"]

DEFAULT_MOTION_THRESHOLD = 10
"""Per-pixel intensity change (0-255) above which a pixel counts as moving."""


def dynamic_region_mask(
    reference: np.ndarray, *, threshold: int = DEFAULT_MOTION_THRESHOLD
) -> np.ndarray:
    """Boolean moving-region mask ``(T, H, W)`` from reference frames.

    Args:
        reference: Ground-truth pixel frames ``(T, H, W, C)`` uint8.
        threshold: A pixel is dynamic at step ``t`` when the max-over-channels
            absolute difference from step ``t-1`` exceeds this (0-255).

    Returns:
        ``(T, H, W)`` boolean array; ``True`` marks a moving pixel. Step 0 reuses
        step 1's motion. A single-frame reference (``T == 1``) has no temporal
        signal, so the whole frame is marked dynamic (nothing is excluded).
    """
    if reference.ndim != 4:
        raise ValueError(
            f"reference must be (T, H, W, C) pixel frames, got shape {reference.shape}"
        )
    if reference.dtype != np.uint8:
        raise ValueError(
            "reference must be uint8 pixels (the threshold is in 0-255 units); "
            f"got dtype {reference.dtype}"
        )
    frames = reference.astype(np.int16)
    horizon = frames.shape[0]
    if horizon == 1:
        return np.ones(frames.shape[:3], dtype=bool)

    step_diff = np.abs(frames[1:] - frames[:-1]).max(axis=-1)  # (T-1, H, W)
    motion = np.concatenate([step_diff[:1], step_diff], axis=0)  # (T, H, W)
    return motion > threshold

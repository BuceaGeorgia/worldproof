"""Tier-1 pixel-fidelity metrics: PSNR and SSIM, as horizon curves.

Each metric compares every predicted frame against the ground-truth frame at the
same step and returns a per-sample ``(n_samples, horizon)`` curve — never a bare
scalar. Every metric also ships a dynamic-region variant (``…@dynamic``) that
scores only the moving pixels (mask from :mod:`worldproof.masking`), so a static
background can't inflate the score.

PSNR and SSIM are implemented in numpy so they always run on a laptop with the
core install (no extra, no GPU). LPIPS — which needs a pretrained network — is
the optional, lazily-imported variant and lives elsewhere.

These are per-rollout metrics. FVD (distribution-level, action-blind) and the
action-recoverability probe (a cross-rollout probe that overfits on one rollout)
are report/aggregation-layer measures, built with the report card, not here.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch

from worldproof.core import Rollout
from worldproof.masking import dynamic_region_mask
from worldproof.metrics.base import Metric
from worldproof.metrics.registry import register

__all__ = ["FidelityMetric", "PSNR", "PSNRDynamic", "SSIM", "SSIMDynamic"]

_UINT8_RANGE = 255.0
_MSE_FLOOR = 1e-10  # keep PSNR finite for a perfect prediction


class FidelityMetric(Metric):
    """Base for per-frame fidelity metrics (needs a ground-truth reference).

    Subclasses implement :meth:`_frame_score`. Setting ``dynamic = True`` makes
    the base restrict scoring to the moving-region mask derived from the
    reference; the subclass just honours the mask it is handed.
    """

    modality = "pixels"
    needs_reference = True
    dynamic: bool = False

    @abstractmethod
    def _frame_score(
        self, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None
    ) -> float:
        """Score one predicted frame ``(H, W, C)`` against its target.

        ``mask`` is a ``(H, W)`` boolean moving-region mask or ``None`` (whole
        frame). Implementations restrict to ``mask`` when it is not ``None`` and
        has any True pixels.
        """

    def _curve(self, rollout: Rollout, device: torch.device) -> np.ndarray:
        target = rollout.ground_truth
        assert target is not None  # runner gates needs_reference
        masks = dynamic_region_mask(target) if self.dynamic else None
        horizon = rollout.horizon
        curve = np.empty((rollout.n_samples, horizon), dtype=np.float64)
        for sample_idx, prediction in enumerate(rollout.predictions):
            for step in range(horizon):
                mask = masks[step] if masks is not None else None
                curve[sample_idx, step] = self._frame_score(
                    prediction[step], target[step], mask
                )
        return curve


@register
class PSNR(FidelityMetric):
    """Peak signal-to-noise ratio (dB), higher is better."""

    name = "psnr"
    version = "1.0.0"
    higher_is_better = True

    def _frame_score(self, prediction, target, mask):
        squared_error = (prediction.astype(np.float64) - target.astype(np.float64)) ** 2
        if mask is not None and mask.any():
            squared_error = squared_error[mask]  # (n_pixels, C)
        mse = max(float(squared_error.mean()), _MSE_FLOOR)
        return 10.0 * np.log10(_UINT8_RANGE**2 / mse)


@register
class PSNRDynamic(PSNR):
    """PSNR on moving-region pixels only."""

    name = "psnr@dynamic"
    dynamic = True


def _gaussian_window(size: int, sigma: float) -> np.ndarray:
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    line = np.exp(-(coords**2) / (2.0 * sigma**2))
    line /= line.sum()
    return np.outer(line, line)


def _ssim_channel_map(x: np.ndarray, y: np.ndarray, window: np.ndarray) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    xw = sliding_window_view(x, window.shape)
    yw = sliding_window_view(y, window.shape)
    mu_x = (xw * window).sum(axis=(-1, -2))
    mu_y = (yw * window).sum(axis=(-1, -2))
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x2 = (xw**2 * window).sum(axis=(-1, -2)) - mu_x2
    sigma_y2 = (yw**2 * window).sum(axis=(-1, -2)) - mu_y2
    sigma_xy = (xw * yw * window).sum(axis=(-1, -2)) - mu_xy
    c1 = (0.01 * _UINT8_RANGE) ** 2
    c2 = (0.03 * _UINT8_RANGE) ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return numerator / denominator


@register
class SSIM(FidelityMetric):
    """Structural similarity index (Gaussian window), higher is better.

    Standard SSIM (Wang et al. 2004): an 11×11 Gaussian window (sigma 1.5),
    computed per channel and averaged, with the window shrunk for frames smaller
    than 11 px.
    """

    name = "ssim"
    version = "1.0.0"
    higher_is_better = True

    def _frame_score(self, prediction, target, mask):
        x = prediction.astype(np.float64)
        y = target.astype(np.float64)
        height, width, channels = x.shape
        win_size = min(11, height, width)
        if win_size % 2 == 0:
            win_size -= 1
        window = _gaussian_window(win_size, 1.5)
        ssim_map = np.mean(
            [_ssim_channel_map(x[..., c], y[..., c], window) for c in range(channels)],
            axis=0,
        )
        if mask is not None and mask.any():
            crop = (win_size - 1) // 2
            map_h, map_w = ssim_map.shape
            mask_crop = mask[crop : crop + map_h, crop : crop + map_w]
            if mask_crop.any():
                return float(ssim_map[mask_crop].mean())
        return float(ssim_map.mean())


@register
class SSIMDynamic(SSIM):
    """SSIM on moving-region pixels only."""

    name = "ssim@dynamic"
    dynamic = True

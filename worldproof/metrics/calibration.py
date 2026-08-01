"""Calibration: does the model's sample variance predict its actual error?

A stochastic world model draws ``n_samples`` futures. Their spread is its
*predicted uncertainty*; how far a sample typically lands from the ground truth
is its *actual error*. A calibrated model's predicted uncertainty matches its
error — high spread where it is actually wrong, low spread where it is right.

Reported as Expected / Maximum Calibration Error (ECE / MCE) per horizon step,
lower is better. Guo et al. 2017 defines ECE/MCE for *classification* (binned
confidence vs accuracy); world-model rollouts are continuous, so we use the
regression-uncertainty analog (Levi et al. 2019): bin every element by its
predicted uncertainty and, in each bin, compare the root-mean variance (RMV,
predicted) to the root-mean-square error (RMSE, observed). Both are at the
*single-sample* scale: RMSE is the RMS distance of a *sample* from the ground
truth (``mean_over_samples (sample - gt)^2``), **not** the error of the sample
mean — comparing the spread (σ) against the error of the mean (σ/√n) would flag
a calibrated model as miscalibrated, worsening as ``n_samples`` grows. ECE is
the population-weighted mean ``|RMV - RMSE|`` gap across bins; MCE is the max
gap. Units match the prediction (intensity for pixels, latent units for latents).

FORMULATION NOTE: this reads the SPEC's "does sample variance predict actual
error?" literally — an uncertainty-vs-error calibration. An alternative reading
is quantile / coverage calibration (Kuleshov et al. 2018): does the ground truth
fall inside the sample distribution at the nominal rate? Say the word to switch.
"""

from __future__ import annotations

import numpy as np
import torch

from worldproof.core import Rollout
from worldproof.metrics.base import Metric
from worldproof.metrics.registry import register

__all__ = ["CalibrationECE", "CalibrationMCE"]


@register
class CalibrationECE(Metric):
    """Expected calibration error of predicted uncertainty vs error, lower better."""

    name = "calibration_ece"
    version = "1.1.0"  # 1.1.0: per-sample RMSE (see module docstring)
    higher_is_better = False
    needs_reference = True
    needs_multisample = True
    n_bins: int = 10
    _reduction = "expected"  # "expected" (weighted mean gap) or "maximum" (max gap)

    def _curve(self, rollout: Rollout, device: torch.device) -> np.ndarray:
        target = rollout.ground_truth
        assert target is not None  # runner gates needs_reference
        target = target.astype(np.float64)
        samples = np.stack(
            [prediction.astype(np.float64) for prediction in rollout.predictions],
            axis=0,
        )  # (n_samples, horizon, *dims)
        variance = samples.var(axis=0)  # predicted uncertainty (spread) per element
        # per-sample squared error (same scale as the spread), NOT error-of-mean
        squared_error = ((samples - target) ** 2).mean(axis=0)

        horizon = rollout.horizon
        curve = np.empty((1, horizon), dtype=np.float64)
        for step in range(horizon):
            curve[0, step] = self._calibration_error(
                variance[step].reshape(-1), squared_error[step].reshape(-1)
            )
        return curve

    def _calibration_error(
        self, variance: np.ndarray, squared_error: np.ndarray
    ) -> float:
        n_elements = variance.size
        n_bins = min(self.n_bins, n_elements)
        order = np.argsort(variance)  # bin by predicted uncertainty (variance)
        variance = variance[order]
        squared_error = squared_error[order]

        gaps = np.empty(n_bins)
        weights = np.empty(n_bins)
        for bin_idx, element_idx in enumerate(
            np.array_split(np.arange(n_elements), n_bins)
        ):
            root_mean_variance = np.sqrt(variance[element_idx].mean())
            root_mean_sq_error = np.sqrt(squared_error[element_idx].mean())
            gaps[bin_idx] = abs(root_mean_variance - root_mean_sq_error)
            weights[bin_idx] = element_idx.size / n_elements

        if self._reduction == "maximum":
            return float(gaps.max())
        return float((gaps * weights).sum())


@register
class CalibrationMCE(CalibrationECE):
    """Maximum calibration error (worst bin), lower is better."""

    name = "calibration_mce"
    _reduction = "maximum"

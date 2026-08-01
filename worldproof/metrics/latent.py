"""Tier-1 latent-track metric: latent prediction error, as a horizon curve.

For a latent (JEPA / DINO-WM) world model, evaluate in the space it predicts in
(SPEC §4 latent track). Latent prediction error is the mean squared error
between each predicted latent frame and the ground-truth latent frame at the
same step — a per-sample ``(n_samples, horizon)`` curve, lower is better.

The action-recoverability probe — the *primary* latent diagnostic (SPEC §4) —
is a probe fit across many rollouts' transitions; on a single rollout it
overfits and is meaningless. It is therefore a report/aggregation-layer metric,
built with the report card, not a per-rollout metric here.
"""

from __future__ import annotations

import numpy as np
import torch

from worldproof.core import Rollout
from worldproof.metrics.base import Metric
from worldproof.metrics.registry import register

__all__ = ["LatentPredictionError"]


@register
class LatentPredictionError(Metric):
    """Per-step mean squared error in latent space, lower is better."""

    name = "latent_prediction_error"
    version = "1.0.0"
    higher_is_better = False
    modality = "latents"
    needs_reference = True

    def _curve(self, rollout: Rollout, device: torch.device) -> np.ndarray:
        target = rollout.ground_truth
        assert target is not None  # runner gates needs_reference
        target = target.astype(np.float64).reshape(rollout.horizon, -1)
        curve = np.empty((rollout.n_samples, rollout.horizon), dtype=np.float64)
        for sample_idx, prediction in enumerate(rollout.predictions):
            flat = prediction.astype(np.float64).reshape(rollout.horizon, -1)
            curve[sample_idx] = ((flat - target) ** 2).mean(axis=1)
        return curve

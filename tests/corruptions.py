"""Corruption helpers for metric validation (SPEC §4).

Spatial corruption (noise) plus temporal / dynamics corruptions (frame swap,
reversal) so a metric can prove it tracks *dynamics*, not just frame quality
(Unterthiner et al. 2018). Not a ``test_*`` module, so pytest does not collect
it directly.
"""

from __future__ import annotations

import numpy as np

from worldproof.core import Rollout


def add_spatial_noise(frames: np.ndarray, *, sigma: float = 30.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    noise = np.round(rng.normal(0, sigma, frames.shape)).astype(np.int16)
    return np.clip(frames.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def swap_adjacent_frames(frames: np.ndarray, index: int = 0) -> np.ndarray:
    out = frames.copy()
    out[[index, index + 1]] = out[[index + 1, index]]
    return out


def add_latent_noise(latents: np.ndarray, *, sigma: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    noisy = latents.astype(np.float64) + rng.normal(0, sigma, latents.shape)
    return noisy.astype(latents.dtype)


def with_predictions(rollout: Rollout, predictions) -> Rollout:
    """A copy of ``rollout`` with its predictions replaced."""
    return Rollout(
        rollout.modality,
        rollout.context,
        rollout.actions,
        tuple(predictions),
        rollout.metadata,
        rollout.ground_truth,
    )


def assert_ranks_real_over_naive_over_broken(metric, real, naive, broken) -> None:
    """The built-in ranking test: real model > naive baseline > broken baseline."""
    scores = [metric.compute(r).summary for r in (real, naive, broken)]
    if metric.higher_is_better:
        assert scores[0] > scores[1] > scores[2], scores
    else:
        assert scores[0] < scores[1] < scores[2], scores

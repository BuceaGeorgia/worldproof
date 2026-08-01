"""Tests for the Tier-1 latent-track metric (latent prediction error)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from corruptions import add_latent_noise, assert_ranks_real_over_naive_over_broken

from worldproof.core import Rollout, RolloutMetadata
from worldproof.metrics import (
    REGISTRY,
    Capabilities,
    LatentPredictionError,
    MetricRunner,
)

CPU = Capabilities(device=torch.device("cpu"))
T = 4
D = 16


def _latent_rollout(prediction, ground_truth):
    return Rollout(
        "latents",
        np.zeros((3, D), np.float32),
        np.zeros((T, 2), np.float32),
        (prediction,),
        RolloutMetadata(fps=10.0, model_id="m"),
        ground_truth,
    )


def _latent_triple():
    rng = np.random.default_rng(1)
    gt = rng.standard_normal((T, D)).astype(np.float32)
    static = np.repeat(rng.standard_normal((1, D)).astype(np.float32), T, axis=0)
    real = (gt + 0.01 * rng.standard_normal(gt.shape)).astype(np.float32)
    broken = (static + 3.0 * rng.standard_normal((T, D))).astype(np.float32)
    mk = lambda pred: _latent_rollout(pred, gt)  # noqa: E731
    return mk(real), mk(static), mk(broken)


def test_registered():
    assert "latent_prediction_error" in REGISTRY


def test_perfect_prediction_has_zero_error():
    gt = np.random.default_rng(0).standard_normal((T, D)).astype(np.float32)
    result = LatentPredictionError().compute(_latent_rollout(gt.copy(), gt))
    assert result.curve.shape == (1, T)
    assert result.higher_is_better is False
    assert result.summary == pytest.approx(0.0, abs=1e-6)


def test_responds_to_latent_noise():
    gt = np.random.default_rng(0).standard_normal((T, D)).astype(np.float32)
    perfect = _latent_rollout(gt.copy(), gt)
    noisy = _latent_rollout(add_latent_noise(gt, sigma=1.0), gt)
    metric = LatentPredictionError()
    assert metric.compute(noisy).summary > metric.compute(perfect).summary


def test_ranking_real_over_naive_over_broken():
    assert_ranks_real_over_naive_over_broken(LatentPredictionError(), *_latent_triple())


def test_runner_skips_latent_metric_on_pixel_rollout():
    pixel = Rollout(
        "pixels",
        np.zeros((3, 8, 8, 3), np.uint8),
        np.zeros((T, 2), np.float32),
        (np.zeros((T, 8, 8, 3), np.uint8),),
        RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8)),
        np.zeros((T, 8, 8, 3), np.uint8),
    )
    run = MetricRunner([LatentPredictionError()], capabilities=CPU).run(pixel)
    assert not run.results
    assert "applies to 'latents'" in run.skips[0].reason

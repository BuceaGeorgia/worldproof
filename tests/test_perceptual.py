"""Tests for the optional LPIPS fidelity metric.

The registration and graceful-skip-without-extra tests always run; the actual
LPIPS computation tests require the ``fidelity`` extra and skip otherwise.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from worldproof.core import Rollout, RolloutMetadata
from worldproof.metrics import (
    LPIPS,
    REGISTRY,
    Capabilities,
    LPIPSDynamic,
    MetricRunner,
)

CPU = Capabilities(device=torch.device("cpu"))
H = W = 64
T = 3


def _rollout(prediction, ground_truth):
    return Rollout(
        "pixels",
        np.zeros((3, H, W, 3), np.uint8),
        np.zeros((T, 2), np.float32),
        (prediction,),
        RolloutMetadata(fps=10.0, model_id="m", resolution=(H, W)),
        ground_truth,
    )


def _gt(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (T, H, W, 3), np.uint8)


def _noise(frames, sigma, seed):
    rng = np.random.default_rng(seed)
    noisy = frames.astype(np.int16) + np.round(rng.normal(0, sigma, frames.shape))
    return np.clip(noisy, 0, 255).astype(np.uint8)


def test_lpips_registered_without_extra():
    assert "lpips" in REGISTRY
    assert "lpips@dynamic" in REGISTRY


def test_lpips_skips_gracefully_without_the_extra(monkeypatch):
    # Simulate the package being absent regardless of the environment.
    monkeypatch.setitem(sys.modules, "lpips", None)
    gt = _gt()
    run = MetricRunner([LPIPS()], capabilities=CPU).run(_rollout(gt.copy(), gt))
    assert not run.results
    assert "worldproof[fidelity]" in run.skips[0].reason


def test_lpips_computes_distance():
    pytest.importorskip("lpips")
    gt = _gt()
    perfect = LPIPS().compute(_rollout(gt.copy(), gt), device=torch.device("cpu"))
    noisy = LPIPS().compute(_rollout(_noise(gt, 40, 1), gt), device=torch.device("cpu"))
    assert perfect.higher_is_better is False
    assert perfect.curve.shape == (1, T)
    assert perfect.summary == pytest.approx(0.0, abs=1e-3)
    assert noisy.summary > perfect.summary


def test_lpips_ranking_by_prediction_quality():
    pytest.importorskip("lpips")
    gt = _gt()
    metric = LPIPS()
    device = torch.device("cpu")
    real = metric.compute(_rollout(_noise(gt, 5, 1), gt), device=device).summary
    naive = metric.compute(_rollout(_noise(gt, 40, 2), gt), device=device).summary
    broken = metric.compute(_rollout(_noise(gt, 120, 3), gt), device=device).summary
    assert real < naive < broken  # LPIPS is a distance: lower = better


def test_lpips_dynamic_variant_runs():
    pytest.importorskip("lpips")
    gt = _gt()
    result = LPIPSDynamic().compute(
        _rollout(_noise(gt, 40, 1), gt), device=torch.device("cpu")
    )
    assert result.curve.shape == (1, T)

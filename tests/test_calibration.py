"""Tests for the calibration signature metric (ECE / MCE).

Calibration compares the ensemble spread against the *per-sample* RMS error (both
at the single-sample scale). A genuinely calibrated ensemble (samples drawn
around the truth with spread == typical error) scores ~0 and converges toward 0
as ``n_samples`` grows — the regression that caught the earlier error-of-the-mean
bug.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from corruptions import assert_ranks_real_over_naive_over_broken

from worldproof.core import Rollout, RolloutMetadata
from worldproof.metrics import (
    REGISTRY,
    CalibrationECE,
    CalibrationMCE,
    Capabilities,
    MetricRunner,
)

CPU = Capabilities(device=torch.device("cpu"))
T = 3
D = 3000
_META = RolloutMetadata(fps=10.0, model_id="m")


def _rollout(bias_scale, spread, *, n=16, seed=0):
    """Ensemble: samples = gt + bias_scale*bias + spread*noise, spread is the
    predicted uncertainty; a fixed per-element ``bias`` is the systematic error."""
    rng = np.random.default_rng(seed)
    gt = rng.standard_normal((T, D)).astype(np.float32)
    bias = rng.standard_normal((T, D)).astype(np.float32)
    samples = tuple(
        (gt + bias_scale * bias + spread * rng.standard_normal((T, D))).astype(
            np.float32
        )
        for _ in range(n)
    )
    return Rollout(
        "latents",
        np.zeros((3, D), np.float32),
        np.zeros((T, 2), np.float32),
        samples,
        _META,
        gt,
    )


def _calibrated(**kw):  # no bias, spread == typical error
    return _rollout(0.0, 1.0, **kw)


def test_registered():
    assert "calibration_ece" in REGISTRY
    assert "calibration_mce" in REGISTRY


def test_shape_and_direction():
    result = CalibrationECE().compute(_calibrated())
    assert result.curve.shape == (1, T)
    assert result.higher_is_better is False


def test_calibrated_ensemble_scores_near_zero():
    # spread matches per-sample error -> low ECE (and it converges as n grows)
    assert CalibrationECE().compute(_calibrated(n=16)).summary < 0.2


def test_ece_converges_toward_zero_with_more_samples():
    metric = CalibrationECE()
    coarse = metric.compute(_calibrated(n=4)).summary
    fine = metric.compute(_calibrated(n=32)).summary
    assert fine < coarse  # more samples -> better calibration estimate, not worse


@pytest.mark.parametrize("metric", [CalibrationECE(), CalibrationMCE()])
def test_responds_to_overconfidence(metric):
    calibrated = _calibrated()
    overconfident = _rollout(2.0, 0.1)  # big systematic error, tiny claimed spread
    assert metric.compute(overconfident).summary > metric.compute(calibrated).summary


@pytest.mark.parametrize("metric", [CalibrationECE(), CalibrationMCE()])
def test_ranking_real_over_naive_over_broken(metric):
    real = _rollout(0.0, 1.0)  # calibrated
    naive = _rollout(1.0, 0.4)  # moderately overconfident
    broken = _rollout(3.0, 0.1)  # wildly overconfident
    assert_ranks_real_over_naive_over_broken(metric, real, naive, broken)


# --------------------------------------------------------------------------- #
# Requirement gating
# --------------------------------------------------------------------------- #


def test_runner_skips_single_sample_rollout():
    gt = np.zeros((T, D), np.float32)
    single = Rollout(
        "latents",
        np.zeros((3, D), np.float32),
        np.zeros((T, 2), np.float32),
        (gt.copy(),),  # n_samples == 1
        _META,
        gt,
    )
    run = MetricRunner([CalibrationECE()], capabilities=CPU).run(single)
    assert not run.results
    assert "at least 2 samples" in run.skips[0].reason


def test_runner_skips_without_ground_truth():
    no_gt = Rollout(
        "latents",
        np.zeros((3, D), np.float32),
        np.zeros((T, 2), np.float32),
        (np.zeros((T, D), np.float32), np.zeros((T, D), np.float32)),
        _META,
    )
    run = MetricRunner([CalibrationECE()], capabilities=CPU).run(no_gt)
    assert not run.results
    assert "ground truth" in run.skips[0].reason


def test_requirements_declares_reference_and_multisample():
    assert CalibrationECE.requirements() == {
        "needs_reference": True,
        "needs_multisample": True,
    }

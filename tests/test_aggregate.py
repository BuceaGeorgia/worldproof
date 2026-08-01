"""Tests for the report-layer (set-level) measures: FVD + action-recoverability.

Covers the Fréchet math in isolation (pure numpy, no weights), the FVD measure
through a small deterministic reference extractor (the plug-point), the
action-recoverability probe's corruption + ranking behavior, and the
skip-and-report paths — including the default (no-extractor) FVD skip that a core
`evaluate` produces.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from worldproof.core import Rollout, RolloutMetadata
from worldproof.metrics.aggregate import (
    ActionRecoverabilityProbe,
    FrechetVideoDistance,
    MeasureSkipped,
    frechet_distance,
    run_aggregates,
)

# --------------------------------------------------------------------------- #
# Fréchet distance — pure numpy, exact-ish checks with controlled distributions
# --------------------------------------------------------------------------- #


def test_frechet_distance_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(64, 8))
    assert frechet_distance(feats, feats) < 1e-6


def test_frechet_distance_equals_squared_mean_shift_under_equal_cov():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(128, 6))
    shift = np.array([1.0, -2.0, 0.5, 0.0, 3.0, -1.0])
    b = a + shift  # identical empirical covariance, mean differs by `shift`
    expected = float(shift @ shift)
    assert abs(frechet_distance(a, b) - expected) < 1e-2


def test_frechet_distance_is_symmetric_and_grows_with_separation():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(200, 5))
    near = a + 0.5
    far = a + 5.0
    assert abs(frechet_distance(a, near) - frechet_distance(near, a)) < 1e-6
    assert frechet_distance(a, far) > frechet_distance(a, near)


# --------------------------------------------------------------------------- #
# FVD measure — a small deterministic reference extractor (the plug-point)
# --------------------------------------------------------------------------- #


def _pooled_features(clips):
    """Deterministic 2x2-grid temporal mean+std per channel — no weights, no net."""
    out = []
    for clip in clips:
        arr = np.asarray(clip, dtype=np.float64)  # (T, H, W, C)
        t, h, w, c = arr.shape
        gh, gw = h // 2, w // 2
        blocks = arr[:, : gh * 2, : gw * 2].reshape(t, 2, gh, 2, gw, c)
        grid = blocks.mean(axis=(2, 4))  # (T, 2, 2, C)
        out.append(np.concatenate([grid.mean(0).ravel(), grid.std(0).ravel()]))
    return np.asarray(out)


def _pixel_rollout(seed, *, noise, horizon=4, size=16):
    rng = np.random.default_rng(seed)
    gt = rng.integers(0, 256, (horizon, size, size, 3), np.uint8)
    ctx = rng.integers(0, 256, (2, size, size, 3), np.uint8)
    pred = np.clip(
        gt.astype(int) + rng.integers(-noise, noise + 1, gt.shape), 0, 255
    ).astype(np.uint8)
    meta = RolloutMetadata(fps=10.0, model_id="m", resolution=(size, size))
    return Rollout("pixels", ctx, np.zeros((horizon, 2), np.float32), (pred,), meta, gt)


def test_fvd_rises_as_predictions_degrade():
    faithful = [_pixel_rollout(i, noise=2) for i in range(10)]
    broken = [_pixel_rollout(i, noise=120) for i in range(10)]
    measure = FrechetVideoDistance(_pooled_features)
    faithful_fvd = measure.compute(faithful).summary
    broken_fvd = measure.compute(broken).summary
    assert faithful_fvd >= 0.0
    assert broken_fvd > faithful_fvd


def test_fvd_reports_provenance():
    rollouts = [_pixel_rollout(i, noise=10) for i in range(6)]
    result = FrechetVideoDistance(_pooled_features).compute(rollouts)
    assert result.name == "fvd" and not result.higher_is_better
    assert result.extra["n_real_clips"] == 6
    assert result.extra["n_generated_clips"] == 6  # one sample each


def test_fvd_skips_without_extractor():
    rollouts = [_pixel_rollout(i, noise=10) for i in range(6)]
    try:
        FrechetVideoDistance(None).compute(rollouts)
    except MeasureSkipped as skip:
        assert "worldproof[fvd]" in str(skip)
    else:  # pragma: no cover
        raise AssertionError("expected a MeasureSkipped")


# --------------------------------------------------------------------------- #
# Action-recoverability probe — corruption + ranking + skips
# --------------------------------------------------------------------------- #


def _latent_rollout(seed, *, horizon=5, dim=4, recoverable=True):
    """Latent rollout. If recoverable, z_t = z_{t-1} + action_t (invertible)."""
    rng = np.random.default_rng(seed)
    actions = rng.normal(size=(horizon, dim)).astype(np.float32)
    context = rng.normal(size=(3, dim)).astype(np.float32)
    if recoverable:
        latents = context[-1].astype(np.float64) + np.cumsum(actions, axis=0)
    else:
        latents = rng.normal(size=(horizon, dim))  # unrelated to actions
    pred = latents.astype(np.float32)
    meta = RolloutMetadata(fps=10.0, model_id="m", seed=seed)
    return Rollout("latents", context, actions, (pred,), meta)


def test_probe_recovers_actions_from_structured_latents():
    rollouts = [_latent_rollout(i, recoverable=True) for i in range(12)]
    result = ActionRecoverabilityProbe().compute(rollouts, seed=0)
    assert result.name == "action_recoverability" and result.higher_is_better
    assert result.summary > 0.7  # invertible dynamics -> high cross-validated R^2
    assert len(result.curve) == 5


def test_probe_ranks_structured_above_noise():
    structured = [_latent_rollout(i, recoverable=True) for i in range(12)]
    noise = [_latent_rollout(i, recoverable=False) for i in range(12)]
    probe = ActionRecoverabilityProbe()
    structured_r2 = probe.compute(structured, seed=0).summary
    noise_r2 = probe.compute(noise, seed=0).summary
    assert structured_r2 > noise_r2 + 0.4
    assert noise_r2 < 0.3  # random latents don't encode the actions


def test_probe_corruption_shuffling_actions_collapses_r2():
    rollouts = [_latent_rollout(i, recoverable=True) for i in range(12)]
    clean = ActionRecoverabilityProbe().compute(rollouts, seed=0).summary
    # break the latent<->action correspondence by permuting action arrays
    shuffled_actions = [rollouts[j].actions for j in np.roll(np.arange(12), 3)]
    corrupted = [
        dataclasses.replace(r, actions=a)
        for r, a in zip(rollouts, shuffled_actions, strict=True)
    ]
    corrupted_r2 = ActionRecoverabilityProbe().compute(corrupted, seed=0).summary
    assert clean > 0.7
    assert corrupted_r2 < clean - 0.4


def test_probe_is_deterministic():
    rollouts = [_latent_rollout(i, recoverable=True) for i in range(12)]
    probe = ActionRecoverabilityProbe()
    assert (
        probe.compute(rollouts, seed=0).summary
        == probe.compute(rollouts, seed=0).summary
    )


def test_probe_skips_on_too_few_latent_rollouts():
    rollouts = [_latent_rollout(i) for i in range(3)]
    try:
        ActionRecoverabilityProbe().compute(rollouts, seed=0)
    except MeasureSkipped as skip:
        assert ">= 4" in str(skip)
    else:  # pragma: no cover
        raise AssertionError("expected a MeasureSkipped")


def test_probe_skips_on_pixel_only_set():
    rollouts = [_pixel_rollout(i, noise=10) for i in range(6)]
    try:
        ActionRecoverabilityProbe().compute(rollouts, seed=0)
    except MeasureSkipped as skip:
        assert "no latent rollouts" in str(skip)
    else:  # pragma: no cover
        raise AssertionError("expected a MeasureSkipped")


# --------------------------------------------------------------------------- #
# run_aggregates — skip-and-report shape
# --------------------------------------------------------------------------- #


def test_run_aggregates_reports_skips_without_crashing():
    rollouts = [_pixel_rollout(i, noise=10) for i in range(6)]
    results, skips = run_aggregates(
        rollouts, [ActionRecoverabilityProbe(), FrechetVideoDistance(None)], seed=0
    )
    assert results == []
    names = {s.metric for s in skips}
    assert names == {"action_recoverability", "fvd"}


def test_run_aggregates_mixed_set_runs_fvd_and_probe():
    pixels = [_pixel_rollout(i, noise=10) for i in range(6)]
    latents = [_latent_rollout(i, recoverable=True) for i in range(6)]
    results, skips = run_aggregates(
        pixels + latents,
        [ActionRecoverabilityProbe(), FrechetVideoDistance(_pooled_features)],
        seed=0,
    )
    names = {r.name for r in results}
    assert names == {"action_recoverability", "fvd"}
    assert skips == []


def test_probe_rejects_degenerate_n_folds():
    with pytest.raises(ValueError, match="n_folds"):
        ActionRecoverabilityProbe(n_folds=1)


def test_capabilities_carry_fvd_extractor_through_evaluate():
    import torch

    from worldproof.metrics import PSNR, Capabilities
    from worldproof.report import evaluate

    rollouts = [_pixel_rollout(i, noise=10) for i in range(6)]
    caps = Capabilities(device=torch.device("cpu"), fvd_extractor=_pooled_features)
    report, _ = evaluate(rollouts, metrics=[PSNR()], capabilities=caps)
    assert any(a.name == "fvd" for a in report.aggregates)
    # and without the capability, FVD is a reported skip instead
    report2, _ = evaluate(
        rollouts, metrics=[PSNR()], capabilities=Capabilities(torch.device("cpu"))
    )
    assert "fvd" in report2.skips

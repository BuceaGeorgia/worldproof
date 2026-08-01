"""Tests for the report / aggregation layer."""

from __future__ import annotations

import json

import numpy as np
import torch

from worldproof.core import Rollout, RolloutMetadata
from worldproof.metrics import PSNR, SSIM, Capabilities, LatentPredictionError
from worldproof.report import (
    bootstrap_ci,
    evaluate,
    interquartile_mean,
    performance_profile,
    report_html,
    report_json,
)

CPU = Capabilities(device=torch.device("cpu"))
H = W = 24
T = 4


def _rollout(seed, *, latency=None):
    rng = np.random.default_rng(seed)
    gt = rng.integers(0, 256, (T, H, W, 3), np.uint8)
    ctx = rng.integers(0, 256, (3, H, W, 3), np.uint8)
    preds = tuple(
        np.clip(gt.astype(int) + rng.integers(-15, 16, gt.shape), 0, 255).astype(
            np.uint8
        )
        for _ in range(2)
    )
    metadata = RolloutMetadata(
        fps=10.0, model_id="m", resolution=(H, W), inference_latency_s=latency
    )
    return Rollout("pixels", ctx, np.zeros((T, 2), np.float32), preds, metadata, gt)


# --------------------------------------------------------------------------- #
# Robust aggregation primitives
# --------------------------------------------------------------------------- #


def test_interquartile_mean():
    assert interquartile_mean(list(range(100))) == 49.5  # middle 50%
    assert interquartile_mean([1, 2, 3]) == 2.0  # < 4 values -> plain mean


def test_bootstrap_ci_is_deterministic_and_brackets_iqm():
    values = list(range(50))
    assert bootstrap_ci(values, seed=0) == bootstrap_ci(values, seed=0)
    low, high = bootstrap_ci(values, seed=0)
    assert low <= interquartile_mean(values) <= high


def test_performance_profile():
    profile = performance_profile([1, 2, 3, 4], [0, 2, 5], higher_is_better=True)
    assert profile == [1.0, 0.75, 0.0]


# --------------------------------------------------------------------------- #
# build_report / evaluate
# --------------------------------------------------------------------------- #


def test_evaluate_aggregates_metrics_and_latency():
    rollouts = [_rollout(i, latency=0.1 * i + 0.05) for i in range(4)]
    report, _ = evaluate(rollouts, metrics=[PSNR(), SSIM()], capabilities=CPU, seed=0)
    assert [m.name for m in report.metrics] == ["psnr", "ssim"]  # sorted
    psnr = report.metrics[0]
    assert psnr.higher_is_better
    assert psnr.ci_low <= psnr.iqm <= psnr.ci_high
    assert len(psnr.mean_curve) == T
    assert report.latency is not None and report.latency["n"] == 4.0
    assert "Evaluated 4 rollout(s)" in report.verdict


def test_evaluate_records_skips():
    report, _ = evaluate(
        [_rollout(0)], metrics=[PSNR(), LatentPredictionError()], capabilities=CPU
    )
    assert "latent_prediction_error" in report.skips  # latent metric on pixels
    assert report.skips["latent_prediction_error"]["count"] == 1


def test_verdict_is_ascii_safe():
    # the verdict is printed to stdout, which may not be UTF-8 (Windows/LANG=C)
    report, _ = evaluate(
        [_rollout(0)], metrics=[PSNR(), LatentPredictionError()], capabilities=CPU
    )
    assert "Skipped" in report.verdict
    report.verdict.encode("ascii")  # must not raise


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def test_report_json_is_serializable():
    report, _ = evaluate([_rollout(i) for i in range(3)], metrics=[PSNR()])
    blob = report_json(report)
    json.dumps(blob)  # must not raise
    assert blob["n_rollouts"] == 3
    assert blob["metrics"][0]["name"] == "psnr"
    assert "performance_profile" in blob["metrics"][0]


def test_report_html_is_self_contained():
    rollouts = [_rollout(i) for i in range(3)]
    report, run_report = evaluate(rollouts, metrics=[PSNR()], capabilities=CPU)
    page = report_html(report, rollouts, run_report)
    assert page.startswith("<!doctype html>")
    assert "<svg" in page  # inline horizon curve
    assert "data:image/gif" in page  # embedded worst-N clip
    assert "http://" not in page and "https://" not in page  # no external resources


def test_report_html_without_rollouts_omits_clips():
    report, _ = evaluate([_rollout(0)], metrics=[PSNR()], capabilities=CPU)
    page = report_html(report)
    assert page.startswith("<!doctype html>")
    assert "data:image/gif" not in page


def _latent_rollout(seed, *, dim=8, horizon=4):
    rng = np.random.default_rng(seed)
    gt = rng.normal(size=(horizon, dim)).astype(np.float32)
    ctx = rng.normal(size=(3, dim)).astype(np.float32)
    pred = (gt + rng.normal(scale=0.1, size=gt.shape)).astype(np.float32)
    metadata = RolloutMetadata(fps=10.0, model_id="m", seed=seed)
    return Rollout(
        "latents", ctx, np.zeros((horizon, 2), np.float32), (pred,), metadata, gt
    )


def test_worst_clips_survive_mixed_modality_without_psnr():
    # The ranking metric must be one the *pixel* rollouts actually have. With no
    # PSNR configured and a latent metric sorting first alphabetically, clips
    # must still be selected (from SSIM), not silently dropped.
    rollouts = [_rollout(i) for i in range(3)] + [_latent_rollout(i) for i in range(3)]
    report, run_report = evaluate(
        rollouts, metrics=[SSIM(), LatentPredictionError()], capabilities=CPU
    )
    page = report_html(report, rollouts, run_report)
    assert "data:image/gif" in page  # clips selected via ssim on pixel rollouts

"""Tests for Tier-1 pixel-fidelity metrics (PSNR, SSIM, and @dynamic variants)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from corruptions import (
    add_spatial_noise,
    assert_ranks_real_over_naive_over_broken,
    swap_adjacent_frames,
    with_predictions,
)

from worldproof.core import Rollout, RolloutMetadata
from worldproof.metrics import (
    PSNR,
    REGISTRY,
    SSIM,
    Capabilities,
    MetricRunner,
    PSNRDynamic,
    SSIMDynamic,
)

CPU = Capabilities(device=torch.device("cpu"))
H = W = 32
T = 4


def _pixel_rollout(prediction, ground_truth, context=None):
    if context is None:
        context = np.zeros((3, H, W, 3), np.uint8)
    metadata = RolloutMetadata(fps=10.0, model_id="m", resolution=(H, W))
    return Rollout(
        "pixels",
        context,
        np.zeros((T, 2), np.float32),
        (prediction,),
        metadata,
        ground_truth,
    )


def _random_gt(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (T, H, W, 3), np.uint8)


def _pixel_triple():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 256, (T, H, W, 3), np.uint8)
    context = rng.integers(0, 256, (3, H, W, 3), np.uint8)
    static = np.repeat(context[-1:], T, axis=0)
    real = np.clip(gt.astype(int) + rng.integers(-2, 3, gt.shape), 0, 255).astype(
        np.uint8
    )
    broken = add_spatial_noise(static, sigma=70)
    mk = lambda pred: _pixel_rollout(pred, gt, context)  # noqa: E731
    return mk(real), mk(static), mk(broken)


# --------------------------------------------------------------------------- #
# Registration & basic properties
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["psnr", "ssim", "psnr@dynamic", "ssim@dynamic"])
def test_registered(name):
    assert name in REGISTRY


def test_perfect_prediction_scores_are_ideal():
    gt = _random_gt()
    rollout = _pixel_rollout(gt.copy(), gt)
    assert PSNR().compute(rollout).summary > 100.0  # near-infinite, but finite
    assert SSIM().compute(rollout).summary == pytest.approx(1.0, abs=1e-9)


def test_curve_shape_is_per_sample_per_step():
    gt = _random_gt()
    rollout = _pixel_rollout(gt.copy(), gt)
    curve = PSNR().compute(rollout).curve
    assert curve.shape == (1, T)


# --------------------------------------------------------------------------- #
# Corruption response (SPEC §4: each metric responds to its corruption)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", [PSNR(), SSIM()])
def test_responds_to_spatial_noise(metric):
    gt = _random_gt()
    perfect = _pixel_rollout(gt.copy(), gt)
    noisy = _pixel_rollout(add_spatial_noise(gt, sigma=30), gt)
    assert metric.compute(noisy).summary < metric.compute(perfect).summary


@pytest.mark.parametrize("metric", [PSNR(), SSIM()])
def test_responds_to_temporal_frame_swap(metric):
    # A per-step metric must notice mis-ordered frames — dynamics, not just quality.
    gt = _random_gt()
    perfect = _pixel_rollout(gt.copy(), gt)
    swapped = with_predictions(perfect, [swap_adjacent_frames(gt, 0)])
    assert metric.compute(swapped).summary < metric.compute(perfect).summary


# --------------------------------------------------------------------------- #
# Ranking test (real > naive > broken)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", [PSNR(), SSIM(), PSNRDynamic(), SSIMDynamic()])
def test_ranking_real_over_naive_over_broken(metric):
    assert_ranks_real_over_naive_over_broken(metric, *_pixel_triple())


# --------------------------------------------------------------------------- #
# Dynamic-region variant focuses on motion
# --------------------------------------------------------------------------- #


def test_dynamic_variant_penalizes_wrong_moving_region():
    gt = np.full((T, H, W, 3), 100, np.uint8)
    pred = gt.copy()
    for t in range(T):
        gt[t, 4 + t : 8 + t, 4 + t : 8 + t] = 200  # moving object present in truth
        pred[t, 4 + t : 8 + t, 4 + t : 8 + t] = 100  # model missed it (blank bg)
    rollout = _pixel_rollout(pred, gt, context=np.full((3, H, W, 3), 100, np.uint8))
    full = PSNR().compute(rollout).summary
    dynamic = PSNRDynamic().compute(rollout).summary
    # full frame is mostly-correct background; the dynamic variant sees only the
    # (entirely wrong) moving region and scores far worse.
    assert dynamic < full


# --------------------------------------------------------------------------- #
# Requirement gating via the runner
# --------------------------------------------------------------------------- #


def test_runner_skips_without_ground_truth():
    no_gt = Rollout(
        "pixels",
        np.zeros((3, H, W, 3), np.uint8),
        np.zeros((T, 2), np.float32),
        (np.zeros((T, H, W, 3), np.uint8),),
        RolloutMetadata(fps=10.0, model_id="m", resolution=(H, W)),
    )
    run = MetricRunner([PSNR()], capabilities=CPU).run(no_gt)
    assert not run.results
    assert "ground truth" in run.skips[0].reason


def test_runner_skips_pixel_metric_on_latent_rollout():
    latent = Rollout(
        "latents",
        np.zeros((3, 16), np.float32),
        np.zeros((T, 2), np.float32),
        (np.zeros((T, 16), np.float32),),
        RolloutMetadata(fps=10.0, model_id="m"),
        np.zeros((T, 16), np.float32),
    )
    run = MetricRunner([PSNR()], capabilities=CPU).run(latent)
    assert not run.results
    assert "applies to 'pixels'" in run.skips[0].reason


# --------------------------------------------------------------------------- #
# SSIM correctness vs the reference implementation (skipped without scikit-image)
# --------------------------------------------------------------------------- #


def test_ssim_matches_skimage():
    skimage = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(7)
    base = np.linspace(0, 255, 40)
    image_a = (
        (
            np.tile(base, (40, 1))[..., None].repeat(3, axis=2)
            + rng.normal(0, 15, (40, 40, 3))
        )
        .clip(0, 255)
        .astype(np.uint8)
    )
    image_b = add_spatial_noise(image_a, sigma=25, seed=1)
    mine = SSIM()._frame_score(image_b, image_a, None)
    theirs = skimage.structural_similarity(
        image_a,
        image_b,
        channel_axis=-1,
        data_range=255,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    assert mine == pytest.approx(theirs, abs=1e-6)

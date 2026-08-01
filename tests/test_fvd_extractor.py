"""Tests for the default FVD extractor (torchvision Kinetics r3d_18).

The wiring tests are light and run everywhere: they check that the extractor
constructs without importing torchvision (lazy load) and that
`default_video_extractor` returns None when torchvision is absent. The real
extraction test downloads the r3d_18 weights, so it is marked `slow` (skipped by
default and in core CI) and gated on torchvision, and it skips if the download
is unavailable.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from worldproof.core import Rollout, RolloutMetadata
from worldproof.metrics.aggregate import (
    FrechetVideoDistance,
    KineticsVideoExtractor,
    default_video_extractor,
    frechet_distance,
)

# --------------------------------------------------------------------------- #
# Wiring (no download, runs in core CI)
# --------------------------------------------------------------------------- #


def test_extractor_constructs_without_loading_torchvision():
    # __init__ must not import torch/torchvision or download anything
    extractor = KineticsVideoExtractor()
    assert callable(extractor)


def test_default_extractor_is_none_without_torchvision(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "torchvision" else real(name, *a, **k),
    )
    assert default_video_extractor() is None


def test_default_extractor_present_when_torchvision_installed():
    if importlib.util.find_spec("torchvision") is None:
        pytest.skip("torchvision not installed")
    assert isinstance(default_video_extractor(), KineticsVideoExtractor)


# --------------------------------------------------------------------------- #
# Real extraction (downloads weights -> slow, gated, skip-on-failure)
# --------------------------------------------------------------------------- #


def _clips(seed, *, noise, n=8, t=6, size=32):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (n, t, size, size, 3), np.uint8)
    if noise == 0:
        return [c for c in base]
    return [
        np.clip(
            c.astype(int) + rng.integers(-noise, noise + 1, c.shape), 0, 255
        ).astype(np.uint8)
        for c in base
    ]


@pytest.mark.slow
def test_kinetics_extractor_monotonic_fvd():
    pytest.importorskip("torchvision")
    extractor = KineticsVideoExtractor(device="cpu")
    real = _clips(0, noise=0)
    try:
        f_real = extractor(real)  # first call downloads + loads the model
    except Exception as exc:  # network / weights unavailable
        pytest.skip(f"torchvision video weights unavailable: {exc}")
    assert f_real.shape == (8, 512)

    faithful = extractor(_clips(0, noise=5))
    broken = extractor(_clips(0, noise=120))
    self_fvd = frechet_distance(f_real, f_real)
    faithful_fvd = frechet_distance(f_real, faithful)
    broken_fvd = frechet_distance(f_real, broken)
    assert self_fvd < 1e-6
    assert self_fvd < faithful_fvd < broken_fvd


def _pixel_rollout(seed, *, noise, horizon=6, size=32):
    rng = np.random.default_rng(seed)
    gt = rng.integers(0, 256, (horizon, size, size, 3), np.uint8)
    ctx = rng.integers(0, 256, (2, size, size, 3), np.uint8)
    pred = np.clip(
        gt.astype(int) + rng.integers(-noise, noise + 1, gt.shape), 0, 255
    ).astype(np.uint8)
    meta = RolloutMetadata(fps=10.0, model_id="m", resolution=(size, size))
    return Rollout("pixels", ctx, np.zeros((horizon, 2), np.float32), (pred,), meta, gt)


@pytest.mark.slow
def test_fvd_measure_runs_with_default_extractor():
    pytest.importorskip("torchvision")
    extractor = KineticsVideoExtractor(device="cpu")
    faithful = [_pixel_rollout(i, noise=3) for i in range(8)]
    broken = [_pixel_rollout(i, noise=120) for i in range(8)]
    measure = FrechetVideoDistance(extractor)
    try:
        faithful_fvd = measure.compute(faithful).summary
    except Exception as exc:
        pytest.skip(f"torchvision video weights unavailable: {exc}")
    broken_fvd = measure.compute(broken).summary
    assert faithful_fvd >= 0.0
    assert broken_fvd > faithful_fvd

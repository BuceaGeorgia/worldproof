"""Folder-of-rollouts round-trip tests (lossless format)."""

from __future__ import annotations

import numpy as np
import pytest

from worldproof.adapters.folder import iter_rollouts, load_rollout, save_rollout
from worldproof.core import Rollout, RolloutMetadata


def make_pixel_rollout(*, channels: int = 3, with_ground_truth: bool = True) -> Rollout:
    rng = np.random.default_rng(0)
    horizon, height, width = 4, 6, 5
    context = rng.integers(0, 256, (3, height, width, channels), dtype=np.uint8)
    actions = rng.standard_normal((horizon, 2)).astype(np.float32)
    predictions = tuple(
        rng.integers(0, 256, (horizon, height, width, channels), dtype=np.uint8)
        for _ in range(2)
    )
    ground_truth = (
        rng.integers(0, 256, (horizon, height, width, channels), dtype=np.uint8)
        if with_ground_truth
        else None
    )
    metadata = RolloutMetadata(
        fps=12.5,
        model_id="synthetic",
        seed=7,
        resolution=(height, width),
        camera_id="wrist",
    )
    return Rollout("pixels", context, actions, predictions, metadata, ground_truth)


def make_latent_rollout() -> Rollout:
    rng = np.random.default_rng(1)
    horizon, dim = 4, 12
    context = rng.standard_normal((3, dim)).astype(np.float32)
    actions = rng.standard_normal((horizon, 2)).astype(np.float32)
    predictions = tuple(
        rng.standard_normal((horizon, dim)).astype(np.float32) for _ in range(3)
    )
    metadata = RolloutMetadata(fps=10.0, model_id="jepa", seed=None)
    return Rollout("latents", context, actions, predictions, metadata)


def assert_rollouts_equal(a: Rollout, b: Rollout) -> None:
    assert a.modality == b.modality
    np.testing.assert_array_equal(a.context, b.context)
    np.testing.assert_array_equal(a.actions, b.actions)
    assert a.n_samples == b.n_samples
    for pa, pb in zip(a.predictions, b.predictions, strict=True):
        np.testing.assert_array_equal(pa, pb)
    if a.has_ground_truth:
        np.testing.assert_array_equal(a.ground_truth, b.ground_truth)
    else:
        assert not b.has_ground_truth
    assert a.metadata == b.metadata


def test_pixel_rollout_round_trip_rgb(tmp_path):
    original = make_pixel_rollout(channels=3)
    save_rollout(original, tmp_path / "r0")
    loaded = load_rollout(tmp_path / "r0")
    assert_rollouts_equal(original, loaded)


def test_pixel_rollout_round_trip_grayscale(tmp_path):
    original = make_pixel_rollout(channels=1)
    save_rollout(original, tmp_path / "r0")
    loaded = load_rollout(tmp_path / "r0")
    assert_rollouts_equal(original, loaded)


def test_pixel_rollout_round_trip_without_ground_truth(tmp_path):
    original = make_pixel_rollout(with_ground_truth=False)
    save_rollout(original, tmp_path / "r0")
    loaded = load_rollout(tmp_path / "r0")
    assert_rollouts_equal(original, loaded)


def test_latent_rollout_round_trip(tmp_path):
    original = make_latent_rollout()
    save_rollout(original, tmp_path / "r0")
    loaded = load_rollout(tmp_path / "r0")
    assert_rollouts_equal(original, loaded)
    assert loaded.metadata.resolution is None


def test_load_missing_manifest_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    try:
        load_rollout(tmp_path / "empty")
    except FileNotFoundError as exc:
        assert "manifest.json" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")


def test_iter_rollouts_walks_sibling_directories(tmp_path):
    save_rollout(make_pixel_rollout(), tmp_path / "episode_00")
    save_rollout(make_latent_rollout(), tmp_path / "episode_01")
    (tmp_path / "not_a_rollout").mkdir()  # must be skipped, not crash

    loaded = list(iter_rollouts(tmp_path))
    assert len(loaded) == 2
    assert {r.modality for r in loaded} == {"pixels", "latents"}


def test_iter_rollouts_on_single_rollout_dir(tmp_path):
    save_rollout(make_pixel_rollout(), tmp_path / "r0")
    loaded = list(iter_rollouts(tmp_path / "r0"))
    assert len(loaded) == 1


def test_inference_latency_round_trips(tmp_path):
    original = make_pixel_rollout()
    timed = Rollout(
        original.modality,
        original.context,
        original.actions,
        original.predictions,
        RolloutMetadata(
            fps=original.metadata.fps,
            model_id=original.metadata.model_id,
            seed=original.metadata.seed,
            resolution=original.metadata.resolution,
            camera_id=original.metadata.camera_id,
            inference_latency_s=0.123456,
        ),
        original.ground_truth,
    )
    save_rollout(timed, tmp_path / "r0")
    loaded = load_rollout(tmp_path / "r0")
    assert loaded.metadata.inference_latency_s == 0.123456
    assert loaded.metadata == timed.metadata


def test_context_id_and_is_failure_round_trip(tmp_path):
    original = make_pixel_rollout()
    tagged = Rollout(
        original.modality,
        original.context,
        original.actions,
        original.predictions,
        RolloutMetadata(
            fps=original.metadata.fps,
            model_id=original.metadata.model_id,
            resolution=original.metadata.resolution,
            context_id="episode-42",
            is_failure=True,
        ),
        original.ground_truth,
    )
    save_rollout(tagged, tmp_path / "r0")
    loaded = load_rollout(tmp_path / "r0")
    assert loaded.metadata.context_id == "episode-42"
    assert loaded.metadata.is_failure is True
    assert loaded.metadata == tagged.metadata


def test_lossy_source_defaults_false_and_round_trips(tmp_path):
    original = make_pixel_rollout()
    assert original.metadata.lossy_source is False
    save_rollout(original, tmp_path / "r0")
    assert load_rollout(tmp_path / "r0").metadata.lossy_source is False


def test_format_version_mismatch_rejected(tmp_path):
    import json

    save_rollout(make_pixel_rollout(), tmp_path / "r0")
    manifest_path = tmp_path / "r0" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = "99.0-fromthefuture"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="format_version"):
        load_rollout(tmp_path / "r0")

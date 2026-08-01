"""Tests for the frozen data contracts (SPEC.md §5)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from worldproof.core import MetricResult, Rollout, RolloutMetadata


def make_pixel_rollout(
    *,
    n_samples: int = 2,
    horizon: int = 4,
    height: int = 8,
    width: int = 8,
    channels: int = 3,
    t_context: int = 3,
    with_ground_truth: bool = True,
) -> Rollout:
    context = np.zeros((t_context, height, width, channels), dtype=np.uint8)
    actions = np.zeros((horizon, 2), dtype=np.float32)
    predictions = [
        np.full((horizon, height, width, channels), i, dtype=np.uint8)
        for i in range(n_samples)
    ]
    ground_truth = (
        np.zeros((horizon, height, width, channels), dtype=np.uint8)
        if with_ground_truth
        else None
    )
    metadata = RolloutMetadata(
        fps=10.0, model_id="test-model", seed=0, resolution=(height, width)
    )
    return Rollout("pixels", context, actions, predictions, metadata, ground_truth)


def make_latent_rollout(
    *, n_samples: int = 2, horizon: int = 4, dim: int = 16, t_context: int = 3
) -> Rollout:
    context = np.zeros((t_context, dim), dtype=np.float32)
    actions = np.zeros((horizon, 2), dtype=np.float32)
    predictions = [np.zeros((horizon, dim), dtype=np.float32) for _ in range(n_samples)]
    metadata = RolloutMetadata(fps=10.0, model_id="jepa", seed=1)
    return Rollout("latents", context, actions, predictions, metadata)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_construct_pixel_rollout():
    rollout = make_pixel_rollout(n_samples=3, horizon=5)
    assert rollout.modality == "pixels"
    assert rollout.n_samples == 3
    assert rollout.horizon == 5
    assert rollout.has_ground_truth is True
    assert isinstance(rollout.predictions, tuple)


def test_construct_latent_rollout():
    rollout = make_latent_rollout(n_samples=2, horizon=4, dim=16)
    assert rollout.modality == "latents"
    assert rollout.n_samples == 2
    assert rollout.horizon == 4
    assert rollout.has_ground_truth is False
    assert rollout.metadata.resolution is None


def test_camera_id_defaults_none_and_is_settable():
    default = RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8))
    assert default.camera_id is None
    tagged = RolloutMetadata(
        fps=10.0, model_id="m", resolution=(8, 8), camera_id="wrist_cam"
    )
    assert tagged.camera_id == "wrist_cam"


def test_camera_id_rejects_non_string():
    with pytest.raises(TypeError, match="camera_id must be a string or None"):
        RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8), camera_id=3)


def test_metadata_accepts_numpy_int_resolution():
    metadata = RolloutMetadata(
        fps=10.0, model_id="m", resolution=(np.int64(8), np.int64(8))
    )
    assert metadata.resolution == (8, 8)
    assert all(type(d) is int for d in metadata.resolution)  # normalized to py int


def test_inference_latency_defaults_none_and_is_settable():
    default = RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8))
    assert default.inference_latency_s is None
    timed = RolloutMetadata(
        fps=10.0, model_id="m", resolution=(8, 8), inference_latency_s=2
    )
    assert timed.inference_latency_s == 2.0
    assert isinstance(timed.inference_latency_s, float)  # int coerced to float


def test_inference_latency_rejects_negative():
    with pytest.raises(ValueError, match="inference_latency_s must be a non-negative"):
        RolloutMetadata(
            fps=10.0, model_id="m", resolution=(8, 8), inference_latency_s=-0.1
        )


def test_predictions_normalized_to_tuple_from_list():
    rollout = make_pixel_rollout()
    assert isinstance(rollout.predictions, tuple)


def test_metric_result_derived_values():
    curve = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = MetricResult("psnr", curve, higher_is_better=True)
    assert result.n_samples == 2
    assert result.horizon == 2
    np.testing.assert_allclose(result.mean_curve, [2.0, 3.0])
    np.testing.assert_allclose(result.best_curve, [3.0, 4.0])
    np.testing.assert_allclose(result.spread_curve, [1.0, 1.0])
    assert result.summary == pytest.approx(2.5)


def test_metric_result_best_curve_respects_direction():
    curve = np.array([[1.0, 2.0], [3.0, 4.0]])
    lower_better = MetricResult("lpips", curve, higher_is_better=False)
    np.testing.assert_allclose(lower_better.best_curve, [1.0, 2.0])


def test_metric_result_normalizes_artifacts_and_requirements():
    curve = np.zeros((1, 3))
    result = MetricResult(
        "psnr",
        curve,
        higher_is_better=True,
        requirements_met={"needs_reference": True, "needs_gpu": False},
        artifacts=["clips/worst.mp4", Path("plots/curve.png")],
    )
    assert all(isinstance(a, Path) for a in result.artifacts)
    assert result.requirements_met["needs_reference"] is True
    assert result.requirements_met["needs_gpu"] is False


def test_metric_result_casts_integer_curve_to_float():
    result = MetricResult("count", np.array([[1, 2, 3]]), higher_is_better=True)
    assert np.issubdtype(result.curve.dtype, np.floating)


# --------------------------------------------------------------------------- #
# Validation failures
# --------------------------------------------------------------------------- #


def test_empty_predictions_rejected():
    metadata = RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8))
    with pytest.raises(ValueError, match="at least one prediction"):
        Rollout(
            "pixels",
            np.zeros((3, 8, 8, 3), dtype=np.uint8),
            np.zeros((4, 2), dtype=np.float32),
            (),
            metadata,
        )


def test_mismatched_prediction_shapes_rejected():
    metadata = RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8))
    predictions = [
        np.zeros((4, 8, 8, 3), dtype=np.uint8),
        np.zeros((4, 8, 8, 1), dtype=np.uint8),
    ]
    with pytest.raises(ValueError, match="share one shape"):
        Rollout(
            "pixels",
            np.zeros((3, 8, 8, 3), dtype=np.uint8),
            np.zeros((4, 2), dtype=np.float32),
            predictions,
            metadata,
        )


def test_actions_horizon_mismatch_rejected():
    metadata = RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8))
    with pytest.raises(ValueError, match="one entry per predicted step"):
        Rollout(
            "pixels",
            np.zeros((3, 8, 8, 3), dtype=np.uint8),
            np.zeros((5, 2), dtype=np.float32),  # horizon is 4
            [np.zeros((4, 8, 8, 3), dtype=np.uint8)],
            metadata,
        )


def test_context_frame_shape_mismatch_rejected():
    metadata = RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8))
    with pytest.raises(ValueError, match="context frame dims"):
        Rollout(
            "pixels",
            np.zeros((3, 16, 16, 3), dtype=np.uint8),  # different H/W than preds
            np.zeros((4, 2), dtype=np.float32),
            [np.zeros((4, 8, 8, 3), dtype=np.uint8)],
            metadata,
        )


def test_ground_truth_shape_mismatch_rejected():
    metadata = RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8))
    with pytest.raises(ValueError, match="ground_truth shape"):
        Rollout(
            "pixels",
            np.zeros((3, 8, 8, 3), dtype=np.uint8),
            np.zeros((4, 2), dtype=np.float32),
            [np.zeros((4, 8, 8, 3), dtype=np.uint8)],
            metadata,
            np.zeros((5, 8, 8, 3), dtype=np.uint8),  # wrong horizon
        )


def test_pixel_dtype_must_be_uint8():
    metadata = RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8))
    with pytest.raises(ValueError, match="uint8"):
        Rollout(
            "pixels",
            np.zeros((3, 8, 8, 3), dtype=np.float32),
            np.zeros((4, 2), dtype=np.float32),
            [np.zeros((4, 8, 8, 3), dtype=np.float32)],
            metadata,
        )


def test_pixel_resolution_must_match_frames():
    metadata = RolloutMetadata(fps=10.0, model_id="m", resolution=(16, 16))
    with pytest.raises(ValueError, match="does not match frame"):
        Rollout(
            "pixels",
            np.zeros((3, 8, 8, 3), dtype=np.uint8),
            np.zeros((4, 2), dtype=np.float32),
            [np.zeros((4, 8, 8, 3), dtype=np.uint8)],
            metadata,
        )


def test_pixel_requires_resolution():
    metadata = RolloutMetadata(fps=10.0, model_id="m")  # no resolution
    with pytest.raises(ValueError, match="require metadata.resolution"):
        Rollout(
            "pixels",
            np.zeros((3, 8, 8, 3), dtype=np.uint8),
            np.zeros((4, 2), dtype=np.float32),
            [np.zeros((4, 8, 8, 3), dtype=np.uint8)],
            metadata,
        )


def test_latent_must_be_floating():
    metadata = RolloutMetadata(fps=10.0, model_id="m")
    with pytest.raises(ValueError, match="floating-point"):
        Rollout(
            "latents",
            np.zeros((3, 16), dtype=np.int32),
            np.zeros((4, 2), dtype=np.float32),
            [np.zeros((4, 16), dtype=np.int32)],
            metadata,
        )


def test_unknown_modality_rejected():
    metadata = RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8))
    with pytest.raises(ValueError, match="modality must be"):
        Rollout(
            "voxels",  # type: ignore[arg-type]
            np.zeros((3, 8, 8, 3), dtype=np.uint8),
            np.zeros((4, 2), dtype=np.float32),
            [np.zeros((4, 8, 8, 3), dtype=np.uint8)],
            metadata,
        )


def test_metadata_rejects_nonpositive_fps():
    with pytest.raises(ValueError, match="fps must be positive"):
        RolloutMetadata(fps=0.0, model_id="m")


def test_metric_result_curve_must_be_2d():
    with pytest.raises(ValueError, match="curve must be 2-D"):
        MetricResult("psnr", np.zeros(4), higher_is_better=True)


def test_metric_result_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty string"):
        MetricResult("", np.zeros((1, 3)), higher_is_better=True)


# --------------------------------------------------------------------------- #
# Frozen contract
# --------------------------------------------------------------------------- #


def test_rollout_is_frozen():
    rollout = make_pixel_rollout()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rollout.actions = np.zeros((4, 2), dtype=np.float32)


def test_metadata_is_frozen():
    metadata = RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8))
    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.fps = 30.0


def test_metric_result_is_frozen():
    result = MetricResult("psnr", np.zeros((1, 3)), higher_is_better=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.name = "renamed"

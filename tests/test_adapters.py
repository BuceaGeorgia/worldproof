"""Adapter protocol + baseline behaviour tests."""

from __future__ import annotations

import numpy as np
import pytest
from conformance import assert_adapter_conforms, make_actions, make_context

from worldproof.baselines import ActionBlindBaseline, CopyLastFrameBaseline


@pytest.mark.parametrize("modality", ["pixels", "latents"])
@pytest.mark.parametrize("factory", [CopyLastFrameBaseline, ActionBlindBaseline])
def test_baselines_conform(factory, modality):
    assert_adapter_conforms(factory(modality=modality, seed=0))


# --------------------------------------------------------------------------- #
# Base-class validation
# --------------------------------------------------------------------------- #


def test_adapter_rejects_bad_modality():
    with pytest.raises(ValueError, match="modality must be"):
        CopyLastFrameBaseline(modality="voxels")  # type: ignore[arg-type]


def test_adapter_rejects_empty_model_id():
    with pytest.raises(ValueError, match="model_id must be"):
        CopyLastFrameBaseline(model_id="")


def test_pixel_adapter_rejects_wrong_context_ndim():
    adapter = CopyLastFrameBaseline()
    with pytest.raises(ValueError, match=r"context shaped \(T, H, W, C\)"):
        adapter.predict(np.zeros((3, 8, 8), dtype=np.uint8), make_actions(4), 4)


# --------------------------------------------------------------------------- #
# CopyLastFrameBaseline — naive anchor
# --------------------------------------------------------------------------- #


def test_predict_stamps_inference_latency():
    adapter = CopyLastFrameBaseline()
    rollout = adapter.predict(make_context("pixels"), make_actions(5), 5)
    latency = rollout.metadata.inference_latency_s
    assert isinstance(latency, float)
    assert latency >= 0.0


def test_copy_last_frame_repeats_final_context_frame():
    adapter = CopyLastFrameBaseline()
    context = make_context("pixels")
    rollout = adapter.predict(context, make_actions(5), 5, n_samples=2)
    expected = np.repeat(context[-1][None, ...], 5, axis=0)
    for prediction in rollout.predictions:
        np.testing.assert_array_equal(prediction, expected)


def test_copy_last_frame_has_zero_spread():
    adapter = CopyLastFrameBaseline()
    rollout = adapter.predict(make_context("pixels"), make_actions(5), 5, n_samples=4)
    for prediction in rollout.predictions[1:]:
        np.testing.assert_array_equal(prediction, rollout.predictions[0])


# --------------------------------------------------------------------------- #
# ActionBlindBaseline — broken anchor
# --------------------------------------------------------------------------- #


def test_action_blind_ignores_actions():
    """Same context, different actions, same seed => identical predictions.

    This is the exact property counterfactual divergence must later catch.
    """
    adapter = ActionBlindBaseline(seed=0)
    context = make_context("pixels")
    horizon = 5
    zeros = np.zeros((horizon, 4), dtype=np.float32)
    ones = np.ones((horizon, 4), dtype=np.float32)

    with_zeros = adapter.predict(context, zeros, horizon, n_samples=3, seed=42)
    with_ones = adapter.predict(context, ones, horizon, n_samples=3, seed=42)

    assert not np.array_equal(zeros, ones)  # inputs really do differ
    for a, b in zip(with_zeros.predictions, with_ones.predictions, strict=True):
        np.testing.assert_array_equal(a, b)


def test_action_blind_has_nonzero_spread():
    adapter = ActionBlindBaseline(seed=0)
    rollout = adapter.predict(make_context("pixels"), make_actions(5), 5, n_samples=3)
    assert not np.array_equal(rollout.predictions[0], rollout.predictions[1])


def test_action_blind_latent_outputs_float32():
    adapter = ActionBlindBaseline(modality="latents", seed=0)
    rollout = adapter.predict(make_context("latents"), make_actions(5), 5, n_samples=2)
    for prediction in rollout.predictions:
        assert prediction.dtype == np.float32

"""Tests for the sim oracle: toy reference, gym backend, and pair builders."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from worldproof.baselines import ActionBlindBaseline, CopyLastFrameBaseline
from worldproof.core import CounterfactualPair, VOEPair
from worldproof.metrics import CounterfactualDivergence, FailureFaithfulness
from worldproof.sim import (
    GymSimOracle,
    OracleRollout,
    ToySimOracle,
    make_counterfactual_pair,
    make_voe_pair,
)

_INTO = np.tile([2.0, 2.0], (6, 1)).astype(np.float32)  # heads into the obstacle
_AWAY = np.tile([-1.0, 0.0], (6, 1)).astype(np.float32)  # heads away


# --------------------------------------------------------------------------- #
# OracleRollout contract
# --------------------------------------------------------------------------- #


def test_oracle_rollout_validates_frame_shapes():
    good = np.zeros((6, 8, 8, 3), np.uint8)
    with pytest.raises(ValueError, match="one entry per future step"):
        OracleRollout(
            context=np.zeros((3, 8, 8, 3), np.uint8),
            actions=np.zeros((5, 2), np.float32),  # horizon 5 != 6 future steps
            future=good,
            context_id="x",
            is_failure=False,
        )


def test_oracle_rollout_requires_uint8_pixels():
    with pytest.raises(ValueError, match="uint8"):
        OracleRollout(
            context=np.zeros((3, 8, 8, 3), np.float32),
            actions=np.zeros((6, 2), np.float32),
            future=np.zeros((6, 8, 8, 3), np.uint8),
            context_id="x",
            is_failure=False,
        )


# --------------------------------------------------------------------------- #
# ToySimOracle
# --------------------------------------------------------------------------- #


def test_toy_oracle_is_deterministic():
    oracle = ToySimOracle(size=32)
    a = oracle.rollout(_INTO, seed=0)
    b = oracle.rollout(_INTO, seed=0)
    np.testing.assert_array_equal(a.future, b.future)
    assert a.context_id == b.context_id == "toy:seed=0"


def test_toy_oracle_shared_context_for_same_seed():
    oracle = ToySimOracle(size=32)
    into = oracle.rollout(_INTO, seed=0)
    away = oracle.rollout(_AWAY, seed=0)
    np.testing.assert_array_equal(into.context, away.context)  # same start
    assert not np.array_equal(into.future, away.future)  # different actions differ


def test_toy_oracle_labels_collision_as_failure():
    oracle = ToySimOracle(size=32)
    assert oracle.rollout(_INTO, seed=0).is_failure is True
    assert oracle.rollout(_AWAY, seed=0).is_failure is False


def test_toy_oracle_shapes_and_action_dim():
    oracle = ToySimOracle(size=32)
    rollout = oracle.rollout(_INTO, seed=0, context_steps=3)
    assert oracle.action_dim == 2
    assert rollout.context.shape == (3, 32, 32, 3)
    assert rollout.future.shape == (6, 32, 32, 3)


def test_toy_oracle_rejects_wrong_action_shape():
    with pytest.raises(ValueError, match=r"\(horizon, 2\)"):
        ToySimOracle().rollout(np.zeros((6, 3), np.float32), seed=0)


# --------------------------------------------------------------------------- #
# Pair builders (oracle + pixel adapter) — the closed loop
# --------------------------------------------------------------------------- #


def test_make_counterfactual_pair_flags_action_blind_model():
    oracle = ToySimOracle(size=32)
    pair = make_counterfactual_pair(
        oracle,
        ActionBlindBaseline(seed=0),
        seed=0,
        actions_a=_INTO,
        actions_b=_AWAY,
        predict_seed=0,
    )
    assert isinstance(pair, CounterfactualPair)
    assert pair.rollout_a.metadata.context_id == pair.rollout_b.metadata.context_id
    assert pair.rollout_a.has_ground_truth and pair.rollout_b.has_ground_truth
    # action-blind model predicts ~no change to the two action sequences -> caught
    assert CounterfactualDivergence().compute(pair).summary > 0.0


def test_make_voe_pair_auto_labels_and_runs():
    oracle = ToySimOracle(size=32)
    voe = make_voe_pair(
        oracle,
        CopyLastFrameBaseline(),
        seed=0,
        failure_actions=_INTO,
        success_actions=_AWAY,
    )
    assert isinstance(voe, VOEPair)
    assert voe.rollout.metadata.is_failure is True  # auto-labeled by the oracle
    assert FailureFaithfulness().compute(voe).curve.shape[1] == len(_INTO)


def test_builders_reject_latent_adapter_without_encode():
    oracle = ToySimOracle(size=32)
    latent = CopyLastFrameBaseline(modality="latents")  # a latent model with no encode
    with pytest.raises(ValueError, match="must implement encode"):
        make_counterfactual_pair(
            oracle, latent, seed=0, actions_a=_INTO, actions_b=_AWAY
        )


def test_make_voe_pair_requires_matching_horizons():
    oracle = ToySimOracle(size=32)
    with pytest.raises(ValueError, match="share a horizon"):
        make_voe_pair(
            oracle,
            CopyLastFrameBaseline(),
            seed=0,
            failure_actions=_INTO,
            success_actions=_AWAY[:3],
        )


# --------------------------------------------------------------------------- #
# GymSimOracle — real swm env (gated) + graceful skip without the extra
# --------------------------------------------------------------------------- #


def test_gym_oracle_skips_gracefully_without_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "gymnasium", None)  # simulate missing extra
    with pytest.raises(ImportError, match=r"pip install worldproof\[swm\]"):
        GymSimOracle("swm/TwoRoom-v1")


def test_gym_oracle_generates_true_futures():
    pytest.importorskip("stable_worldmodel")
    pytest.importorskip("gymnasium")
    oracle = GymSimOracle("swm/TwoRoom-v1")
    try:
        actions = np.tile([1.0, 0.0], (3, 1)).astype(np.float32)
        rollout = oracle.rollout(actions, seed=0)
        assert rollout.future.shape[0] == 3
        assert rollout.future.dtype == np.uint8
        assert rollout.context_id == "swm/TwoRoom-v1:seed=0"
        # deterministic in (seed, actions)
        np.testing.assert_array_equal(
            rollout.future, oracle.rollout(actions, seed=0).future
        )
    finally:
        oracle.close()

"""Tests for the signature metrics: counterfactual divergence + failure faithfulness."""

from __future__ import annotations

import numpy as np
import pytest

from worldproof.core import CounterfactualPair, Rollout, RolloutMetadata, VOEPair
from worldproof.metrics import (
    CounterfactualDivergence,
    FailureFaithfulness,
    SignatureRunner,
)

T = 3
D = 16


def _meta(context_id=None, is_failure=None):
    return RolloutMetadata(
        fps=10.0, model_id="m", context_id=context_id, is_failure=is_failure
    )


def _rollout(prediction, ground_truth, *, actions, context_id=None, is_failure=None):
    return Rollout(
        "latents",
        np.zeros((3, D), np.float32),
        actions,
        (prediction,),
        _meta(context_id, is_failure),
        ground_truth,
    )


_ACT_A = np.zeros((T, 2), np.float32)
_ACT_B = np.ones((T, 2), np.float32)


def _futures(seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((T, D)).astype(np.float32),
        rng.standard_normal((T, D)).astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# CounterfactualPair contract
# --------------------------------------------------------------------------- #


def _cf_pair(pred_a, pred_b, true_a, true_b):
    return CounterfactualPair(
        _rollout(pred_a, true_a, actions=_ACT_A, context_id="ep1"),
        _rollout(pred_b, true_b, actions=_ACT_B, context_id="ep1"),
    )


def test_pair_requires_ground_truth_on_both():
    a, b = _futures()
    with pytest.raises(ValueError, match="ground-truth"):
        CounterfactualPair(
            Rollout(
                "latents",
                np.zeros((3, D), np.float32),
                _ACT_A,
                (a,),
                _meta("ep1"),
            ),  # no ground truth
            _rollout(b, b, actions=_ACT_B, context_id="ep1"),
        )


def test_pair_requires_matching_context_id():
    a, b = _futures()
    with pytest.raises(ValueError, match="context_id"):
        CounterfactualPair(
            _rollout(a, a, actions=_ACT_A, context_id="ep1"),
            _rollout(b, b, actions=_ACT_B, context_id="ep2"),
        )


def test_pair_requires_context_id_set():
    a, b = _futures()
    with pytest.raises(ValueError, match="context_id"):
        CounterfactualPair(
            _rollout(a, a, actions=_ACT_A),  # context_id None
            _rollout(b, b, actions=_ACT_B),
        )


def test_pair_rejects_identical_actions():
    a, b = _futures()
    with pytest.raises(ValueError, match="different"):
        CounterfactualPair(
            _rollout(a, a, actions=_ACT_A, context_id="ep1"),
            _rollout(b, b, actions=_ACT_A, context_id="ep1"),  # same actions
        )


def test_pair_rejects_modality_mismatch():
    a, _ = _futures()
    latent = _rollout(a, a, actions=_ACT_A, context_id="ep1")
    pixel = Rollout(
        "pixels",
        np.zeros((3, 8, 8, 3), np.uint8),
        _ACT_B,
        (np.zeros((T, 8, 8, 3), np.uint8),),
        RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8), context_id="ep1"),
        np.zeros((T, 8, 8, 3), np.uint8),
    )
    with pytest.raises(ValueError, match="modality"):
        CounterfactualPair(latent, pixel)


# --------------------------------------------------------------------------- #
# CounterfactualDivergence
# --------------------------------------------------------------------------- #


def test_counterfactual_shape_and_direction():
    a, b = _futures()
    result = CounterfactualDivergence().compute(_cf_pair(a, b, a, b))
    assert result.curve.shape == (1, T)
    assert result.higher_is_better is False


def test_counterfactual_perfect_prediction_is_zero():
    a, b = _futures()
    # predictions equal the true futures → predicted contrast == true contrast
    assert CounterfactualDivergence().compute(_cf_pair(a, b, a, b)).summary == (
        pytest.approx(0.0, abs=1e-6)
    )


def test_counterfactual_responds_to_action_blindness():
    a, b = _futures()
    metric = CounterfactualDivergence()
    faithful = metric.compute(_cf_pair(a, b, a, b)).summary
    # action-blind: same prediction regardless of the action sequence
    action_blind = metric.compute(_cf_pair(a, a, a, b)).summary
    assert action_blind > faithful


def test_counterfactual_ranking_real_over_naive_over_broken():
    a, b = _futures()
    metric = CounterfactualDivergence()
    real = metric.compute(_cf_pair(a, b, a, b)).summary  # perfect contrast
    naive = metric.compute(_cf_pair(a, a + 0.5 * (b - a), a, b)).summary  # weak
    broken = metric.compute(_cf_pair(a, a, a, b)).summary  # action-blind
    assert real < naive < broken


# --------------------------------------------------------------------------- #
# VOEPair contract + FailureFaithfulness
# --------------------------------------------------------------------------- #


def test_voe_requires_matching_reference_shape():
    failure, _ = _futures()
    rollout = _rollout(failure, failure, actions=_ACT_A, is_failure=True)
    with pytest.raises(ValueError, match="shape"):
        VOEPair(rollout, np.zeros((T, D + 1), np.float32))


def test_voe_requires_matching_reference_dtype():
    failure, _ = _futures()
    rollout = _rollout(failure, failure, actions=_ACT_A, is_failure=True)
    with pytest.raises(ValueError, match="dtype"):
        VOEPair(rollout, np.zeros((T, D), np.float64))


def _voe(prediction, failure, success):
    return VOEPair(
        _rollout(prediction, failure, actions=_ACT_A, is_failure=True), success
    )


def test_failure_faithfulness_shape_and_direction():
    failure, success = _futures()
    result = FailureFaithfulness().compute(_voe(failure, failure, success))
    assert result.curve.shape == (1, T)
    assert result.higher_is_better is True


def test_failure_faithfulness_positive_when_reproducing_failure():
    failure, success = _futures()
    metric = FailureFaithfulness()
    faithful = metric.compute(
        _voe(failure, failure, success)
    ).summary  # predicts failure
    hallucinated = metric.compute(
        _voe(success, failure, success)
    ).summary  # predicts success
    assert faithful > 0 > hallucinated


def test_failure_faithfulness_responds_to_success_drift_corruption():
    failure, success = _futures()
    metric = FailureFaithfulness()
    faithful = metric.compute(_voe(failure, failure, success)).summary
    drifted = metric.compute(
        _voe(0.5 * (failure + success), failure, success)
    ).summary  # drifted toward the plausible success
    assert drifted < faithful


def test_failure_faithfulness_ranking_real_over_naive_over_broken():
    failure, success = _futures()
    metric = FailureFaithfulness()
    real = metric.compute(_voe(failure, failure, success)).summary
    naive = metric.compute(_voe(0.5 * (failure + success), failure, success)).summary
    broken = metric.compute(_voe(success, failure, success)).summary
    assert real > naive > broken


# --------------------------------------------------------------------------- #
# SignatureRunner
# --------------------------------------------------------------------------- #


def test_signature_runner_runs_and_reports():
    a, b = _futures()
    report = SignatureRunner([CounterfactualDivergence()]).run(
        [_cf_pair(a, b, a, b), _cf_pair(a, a, a, b)]
    )
    assert len(report.runs) == 2
    assert len(report.all_results) == 2
    assert not report.all_skips
    assert report.metric_versions["counterfactual_divergence"] == "1.0.0"


def test_signature_runner_catches_failure():
    class _Boom(CounterfactualDivergence):
        name = "boom"

        def compute(self, pair, *, device=None):
            raise RuntimeError("kaboom")

    a, b = _futures()
    report = SignatureRunner([_Boom()]).run([_cf_pair(a, b, a, b)])
    assert not report.all_results
    assert "failed at runtime" in report.all_skips[0].reason

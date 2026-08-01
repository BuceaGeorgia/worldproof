"""Tests for the invariant metrics: object-count conservation + object permanence."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from corruptions import assert_ranks_real_over_naive_over_broken

from worldproof.core import Rollout, RolloutMetadata
from worldproof.metrics import (
    REGISTRY,
    Capabilities,
    MetricRunner,
    ObjectCountConservation,
    ObjectPermanence,
)

SIZE = 32
T = 4
_COLORS = [(220, 30, 30), (30, 180, 30), (30, 30, 220)]
_META = RolloutMetadata(fps=10.0, model_id="m", resolution=(SIZE, SIZE))
CPU = Capabilities(device=torch.device("cpu"))  # has_tracker=False
CPU_TRACKER = Capabilities(device=torch.device("cpu"), has_tracker=True)


def _frame(centers):
    frame = np.full((SIZE, SIZE, 3), 255, np.uint8)
    for i, (y, x) in enumerate(centers):
        frame[y - 1 : y + 2, x - 1 : x + 2] = _COLORS[i % len(_COLORS)]
    return frame


_THREE = [(6, 6), (6, 22), (22, 12)]


def _rollout(pred_centers_per_step, gt_centers_per_step=None):
    context = np.stack([_frame(_THREE)] * 3)  # 3 objects at the start
    predictions = np.stack([_frame(c) for c in pred_centers_per_step])
    ground_truth = (
        None
        if gt_centers_per_step is None
        else np.stack([_frame(c) for c in gt_centers_per_step])
    )
    return Rollout(
        "pixels",
        context,
        np.zeros((T, 2), np.float32),
        (predictions,),
        _META,
        ground_truth,
    )


# --------------------------------------------------------------------------- #
# Registration / gating
# --------------------------------------------------------------------------- #


def test_registered():
    assert "object_count_conservation" in REGISTRY
    assert "object_permanence" in REGISTRY


def test_runner_skips_without_tracker_capability():
    rollout = _rollout([_THREE] * T)
    run = MetricRunner([ObjectCountConservation()], capabilities=CPU).run(rollout)
    assert not run.results
    assert "tracker" in run.skips[0].reason


def test_runner_runs_with_tracker_capability():
    rollout = _rollout([_THREE] * T)
    run = MetricRunner([ObjectCountConservation()], capabilities=CPU_TRACKER).run(
        rollout
    )
    assert len(run.results) == 1


# --------------------------------------------------------------------------- #
# Object-count conservation (reference-free)
# --------------------------------------------------------------------------- #


def test_count_conservation_shape_and_direction():
    result = ObjectCountConservation().compute(_rollout([_THREE] * T))
    assert result.curve.shape == (1, T)
    assert result.higher_is_better is False


def test_count_conservation_zero_when_count_held():
    assert ObjectCountConservation().compute(_rollout([_THREE] * T)).summary == (
        pytest.approx(0.0)
    )


def test_count_conservation_responds_to_hallucinated_object():
    metric = ObjectCountConservation()
    held = metric.compute(_rollout([_THREE] * T)).summary
    grown = metric.compute(_rollout([_THREE + [(14, 14)]] * T)).summary  # extra object
    assert grown > held


def test_count_conservation_ranking():
    # real holds 3, naive loses one, broken loses two
    real = _rollout([_THREE] * T)
    naive = _rollout([_THREE[:2]] * T)
    broken = _rollout([_THREE[:1]] * T)
    assert_ranks_real_over_naive_over_broken(
        ObjectCountConservation(), real, naive, broken
    )


# --------------------------------------------------------------------------- #
# Object permanence (reference-based)
# --------------------------------------------------------------------------- #


def test_permanence_shape_and_direction():
    result = ObjectPermanence().compute(_rollout([_THREE] * T, [_THREE] * T))
    assert result.curve.shape == (1, T)
    assert result.higher_is_better is True


def test_permanence_perfect_prediction_preserves_all():
    assert ObjectPermanence().compute(
        _rollout([_THREE] * T, [_THREE] * T)
    ).summary == pytest.approx(1.0)


def test_permanence_penalizes_dropped_object():
    metric = ObjectPermanence()
    faithful = metric.compute(_rollout([_THREE] * T, [_THREE] * T)).summary
    forgets = metric.compute(_rollout([_THREE[:2]] * T, [_THREE] * T)).summary
    assert forgets < faithful


def test_permanence_ranking():
    gt = [_THREE] * T
    real = _rollout([_THREE] * T, gt)
    naive = _rollout([_THREE[:2]] * T, gt)  # drops one of three
    broken = _rollout([_THREE[:1]] * T, gt)  # drops two of three
    assert_ranks_real_over_naive_over_broken(ObjectPermanence(), real, naive, broken)


def test_permanence_skips_on_rollout_without_ground_truth():
    run = MetricRunner([ObjectPermanence()], capabilities=CPU_TRACKER).run(
        _rollout([_THREE] * T)  # no ground truth
    )
    assert not run.results
    assert "ground truth" in run.skips[0].reason

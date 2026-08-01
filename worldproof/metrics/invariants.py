"""Invariant metrics: physical consistency a prediction must not violate.

Tracker-based (SPEC §4). Both default to the core
:class:`~worldproof.tracking.BlobTracker` so they run everywhere; inject a pinned
segmenter/point-tracker (behind the ``trackers`` extra) for messy real video.
They declare ``needs_tracker`` — the runner runs them only when a tracker
capability is enabled (``has_tracker``).

- :class:`ObjectCountConservation` (reference-free): in a closed scene the object
  count is conserved, so the predicted per-frame count should not drift from the
  starting count. Lower is better.
- :class:`ObjectPermanence` (reference-based): an object that persists in the
  ground truth (including reappearing after occlusion) must appear in the
  prediction. Score = the fraction of ground-truth objects the prediction
  preserves, per step. Higher is better.

Both assume a **closed scene** (fixed object set — no legitimate entry/exit).
"""

from __future__ import annotations

import numpy as np
import torch

from worldproof.core import Rollout
from worldproof.metrics.base import Metric
from worldproof.metrics.registry import register
from worldproof.tracking import BlobTracker, Detection, Tracker

__all__ = ["ObjectCountConservation", "ObjectPermanence"]


@register
class ObjectCountConservation(Metric):
    """Deviation of the predicted object count from the starting count, lower better."""

    name = "object_count_conservation"
    version = "1.0.0"
    higher_is_better = False
    modality = "pixels"
    needs_tracker = True

    def __init__(self, tracker: Tracker | None = None) -> None:
        self._tracker = tracker or BlobTracker()

    def _curve(self, rollout: Rollout, device: torch.device) -> np.ndarray:
        start_count = len(self._tracker.detect(rollout.context[-1:])[0])
        curve = np.empty((rollout.n_samples, rollout.horizon), dtype=np.float64)
        for sample_idx, prediction in enumerate(rollout.predictions):
            counts = [len(frame) for frame in self._tracker.detect(prediction)]
            curve[sample_idx] = np.abs(np.array(counts) - start_count)
        return curve


def _preserved_fraction(
    truth: tuple[Detection, ...],
    prediction: tuple[Detection, ...],
    threshold: float,
) -> float:
    """Fraction of ``truth`` objects with a nearby ``prediction`` object."""
    if not truth:
        return 1.0
    predicted = [np.asarray(p.centroid) for p in prediction]
    used: set[int] = set()
    matched = 0
    for obj in truth:
        centroid = np.asarray(obj.centroid)
        best_idx, best_dist = None, threshold
        for idx, other in enumerate(predicted):
            if idx in used:
                continue
            distance = float(np.hypot(*(centroid - other)))
            if distance <= best_dist:
                best_idx, best_dist = idx, distance
        if best_idx is not None:
            used.add(best_idx)
            matched += 1
    return matched / len(truth)


@register
class ObjectPermanence(Metric):
    """Fraction of ground-truth objects the prediction preserves, higher better."""

    name = "object_permanence"
    version = "1.0.0"
    higher_is_better = True
    modality = "pixels"
    needs_tracker = True
    needs_reference = True

    def __init__(
        self, tracker: Tracker | None = None, *, match_threshold: float | None = None
    ) -> None:
        self._tracker = tracker or BlobTracker()
        self._match_threshold = match_threshold

    def _curve(self, rollout: Rollout, device: torch.device) -> np.ndarray:
        target = rollout.ground_truth
        assert target is not None  # runner gates needs_reference
        threshold = self._match_threshold
        if threshold is None:
            threshold = max(3.0, 0.1 * min(target.shape[1], target.shape[2]))
        truth = self._tracker.detect(target)
        curve = np.empty((rollout.n_samples, rollout.horizon), dtype=np.float64)
        for sample_idx, prediction in enumerate(rollout.predictions):
            predicted = self._tracker.detect(prediction)
            curve[sample_idx] = [
                _preserved_fraction(truth[t], predicted[t], threshold)
                for t in range(rollout.horizon)
            ]
        return curve

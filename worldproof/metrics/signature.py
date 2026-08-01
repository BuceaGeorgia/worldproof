"""Signature metrics — the moat (SPEC §4).

These are multi-arity: they consume a *pair* of rollouts, not one, so they live
outside the per-rollout :class:`~worldproof.metrics.base.Metric` hierarchy and
are driven by :class:`SignatureRunner`.

- :class:`CounterfactualDivergence` — same context, two action sequences: does
  the model's predicted *change* match reality's? Compared as the mean-absolute
  error of the difference-of-differences per horizon step (lower is better). An
  action-blind model predicts ~zero change and is caught immediately.
- :class:`FailureFaithfulness` — on a curated failure episode paired with a
  plausible success reference (a Violation-of-Expectation setup): does the model
  land closer to the real (failure) outcome than to the plausible alternative?
  ``error(pred, success) - error(pred, failure)`` per step — positive means
  faithful to the real failure, negative means hallucinated success (higher is
  better).

Both use each rollout's **sample-mean** prediction as the representative
trajectory when ``n_samples > 1`` (the per-sample spread is the Calibration
signature's concern, not this one). v0.1 is the curated-data / VOE path; the
sim oracle (SPEC §5) will later generate true futures on-the-fly and auto-label
outcomes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from worldproof.core import CounterfactualPair, MetricResult, Rollout, VOEPair
from worldproof.metrics.base import MetricIdentity
from worldproof.metrics.runner import Capabilities, MetricSkip, RolloutRun, RunReport

__all__ = [
    "CounterfactualDivergence",
    "FailureFaithfulness",
    "SignatureRunner",
]


def _sample_mean(rollout: Rollout) -> np.ndarray:
    """Representative trajectory: the mean prediction across samples, float64."""
    return np.mean(
        [prediction.astype(np.float64) for prediction in rollout.predictions], axis=0
    )


def _mean_abs_per_step(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean-absolute difference over all non-time dims → ``(horizon,)``."""
    horizon = a.shape[0]
    return np.abs(a - b).reshape(horizon, -1).mean(axis=1)


class _SignatureMetric(MetricIdentity, ABC):
    """Result-stamping base for the multi-arity signature metrics.

    Identity (``name`` / ``version`` / ``higher_is_better``) comes from
    :class:`~worldproof.metrics.base.MetricIdentity`, shared with the other
    metric families.
    """

    input_kind: ClassVar[str]

    @abstractmethod
    def compute(self, item: object, *, device: object | None = None) -> MetricResult:
        """Score one input (a pair / VOE episode) and return a stamped result."""

    def _result(self, curve: np.ndarray) -> MetricResult:
        return MetricResult(
            name=self.name,
            curve=np.asarray(curve, dtype=np.float64),
            higher_is_better=self.higher_is_better,
        )


class CounterfactualDivergence(_SignatureMetric):
    """Does the model's predicted action-driven change match reality's?"""

    name = "counterfactual_divergence"
    version = "1.0.0"
    higher_is_better = False
    input_kind = "counterfactual_pair"

    def compute(
        self, pair: CounterfactualPair, *, device: object | None = None
    ) -> MetricResult:
        predicted_contrast = _sample_mean(pair.rollout_b) - _sample_mean(pair.rollout_a)
        true_contrast = pair.rollout_b.ground_truth.astype(  # type: ignore[union-attr]
            np.float64
        ) - pair.rollout_a.ground_truth.astype(np.float64)  # type: ignore[union-attr]
        curve = _mean_abs_per_step(predicted_contrast, true_contrast)
        return self._result(curve[None, :])


class FailureFaithfulness(_SignatureMetric):
    """Does the model reproduce the real failure or hallucinate a success?"""

    name = "failure_faithfulness"
    version = "1.0.0"
    higher_is_better = True
    input_kind = "voe_pair"

    def compute(
        self, episode: VOEPair, *, device: object | None = None
    ) -> MetricResult:
        prediction = _sample_mean(episode.rollout)
        failure = episode.rollout.ground_truth.astype(np.float64)  # type: ignore[union-attr]
        success = episode.success_reference.astype(np.float64)
        error_to_failure = _mean_abs_per_step(prediction, failure)
        error_to_success = _mean_abs_per_step(prediction, success)
        faithfulness = error_to_success - error_to_failure
        return self._result(faithfulness[None, :])


class SignatureRunner:
    """Runs signature metrics over curated pairs/episodes, skip-and-report.

    Like :class:`~worldproof.metrics.runner.MetricRunner` but for the multi-arity
    signature metrics: the input contracts (``CounterfactualPair`` / ``VOEPair``)
    validate their own data, so the only skip is a runtime failure — never a
    crash. All metrics in one runner must accept the same input type.
    """

    def __init__(
        self,
        metrics: Sequence[_SignatureMetric],
        *,
        capabilities: Capabilities | None = None,
    ) -> None:
        self._metrics = tuple(metrics)
        self._caps = capabilities if capabilities is not None else Capabilities.detect()

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    def run(self, inputs: Sequence[object]) -> RunReport:
        runs = []
        for index, item in enumerate(inputs):
            results = []
            skips = []
            for metric in self._metrics:
                try:
                    results.append(metric.compute(item, device=self._caps.device))
                except Exception as exc:  # graceful degradation — never crash
                    skips.append(
                        MetricSkip(
                            metric=metric.name,
                            reason=(
                                f"metric {metric.name!r} skipped: failed at runtime "
                                f"({type(exc).__name__}: {exc})."
                            ),
                        )
                    )
            runs.append(
                RolloutRun(index=index, results=tuple(results), skips=tuple(skips))
            )
        versions = MappingProxyType(
            {metric.name: metric.version for metric in self._metrics}
        )
        return RunReport(
            runs=tuple(runs), device=str(self._caps.device), metric_versions=versions
        )

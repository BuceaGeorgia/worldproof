"""Device-aware metric runner: run what the machine supports, report the rest.

Encodes SPEC §6.1 (the anti-VBench decision). Each metric declares hard
requirements; the runner executes the ones the environment satisfies and
records every skip with an actionable reason. A missing capability — no
ground-truth reference, no sim oracle, no tracker, no accelerator — or a metric
that raises at runtime becomes a *reported skip*, never a crash or an install
error.

This runs the per-rollout metric layer and returns per-rollout results + skips.
Report-layer aggregation across the returned :class:`RunReport` (IQM,
performance profiles, FVD) is a separate consumer, built with the report card.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from worldproof.core import MetricResult, Rollout
from worldproof.device import get_device
from worldproof.metrics.base import Metric

if TYPE_CHECKING:  # typing only — avoids a runtime import cycle with aggregate
    from worldproof.metrics.aggregate import VideoFeatureExtractor

__all__ = [
    "Capabilities",
    "MetricSkip",
    "RolloutRun",
    "RunReport",
    "MetricRunner",
]


@dataclass(frozen=True)
class Capabilities:
    """What the current environment can satisfy for requirement checks.

    ``device`` comes from :func:`worldproof.device.get_device`. Per-rollout
    reference availability is read from the rollout itself, not stored here.
    ``fvd_extractor`` is the video feature extractor FVD uses (``None`` means
    FVD skips-and-reports); every environment capability travels through this
    one object, so callers configure the runner and the report layer the same
    way.
    """

    device: torch.device
    has_sim_oracle: bool = False
    has_tracker: bool = False
    fvd_extractor: VideoFeatureExtractor | None = None

    @classmethod
    def detect(
        cls,
        device: torch.device | str | None = None,
        *,
        has_sim_oracle: bool = False,
        has_tracker: bool = False,
        fvd_extractor: VideoFeatureExtractor | None = None,
    ) -> Capabilities:
        if device is None:
            device = get_device()
        elif isinstance(device, str):
            device = get_device(device)
        return cls(
            device=device,
            has_sim_oracle=has_sim_oracle,
            has_tracker=has_tracker,
            fvd_extractor=fvd_extractor,
        )

    @property
    def has_gpu(self) -> bool:
        """True when a non-CPU accelerator (mps or cuda) is selected."""
        return self.device.type != "cpu"


@dataclass(frozen=True)
class MetricSkip:
    """A metric that did not run, with an actionable reason."""

    metric: str
    reason: str


@dataclass(frozen=True)
class RolloutRun:
    """Per-rollout outcome: the metrics that ran, and the ones that were skipped."""

    index: int
    results: tuple[MetricResult, ...]
    skips: tuple[MetricSkip, ...]


@dataclass(frozen=True)
class RunReport:
    """Outcome over a set of rollouts, plus provenance for reproducibility."""

    runs: tuple[RolloutRun, ...]
    device: str
    metric_versions: Mapping[str, str]

    @property
    def all_results(self) -> list[MetricResult]:
        return [result for run in self.runs for result in run.results]

    @property
    def all_skips(self) -> list[MetricSkip]:
        return [skip for run in self.runs for skip in run.skips]


class MetricRunner:
    """Runs a fixed set of metrics over rollouts, skipping the unsupported ones."""

    def __init__(
        self,
        metrics: Sequence[Metric],
        *,
        capabilities: Capabilities | None = None,
    ) -> None:
        metrics = tuple(metrics)
        for metric in metrics:
            if not isinstance(metric, Metric):
                raise TypeError(
                    f"MetricRunner expects Metric instances, got {type(metric)}"
                )
        self._metrics = metrics
        self._caps = capabilities if capabilities is not None else Capabilities.detect()

    @classmethod
    def from_registry(
        cls,
        names: Sequence[str] | None = None,
        *,
        registry: object | None = None,
        capabilities: Capabilities | None = None,
    ) -> MetricRunner:
        """Build a runner from registered metric classes (all, or a named subset)."""
        if registry is None:
            from worldproof.metrics.registry import REGISTRY

            registry = REGISTRY
        classes = (
            [registry.get(name) for name in names]
            if names is not None
            else registry.classes()
        )
        return cls([cls_() for cls_ in classes], capabilities=capabilities)

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    @property
    def metrics(self) -> tuple[Metric, ...]:
        return self._metrics

    def _skip_reason(self, metric: Metric, rollout: Rollout) -> str | None:
        if metric.modality is not None and metric.modality != rollout.modality:
            return (
                f"metric {metric.name!r} skipped: applies to {metric.modality!r} "
                f"rollouts, but this rollout is {rollout.modality!r}."
            )
        if metric.needs_reference and not rollout.has_ground_truth:
            return (
                f"metric {metric.name!r} skipped: it needs a ground truth to "
                "compare against, and this rollout has none. Give it a rollout "
                "with ground_truth, for example from a dataset."
            )
        if metric.needs_multisample and rollout.n_samples < 2:
            return (
                f"metric {metric.name!r} skipped: it needs at least 2 samples (the "
                "spread across samples is the signal), and this rollout has "
                f"{rollout.n_samples}. Run predict with more samples."
            )
        if metric.needs_tracker and not self._caps.has_tracker:
            return (
                f"metric {metric.name!r} skipped: it needs a tracker. Install one "
                "with `pip install worldproof[trackers]`."
            )
        if metric.needs_gpu and not self._caps.has_gpu:
            return (
                f"metric {metric.name!r} skipped: it needs a GPU, and the selected "
                "device is cpu. Run it on a machine with mps or cuda."
            )
        if metric.needs_sim_oracle and not self._caps.has_sim_oracle:
            return (
                f"metric {metric.name!r} skipped: it needs a sim oracle. Generate "
                "rollouts with one first (see `worldproof generate`)."
            )
        return None

    def run(self, rollout: Rollout, *, index: int = 0) -> RolloutRun:
        """Run every metric on one rollout; unsupported/failing ones are skipped."""
        results: list[MetricResult] = []
        skips: list[MetricSkip] = []
        for metric in self._metrics:
            reason = self._skip_reason(metric, rollout)
            if reason is not None:
                skips.append(MetricSkip(metric=metric.name, reason=reason))
                continue
            try:
                result = metric.compute(rollout, device=self._caps.device)
            except Exception as exc:  # graceful degradation — never crash the run
                skips.append(
                    MetricSkip(
                        metric=metric.name,
                        reason=(
                            f"metric {metric.name!r} skipped: failed at runtime "
                            f"({type(exc).__name__}: {exc})."
                        ),
                    )
                )
                continue
            results.append(result)
        return RolloutRun(index=index, results=tuple(results), skips=tuple(skips))

    def run_many(self, rollouts: Sequence[Rollout]) -> RunReport:
        """Run over a set of rollouts, returning a report with per-rollout runs."""
        runs = tuple(self.run(rollout, index=i) for i, rollout in enumerate(rollouts))
        versions = MappingProxyType(
            {metric.name: metric.version for metric in self._metrics}
        )
        return RunReport(
            runs=runs, device=str(self._caps.device), metric_versions=versions
        )

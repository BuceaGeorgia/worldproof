"""The Metric base class — subclass it, declare requirements, register it.

A concrete metric sets four kinds of class attribute and implements one method:

- ``name`` / ``version`` — identity. ``version`` bumps whenever the metric's
  computation changes, so numbers people cite stay reproducible.
- ``higher_is_better`` — the **authoritative** score direction. It is a required
  class attribute (the single source of truth) and is stamped onto every
  :class:`~worldproof.core.MetricResult` the metric produces, keeping serialized
  results self-describing (SPEC §5 decision log).
- ``needs_reference`` / ``needs_tracker`` / ``needs_gpu`` / ``needs_sim_oracle``
  — hard requirements the runner checks *before* running. Unmet ⇒ the runner
  skips-and-reports; it never crashes (SPEC §6.1).
- ``_curve(rollout, device)`` — returns the per-sample ``(n_samples, horizon)``
  curve; :meth:`compute` wraps it into a ``MetricResult``.

This is the **per-rollout metric layer**. Report-layer aggregation across many
rollouts (IQM, performance profiles, FVD) is a separate consumer of the runner's
report (SPEC §4, §6.6) and is not implemented here.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from worldproof.core import MetricResult, Modality, Rollout
from worldproof.device import get_device

__all__ = ["MetricIdentity", "Metric"]

_REQUIREMENT_FLAGS = (
    "needs_reference",
    "needs_multisample",
    "needs_tracker",
    "needs_gpu",
    "needs_sim_oracle",
)


class MetricIdentity:
    """Shared identity for every metric family in worldproof.

    Three families exist (per-rollout :class:`Metric`, the multi-arity signature
    metrics, and the set-level aggregate measures); all of them identify a
    measure the same way, so the identity lives in one place:

    - ``name`` / ``version``: ``version`` bumps whenever the computation
      changes, so cited numbers stay reproducible (CLAUDE.md).
    - ``higher_is_better``: the authoritative score direction.

    Concrete subclasses are validated at class-creation time; abstract
    intermediate classes are exempt.
    """

    name: ClassVar[str] = ""
    version: ClassVar[str] = ""
    higher_is_better: ClassVar[bool]  # required — no default; enforced below

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return  # intermediate abstract classes are not yet obligated
        if not getattr(cls, "name", ""):
            raise TypeError(f"{cls.__name__} must set a non-empty `name`")
        if not getattr(cls, "version", ""):
            raise TypeError(f"{cls.__name__} must set a non-empty `version`")
        if not isinstance(getattr(cls, "higher_is_better", None), bool):
            raise TypeError(
                f"{cls.__name__} must set `higher_is_better` to a bool (the "
                "authoritative score direction)"
            )


class Metric(MetricIdentity, ABC):
    """Base class for a per-rollout diagnostic metric.

    Concrete subclasses set ``name``, ``version``, ``higher_is_better`` (from
    :class:`MetricIdentity`) and any requirement flags, then implement
    :meth:`_curve`. ``needs_gpu`` means "needs a non-CPU accelerator" (mps or
    cuda); a CUDA-only need would warrant a finer flag, added when a metric
    requires it. ``modality`` restricts the metric to ``"pixels"`` or
    ``"latents"`` rollouts (``None`` = any); the runner skips it with a clear
    reason on a mismatched rollout rather than failing at runtime.
    """

    modality: ClassVar[Modality | None] = None  # None = applies to any modality
    needs_reference: ClassVar[bool] = False
    needs_multisample: ClassVar[bool] = False  # requires rollout.n_samples >= 2
    needs_tracker: ClassVar[bool] = False
    needs_gpu: ClassVar[bool] = False
    needs_sim_oracle: ClassVar[bool] = False

    @classmethod
    def requirements(cls) -> dict[str, bool]:
        """The requirements this metric declares (only the ones it needs)."""
        return {flag: True for flag in _REQUIREMENT_FLAGS if getattr(cls, flag)}

    @abstractmethod
    def _curve(self, rollout: Rollout, device: torch.device) -> np.ndarray:
        """Compute the per-sample ``(n_samples, horizon)`` curve for ``rollout``.

        Read whatever the metric needs off the rollout (``predictions``,
        ``ground_truth``, ``actions``); the runner has already verified this
        metric's declared requirements are satisfied.
        """

    def _artifacts(self, rollout: Rollout, device: torch.device) -> tuple[Path, ...]:
        """Optional artifact paths (clips, plots). Defaults to none."""
        return ()

    def compute(
        self, rollout: Rollout, *, device: torch.device | None = None
    ) -> MetricResult:
        """Run the metric on one rollout and return a stamped ``MetricResult``.

        ``higher_is_better`` and the declared requirements are stamped from the
        class, so the result is self-describing regardless of who called it.
        """
        if device is None:
            device = get_device()
        curve = self._curve(rollout, device)
        return MetricResult(
            name=self.name,
            curve=curve,
            higher_is_better=self.higher_is_better,
            requirements_met=self.requirements(),
            artifacts=self._artifacts(rollout, device),
        )

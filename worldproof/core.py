"""Frozen data contracts for worldproof.

This module defines the two dataclasses every other component builds on:

- :class:`Rollout` — the unit of evaluation: past context, the action
  sequence that was applied, one or more predicted futures (``n_samples`` is
  first-class), an optional ground-truth future, and metadata.
- :class:`MetricResult` — a metric's verdict on a rollout: a per-sample,
  per-step horizon curve, with summary scalars derived on demand.

Both are ``frozen`` dataclasses. Changing their shape or semantics is a
contract change and requires updating SPEC.md §5 in the same commit.

Array conventions
-----------------
- Pixel frames are ``numpy`` ``uint8`` arrays laid out ``(T, H, W, C)`` with
  ``C in {1, 3}``.
- Latents are floating-point arrays laid out ``(T, *latent_dims)``.
- The leading axis is always time. A prediction / ground-truth future spans
  ``horizon`` steps; ``context`` spans however many past steps were given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np

Modality = Literal["pixels", "latents"]

__all__ = [
    "Modality",
    "Rollout",
    "RolloutMetadata",
    "MetricResult",
    "CounterfactualPair",
    "VOEPair",
]


@dataclass(frozen=True)
class RolloutMetadata:
    """Provenance and playback metadata for a :class:`Rollout`.

    Attributes:
        fps: Playback / sampling rate of the frames, in frames per second.
        model_id: Identifier of the model that produced the predictions, or of
            the dataset / source for a pre-generated rollout folder.
        seed: RNG seed used to draw the samples, if known.
        resolution: ``(height, width)`` of the pixel frames. Required for
            ``modality == "pixels"`` (and validated against the frames);
            ``None`` is only valid for latent rollouts.
        camera_id: Identifier of the camera/view this rollout was recorded
            from. v0.1 is one camera per :class:`Rollout` (many rollouts per
            episode); ``camera_id`` is what distinguishes two views of the same
            episode. Only type-validated here.
        lossy_source: ``True`` when the frames were decoded from a lossily
            re-encoded source (e.g. an mp4-backed rollout folder) rather than a
            lossless one. Fidelity scores computed on such frames are polluted
            by codec artefacts; reports must annotate results from lossy
            sources. Set by loaders (see :mod:`worldproof.adapters.folder`).
        inference_latency_s: Wall-clock seconds the model took to produce this
            rollout's predictions, measured at generate time by the adapter.
            Stored so the report card can show latency/cost without re-running
            the model on the (laptop-only) evaluate path. ``None`` when unknown
            (e.g. a pre-generated folder with no timing recorded).
        context_id: Identifier of the shared context / event history this
            rollout branches from. Rollouts with the same ``context_id`` but
            different actions form a counterfactual pair (see
            :class:`CounterfactualPair`). ``None`` when not part of a pair.
        is_failure: Whether this episode's ground-truth outcome is a failure
            (drop, miss, collision). Lets reports slice fidelity by
            failure-only episodes and marks curated failure episodes for the
            failure-faithfulness signature. ``None`` when unlabeled.
    """

    fps: float
    model_id: str
    seed: int | None = None
    resolution: tuple[int, int] | None = None
    camera_id: str | None = None
    lossy_source: bool = False
    inference_latency_s: float | None = None
    context_id: str | None = None
    is_failure: bool | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps!r}")
        if self.camera_id is not None and not isinstance(self.camera_id, str):
            raise TypeError(
                f"camera_id must be a string or None, got {type(self.camera_id)}"
            )
        if self.context_id is not None and not isinstance(self.context_id, str):
            raise TypeError(
                f"context_id must be a string or None, got {type(self.context_id)}"
            )
        if self.is_failure is not None and not isinstance(self.is_failure, bool):
            raise TypeError(
                f"is_failure must be a bool or None, got {type(self.is_failure)}"
            )
        if not isinstance(self.lossy_source, bool):
            raise TypeError(
                f"lossy_source must be a bool, got {type(self.lossy_source)}"
            )
        if self.inference_latency_s is not None:
            latency = self.inference_latency_s
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or latency < 0
            ):
                raise ValueError(
                    "inference_latency_s must be a non-negative number or None, "
                    f"got {latency!r}"
                )
            object.__setattr__(self, "inference_latency_s", float(latency))
        if self.resolution is not None:
            res = self.resolution
            if len(res) != 2 or any(
                isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d <= 0
                for d in res
            ):
                raise ValueError(
                    "resolution must be a positive integer (height, width) pair, "
                    f"got {self.resolution!r}"
                )
            object.__setattr__(self, "resolution", (int(res[0]), int(res[1])))


@dataclass(frozen=True)
class Rollout:
    """A single unit of evaluation.

    A rollout bundles the past ``context``, the ``actions`` applied over the
    prediction ``horizon``, one or more predicted futures (``predictions`` —
    one array per sample, so ``n_samples`` is first-class), an optional
    ``ground_truth`` future, and :class:`RolloutMetadata`.

    All prediction arrays must share one shape ``(horizon, *frame_dims)``; the
    frame dims must match ``context``'s trailing dims, and ``ground_truth`` (if
    present) must match a single prediction exactly. ``actions`` must carry one
    entry per predicted step (leading axis of length ``horizon``).

    Attributes:
        modality: ``"pixels"`` or ``"latents"``; selects the array conventions.
        context: Past frames/latents, ``(T_context, *frame_dims)``.
        actions: Applied action sequence. The leading axis is always the
            prediction ``horizon`` — one entry per predicted step. Sub-stepped
            control is expressed as richer per-step shape, e.g.
            ``(horizon, substeps, action_dim)``; the length equality is never
            relaxed, so predicted steps stay 1:1 aligned with the sim oracle for
            counterfactual divergence.
        predictions: One array per sample, each ``(horizon, *frame_dims)``.
            Length is ``n_samples`` and must be ``>= 1``.
        metadata: Provenance / playback metadata.
        ground_truth: Optional true future, ``(horizon, *frame_dims)``.
    """

    modality: Modality
    context: np.ndarray
    actions: np.ndarray
    predictions: tuple[np.ndarray, ...]
    metadata: RolloutMetadata
    ground_truth: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.modality not in ("pixels", "latents"):
            raise ValueError(
                f"modality must be 'pixels' or 'latents', got {self.modality!r}"
            )

        predictions = tuple(self.predictions)
        object.__setattr__(self, "predictions", predictions)
        if len(predictions) < 1:
            raise ValueError(
                "a Rollout needs at least one prediction (n_samples >= 1); "
                "predictions was empty"
            )

        named_arrays: list[tuple[str, np.ndarray]] = [
            ("context", self.context),
            ("actions", self.actions),
            *((f"predictions[{i}]", p) for i, p in enumerate(predictions)),
        ]
        if self.ground_truth is not None:
            named_arrays.append(("ground_truth", self.ground_truth))
        for name, arr in named_arrays:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr)}")

        base_shape = predictions[0].shape
        for i, p in enumerate(predictions):
            if p.shape != base_shape:
                raise ValueError(
                    "all predictions must share one shape; predictions[0] has "
                    f"{base_shape} but predictions[{i}] has {p.shape}"
                )
        if len(base_shape) < 2:
            raise ValueError(
                "predictions must be at least 2-D (horizon, *frame_dims); got "
                f"shape {base_shape}"
            )

        horizon = base_shape[0]
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        if self.actions.ndim < 1 or self.actions.shape[0] != horizon:
            raise ValueError(
                "actions must carry one entry per predicted step: expected "
                f"leading axis {horizon}, got shape {self.actions.shape}"
            )

        frame_dims = base_shape[1:]
        if self.context.ndim < 1 or self.context.shape[1:] != frame_dims:
            raise ValueError(
                f"context frame dims {self.context.shape[1:]} do not match "
                f"prediction frame dims {frame_dims}"
            )

        if self.ground_truth is not None and self.ground_truth.shape != base_shape:
            raise ValueError(
                f"ground_truth shape {self.ground_truth.shape} must match a "
                f"single prediction {base_shape}"
            )

        future_like = [self.context, *predictions]
        if self.ground_truth is not None:
            future_like.append(self.ground_truth)

        if self.modality == "pixels":
            self._validate_pixels(base_shape, future_like)
        else:
            self._validate_latents(future_like)

    def _validate_pixels(
        self, base_shape: tuple[int, ...], arrays: Sequence[np.ndarray]
    ) -> None:
        for arr in arrays:
            if arr.ndim != 4:
                raise ValueError(
                    "pixel rollouts require (T, H, W, C) arrays, got ndim "
                    f"{arr.ndim} with shape {arr.shape}"
                )
            if arr.dtype != np.uint8:
                raise ValueError(f"pixel frames must be uint8, got dtype {arr.dtype}")
        _, height, width, channels = base_shape
        if channels not in (1, 3):
            raise ValueError(f"pixel frames must have 1 or 3 channels, got {channels}")
        if self.metadata.resolution is None:
            raise ValueError(
                "pixel rollouts require metadata.resolution; set it to "
                f"(height, width) = ({height}, {width})"
            )
        if tuple(self.metadata.resolution) != (height, width):
            raise ValueError(
                f"metadata.resolution {self.metadata.resolution} does not match "
                f"frame (height, width) = ({height}, {width})"
            )

    def _validate_latents(self, arrays: Sequence[np.ndarray]) -> None:
        for arr in arrays:
            if arr.ndim < 2:
                raise ValueError(
                    "latent rollouts require (T, *latent_dims) arrays with "
                    f"ndim >= 2, got ndim {arr.ndim}"
                )
            if not np.issubdtype(arr.dtype, np.floating):
                raise ValueError(
                    f"latent arrays must be floating-point, got dtype {arr.dtype}"
                )

    @property
    def n_samples(self) -> int:
        """Number of predicted samples (``len(predictions)``)."""
        return len(self.predictions)

    @property
    def horizon(self) -> int:
        """Number of predicted steps per sample."""
        return self.predictions[0].shape[0]

    @property
    def has_ground_truth(self) -> bool:
        """Whether a ground-truth future is attached."""
        return self.ground_truth is not None


@dataclass(frozen=True)
class MetricResult:
    """A metric's verdict on a :class:`Rollout`.

    The primary payload is ``curve``, a per-sample, per-step horizon curve of
    shape ``(n_samples, horizon)``. Summary scalars and aggregate curves are
    *derived* on demand (never stored alone, per the "curves, not scalars"
    principle); ``higher_is_better`` fixes the direction so best-of-N and the
    summary scalar are well defined.

    Attributes:
        name: Metric identifier (e.g. ``"psnr"``, ``"psnr@dynamic"``).
        curve: Per-sample horizon curve, ``(n_samples, horizon)``, float. An
            aggregate-only metric (e.g. calibration) stores ``(1, horizon)``.
        higher_is_better: Whether larger curve values are better; sets the
            direction of :attr:`best_curve` and any ranking.
        requirements_met: The requirements this metric declared, each mapped to
            whether it was satisfied. A produced result only exists when its
            requirements were met, so ``Metric.compute`` stamps the declared
            ones as ``True`` (e.g. ``{"needs_reference": True}``); a ``False``
            entry would only appear for a soft/optional requirement a metric
            chose to record.
        artifacts: Paths to artifacts the metric produced (clips, plots).
    """

    name: str
    curve: np.ndarray
    higher_is_better: bool
    requirements_met: Mapping[str, bool] = field(default_factory=dict)
    artifacts: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("MetricResult.name must be a non-empty string")

        if not isinstance(self.curve, np.ndarray):
            raise TypeError(f"curve must be a numpy.ndarray, got {type(self.curve)}")
        curve = self.curve
        if curve.ndim != 2:
            raise ValueError(
                "curve must be 2-D (n_samples, horizon); got ndim "
                f"{curve.ndim} with shape {curve.shape}"
            )
        if curve.shape[0] < 1 or curve.shape[1] < 1:
            raise ValueError(
                "curve must have at least one sample and one step; got shape "
                f"{curve.shape}"
            )
        if not np.issubdtype(curve.dtype, np.number):
            raise TypeError(f"curve must be numeric, got dtype {curve.dtype}")
        if not np.issubdtype(curve.dtype, np.floating):
            object.__setattr__(self, "curve", curve.astype(np.float64))

        requirements = dict(self.requirements_met)
        for key, value in requirements.items():
            if not isinstance(key, str) or not isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f"requirements_met must map str -> bool, got {key!r} -> {value!r}"
                )
        object.__setattr__(
            self,
            "requirements_met",
            MappingProxyType({k: bool(v) for k, v in requirements.items()}),
        )

        object.__setattr__(self, "artifacts", tuple(Path(a) for a in self.artifacts))

    @property
    def n_samples(self) -> int:
        """Number of samples the curve spans."""
        return self.curve.shape[0]

    @property
    def horizon(self) -> int:
        """Number of horizon steps the curve spans."""
        return self.curve.shape[1]

    @property
    def mean_curve(self) -> np.ndarray:
        """Per-step mean across samples, ``(horizon,)``."""
        return self.curve.mean(axis=0)

    @property
    def spread_curve(self) -> np.ndarray:
        """Per-step sample spread (std dev across samples), ``(horizon,)``.

        This is the signal calibration metrics compare against actual error.
        """
        return self.curve.std(axis=0)

    @property
    def best_curve(self) -> np.ndarray:
        """Per-step best-of-N across samples, ``(horizon,)``.

        Uses the max when ``higher_is_better`` else the min.
        """
        reducer = self.curve.max if self.higher_is_better else self.curve.min
        return reducer(axis=0)

    @property
    def summary(self) -> float:
        """The derived summary scalar.

        Canonical default: the mean over the horizon of the sample-mean curve.
        The full ``curve`` is always retained, so alternative reductions
        (area-under-curve, final-step) can be derived without a contract change.
        """
        return float(self.mean_curve.mean())


@dataclass(frozen=True)
class CounterfactualPair:
    """Two rollouts from the same context under different action sequences.

    The unit the counterfactual-divergence signature consumes: same context
    (matched by ``metadata.context_id``), different ``actions``, and a
    ground-truth future on *both* sides so predicted divergence can be compared
    against the true divergence. Both rollouts must share modality, horizon, and
    frame shape.

    Attributes:
        rollout_a: One branch (context + action sequence A + its true future).
        rollout_b: The other branch (same context, action sequence B).
    """

    rollout_a: Rollout
    rollout_b: Rollout

    def __post_init__(self) -> None:
        a, b = self.rollout_a, self.rollout_b
        for name, rollout in (("rollout_a", a), ("rollout_b", b)):
            if not isinstance(rollout, Rollout):
                raise TypeError(f"{name} must be a Rollout, got {type(rollout)}")
            if not rollout.has_ground_truth:
                raise ValueError(
                    f"{name} needs a ground-truth future so the predicted "
                    "divergence can be compared against the true divergence"
                )
        if a.modality != b.modality:
            raise ValueError(
                f"both rollouts must share modality, got {a.modality!r} and "
                f"{b.modality!r}"
            )
        if a.horizon != b.horizon:
            raise ValueError(
                f"both rollouts must share horizon, got {a.horizon} and {b.horizon}"
            )
        if a.predictions[0].shape[1:] != b.predictions[0].shape[1:]:
            raise ValueError(
                "both rollouts must share frame shape; got "
                f"{a.predictions[0].shape[1:]} and {b.predictions[0].shape[1:]}"
            )
        if a.metadata.context_id is None or b.metadata.context_id is None:
            raise ValueError(
                "a counterfactual pair is matched by metadata.context_id; set it "
                "on both rollouts"
            )
        if a.metadata.context_id != b.metadata.context_id:
            raise ValueError(
                "both rollouts must share metadata.context_id, got "
                f"{a.metadata.context_id!r} and {b.metadata.context_id!r}"
            )
        if np.array_equal(a.actions, b.actions):
            raise ValueError(
                "a counterfactual pair needs *different* action sequences; the "
                "two rollouts have identical actions"
            )

    @property
    def modality(self) -> Modality:
        return self.rollout_a.modality

    @property
    def horizon(self) -> int:
        return self.rollout_a.horizon

    @property
    def context_id(self) -> str:
        cid = self.rollout_a.metadata.context_id
        assert cid is not None  # validated in __post_init__
        return cid


@dataclass(frozen=True)
class VOEPair:
    """A Violation-of-Expectation episode for failure faithfulness.

    A curated episode whose ground-truth outcome actually happened (typically a
    failure — ``rollout.metadata.is_failure``), paired with a
    ``success_reference``: a physically-plausible alternative future that did
    *not* happen. The metric checks whether the model's prediction lands closer
    to the real outcome than to the plausible alternative (faithful) or closer
    to the alternative (hallucinated success).

    Attributes:
        rollout: The episode; ``rollout.ground_truth`` is the actual outcome.
        success_reference: The alternative (e.g. success) future, same shape and
            dtype as ``rollout.ground_truth``.
    """

    rollout: Rollout
    success_reference: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.rollout, Rollout):
            raise TypeError(f"rollout must be a Rollout, got {type(self.rollout)}")
        if not self.rollout.has_ground_truth:
            raise ValueError(
                "rollout needs a ground-truth future (the actual outcome the "
                "success_reference is contrasted against)"
            )
        if not isinstance(self.success_reference, np.ndarray):
            raise TypeError(
                f"success_reference must be a numpy.ndarray, got "
                f"{type(self.success_reference)}"
            )
        target = self.rollout.ground_truth
        assert target is not None
        if self.success_reference.shape != target.shape:
            raise ValueError(
                f"success_reference shape {self.success_reference.shape} must "
                f"match ground_truth {target.shape}"
            )
        if self.success_reference.dtype != target.dtype:
            raise ValueError(
                f"success_reference dtype {self.success_reference.dtype} must "
                f"match ground_truth {target.dtype}"
            )

    @property
    def modality(self) -> Modality:
        return self.rollout.modality

    @property
    def horizon(self) -> int:
        return self.rollout.horizon

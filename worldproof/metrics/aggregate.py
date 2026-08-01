"""Report / aggregation-layer measures — set-level, not per-rollout (SPEC §4).

Two v0.1 measures live here. Unlike the per-rollout :class:`~worldproof.metrics.
base.Metric` layer (one rollout → one :class:`~worldproof.core.MetricResult`),
these consume the *whole* evaluation set at once and return a single
:class:`AggregateResult`. They plug into the report card (``worldproof.report``)
alongside IQM / performance profiles, and they skip-and-report exactly like the
per-rollout runner (SPEC §6.1).

- :class:`ActionRecoverabilityProbe` — **the primary latent diagnostic** (SPEC
  §4). Fits one inverse-dynamics probe across the eval set to recover the
  conditioning actions from consecutive *predicted* latents, and reports the
  cross-validated action R² per horizon step. A latent space can be perceptually
  rich yet have near-zero action recoverability; this is the task-grounded test
  that it is organized around *controllable* variables. Pure numpy (a ridge
  probe with K-fold cross-validation), deterministic, core — it runs on a laptop.

- :class:`FrechetVideoDistance` — FVD, a **labeled weak reference** (SPEC §4):
  a distribution-level Fréchet distance between ground-truth and predicted clip
  features. The Fréchet math (:func:`frechet_distance`) is pure numpy and fully
  tested; the video feature extractor is a **plug-point** (:data:`VideoFeature
  Extractor`) so callers bring their own I3D (the pinned default ships behind the
  ``worldproof[fvd]`` extra, deferred). With no extractor configured FVD
  skips-and-reports — it never emits an un-anchored number.

Neither measure is a registered per-rollout ``Metric`` (they are multi-rollout),
so they are built by :func:`default_aggregates` and driven by
:func:`run_aggregates`, mirroring the per-rollout runner's skip-and-report shape.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from worldproof.core import Rollout
from worldproof.metrics.base import MetricIdentity
from worldproof.metrics.runner import MetricSkip

__all__ = [
    "AggregateResult",
    "MeasureSkipped",
    "AggregateMeasure",
    "VideoFeatureExtractor",
    "frechet_distance",
    "ActionRecoverabilityProbe",
    "FrechetVideoDistance",
    "KineticsVideoExtractor",
    "default_video_extractor",
    "default_aggregates",
    "run_aggregates",
]


# --------------------------------------------------------------------------- #
# Result / skip data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AggregateResult:
    """A set-level measure's verdict over the whole evaluation set.

    Attributes:
        name: Measure identifier (e.g. ``"action_recoverability"``, ``"fvd"``).
        version: Bumped whenever the computation changes (cited numbers stay
            reproducible — CLAUDE.md).
        higher_is_better: Authoritative score direction.
        summary: The headline scalar.
        curve: Optional per-horizon-step curve (the probe reports R² per step;
            FVD is a single distribution-level scalar and leaves this ``None``).
            Steps where the measure is undefined are ``float('nan')``.
        n_items: How many items (rollouts / clips) contributed — provenance.
        extra: Free-form measure-specific provenance (kept JSON-friendly).
    """

    name: str
    version: str
    higher_is_better: bool
    summary: float
    curve: tuple[float, ...] | None = None
    n_items: int = 0
    extra: dict[str, object] = field(default_factory=dict)


class MeasureSkipped(Exception):
    """Raised by a measure when it cannot run; carries an actionable reason.

    Distinct from an unexpected failure: both become a
    :class:`~worldproof.metrics.runner.MetricSkip` (the one skip record shared by
    every metric family), but this one means "not applicable here", with a
    message that says what to do.
    """


@runtime_checkable
class AggregateMeasure(Protocol):
    """A set-level measure: identity + a ``compute`` over the whole set."""

    name: str
    version: str
    higher_is_better: bool

    def compute(self, rollouts: Sequence[Rollout], *, seed: int) -> AggregateResult: ...


# --------------------------------------------------------------------------- #
# FVD — Fréchet distance (pure numpy) + the feature-extractor plug-point
# --------------------------------------------------------------------------- #


@runtime_checkable
class VideoFeatureExtractor(Protocol):
    """Embeds video clips into a feature space (the FVD plug-point).

    Called with a sequence of clips, each a ``(T, H, W, C)`` uint8 array (clip
    lengths / sizes may differ between clips — the extractor owns any resizing or
    temporal handling), and returns a ``(N, D)`` float feature matrix, one row
    per clip. The pinned default I3D extractor ships behind ``worldproof[fvd]``
    (deferred); any callable with this shape can be substituted, so a user can
    match the exact extractor their baseline paper used.
    """

    def __call__(self, clips: Sequence[np.ndarray]) -> np.ndarray: ...


def _symmetric_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Principal square root of a symmetric PSD matrix, via eigendecomposition.

    Stable and dependency-free (no scipy ``sqrtm``): symmetrize, clip negative
    eigenvalues (numerical dust) to zero, and rebuild with square-rooted
    eigenvalues.
    """
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def frechet_distance(
    features_a: np.ndarray, features_b: np.ndarray, *, eps: float = 1e-6
) -> float:
    """Fréchet distance between two feature distributions (the FID/FVD formula).

    ``||μ_a − μ_b||² + Tr(Σ_a + Σ_b − 2·(Σ_a Σ_b)^½)``, with the matrix-square-root
    trace computed through the symmetric route
    ``Tr((Σ_a^½ Σ_b Σ_a^½)^½)`` so it stays pure-numpy and numerically stable
    (no scipy). ``eps`` regularizes the covariances. Lower is better; clamped at 0.
    """
    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("features must be 2-D (n_clips, feature_dim)")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"feature dims must match, got {a.shape[1]} and {b.shape[1]}")
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("each distribution needs >= 2 clips to estimate a covariance")

    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    dim = a.shape[1]
    cov_a = np.cov(a, rowvar=False).reshape(dim, dim) + eps * np.eye(dim)
    cov_b = np.cov(b, rowvar=False).reshape(dim, dim) + eps * np.eye(dim)

    cov_a_sqrt = _symmetric_sqrt(cov_a)
    middle = cov_a_sqrt @ cov_b @ cov_a_sqrt
    eigenvalues = np.linalg.eigvalsh((middle + middle.T) / 2.0)
    trace_cross = float(np.sqrt(np.clip(eigenvalues, 0.0, None)).sum())

    mean_term = float((mu_a - mu_b) @ (mu_a - mu_b))
    distance = mean_term + float(np.trace(cov_a) + np.trace(cov_b)) - 2.0 * trace_cross
    return max(distance, 0.0)


class FrechetVideoDistance(MetricIdentity):
    """FVD over the set: Fréchet distance between GT and predicted clip features.

    A labeled *weak reference* (SPEC §4): it tracks perceptual quality, not
    dynamics, and is action-blind — the report card labels it as such. Real
    distribution = the ground-truth clips; generated = every predicted sample
    (pooled), so ``n_samples`` contributes more clips. Requires pixel rollouts
    with ground truth and a :data:`VideoFeatureExtractor`; without one it
    skips-and-reports rather than emit an un-anchored number.
    """

    name: ClassVar[str] = "fvd"
    version: ClassVar[str] = "1.0.0"
    higher_is_better: ClassVar[bool] = False

    def __init__(self, extractor: VideoFeatureExtractor | None = None) -> None:
        self._extractor = extractor

    def compute(self, rollouts: Sequence[Rollout], *, seed: int = 0) -> AggregateResult:
        if self._extractor is None:
            raise MeasureSkipped(
                "fvd skipped: no video feature extractor is configured. Install "
                "the default with `pip install worldproof[fvd]` and enable it "
                "(the `evaluate --fvd` flag, or "
                "Capabilities.detect(fvd_extractor=...)), or supply your own "
                "extractor to match your baseline's."
            )
        usable = [r for r in rollouts if r.modality == "pixels" and r.has_ground_truth]
        if not usable:
            raise MeasureSkipped(
                "fvd skipped: needs pixel rollouts with ground truth (FVD is a "
                "pixel/video metric; latent rollouts and reference-free rollouts "
                "do not apply)."
            )
        real_clips = [r.ground_truth for r in usable]
        generated_clips = [sample for r in usable for sample in r.predictions]
        if len(real_clips) < 2 or len(generated_clips) < 2:
            raise MeasureSkipped(
                "fvd skipped: it needs at least 2 clips on each side to estimate a "
                f"covariance, and it has {len(real_clips)} real and "
                f"{len(generated_clips)} generated. Evaluate more rollouts."
            )

        features_real = np.asarray(self._extractor(real_clips), dtype=np.float64)
        features_generated = np.asarray(
            self._extractor(generated_clips), dtype=np.float64
        )
        distance = frechet_distance(features_real, features_generated)
        return AggregateResult(
            name=self.name,
            version=self.version,
            higher_is_better=self.higher_is_better,
            summary=distance,
            curve=None,
            n_items=len(real_clips) + len(generated_clips),
            extra={
                "n_real_clips": len(real_clips),
                "n_generated_clips": len(generated_clips),
                "feature_dim": int(features_real.shape[1]),
                "note": "quality, not dynamics; action-blind (SPEC §4)",
            },
        )


class KineticsVideoExtractor:
    """The default FVD extractor: a torchvision r3d_18 pretrained on Kinetics-400.

    A :data:`VideoFeatureExtractor` that embeds each clip into a 512-d feature.
    It runs on CPU or MPS (verified), loads the model lazily on the first call,
    and caches it. This is not the I3D used in most published FVD work, so its
    numbers are not comparable to those; FVD is a weak reference (quality, not
    dynamics) either way. Needs ``pip install worldproof[fvd]``. Pass your own
    extractor to match a specific paper's backbone.
    """

    _MEAN: ClassVar[tuple[float, float, float]] = (0.43216, 0.394666, 0.37645)
    _STD: ClassVar[tuple[float, float, float]] = (0.22803, 0.22145, 0.216989)

    def __init__(self, device: object | None = None, *, frame_size: int = 112) -> None:
        self._device_arg = device
        self._frame_size = frame_size
        self._model = None
        self._device = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from torchvision.models.video import R3D_18_Weights, r3d_18
        except ImportError as exc:
            raise ImportError(
                "FVD's default extractor needs torchvision; install it with "
                "`pip install worldproof[fvd]`, or pass your own fvd_extractor."
            ) from exc
        from worldproof.device import get_device

        device = self._device_arg
        if not isinstance(device, torch.device):
            device = get_device(device)
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        model.fc = torch.nn.Identity()  # tap the pooled feature, drop the classifier
        self._model = model.eval().to(device)
        self._device = device

    def __call__(self, clips: Sequence[np.ndarray]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self._ensure_model()
        mean = torch.tensor(self._MEAN, device=self._device).view(1, 3, 1, 1, 1)
        std = torch.tensor(self._STD, device=self._device).view(1, 3, 1, 1, 1)
        features = []
        with torch.no_grad():
            for clip in clips:
                arr = np.asarray(clip)
                if arr.ndim != 4:
                    raise ValueError("each clip must be (T, H, W, C)")
                if arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=-1)  # grayscale to 3 channels
                x = torch.from_numpy(np.ascontiguousarray(arr)).float().div_(255.0)
                x = x.permute(0, 3, 1, 2)  # (T, C, H, W)
                x = F.interpolate(
                    x,
                    size=(self._frame_size, self._frame_size),
                    mode="bilinear",
                    align_corners=False,
                )
                x = x.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
                x = (x.to(self._device) - mean) / std
                features.append(self._model(x).squeeze(0).cpu().numpy())
        return np.asarray(features, dtype=np.float64)


def default_video_extractor(
    device: object | None = None,
) -> VideoFeatureExtractor | None:
    """The default FVD extractor when torchvision is present, else ``None``.

    Returns a :class:`KineticsVideoExtractor` if torchvision is installed
    (``worldproof[fvd]``), otherwise ``None`` so FVD skips-and-reports. This is
    what the ``worldproof evaluate --fvd`` flag uses.
    """
    import importlib.util

    if importlib.util.find_spec("torchvision") is None:
        return None
    return KineticsVideoExtractor(device)


# --------------------------------------------------------------------------- #
# Action-recoverability probe — the primary latent diagnostic (pure numpy)
# --------------------------------------------------------------------------- #


def _ridge_predict_out_of_fold(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """K-fold-CV ridge, folds split by ``groups`` (rollout id) → out-of-fold preds.

    Splitting by group keeps every transition of a rollout in the same fold, so a
    rollout never leaks across the train/test boundary. Features are standardized
    and targets centered on each *train* fold; the closed-form ridge solution is
    applied to the held-out fold. Returns predictions aligned with ``features``.
    """
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of_group = {
        group: int(fold)
        for group, fold in zip(
            rng.permutation(unique_groups),
            np.arange(unique_groups.size) % n_folds,
            strict=True,
        )
    }
    row_fold = np.array([fold_of_group[g] for g in groups])

    # NaN-filled, never np.empty: a bug that leaves rows unwritten must surface
    # as NaN downstream, not as silent garbage in the R^2.
    predictions = np.full_like(targets, np.nan, dtype=np.float64)
    identity = np.eye(features.shape[1])
    for fold in range(n_folds):
        test_mask = row_fold == fold
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            # Callers guarantee 2 <= n_folds <= n_groups, which makes every fold
            # non-empty on both sides; reaching this is a bug, so say so loudly.
            raise RuntimeError(
                "internal error: cross-validation produced an empty fold "
                f"(fold {fold} of {n_folds}); this should be impossible"
            )
        x_train, y_train = features[train_mask], targets[train_mask]
        mean_x = x_train.mean(axis=0)
        std_x = x_train.std(axis=0)
        std_x[std_x < 1e-8] = 1.0
        mean_y = y_train.mean(axis=0)

        x_train_std = (x_train - mean_x) / std_x
        weights = np.linalg.solve(
            x_train_std.T @ x_train_std + alpha * identity,
            x_train_std.T @ (y_train - mean_y),
        )
        x_test_std = (features[test_mask] - mean_x) / std_x
        predictions[test_mask] = x_test_std @ weights + mean_y
    return predictions


class ActionRecoverabilityProbe(MetricIdentity):
    """Recover the conditioning actions from consecutive predicted latents.

    The primary latent diagnostic (SPEC §4). For each latent rollout we form the
    predicted latent trajectory (last context latent + the sample-mean predicted
    latents) and, at each step ``t``, an inverse-dynamics example: features
    ``[z_{t-1}, z_t]`` → target ``action_t``. One ridge probe is fit *across the
    whole eval set* with K-fold cross-validation (folds split by rollout, so no
    rollout leaks), and the **out-of-fold** action R² is reported per horizon
    step (in-sample R² is trivially ~1 on high-dim latents — only cross-validated
    R² is honest). Higher is better; ~0 means the latents do not encode the
    controllable variables.

    Pure numpy and deterministic (seeded folds, closed-form ridge). A linear
    probe is the v0.1 default (SPEC allows "linear / MLP"); an MLP probe is a
    documented follow-on.
    """

    name: ClassVar[str] = "action_recoverability"
    version: ClassVar[str] = "1.0.0"
    higher_is_better: ClassVar[bool] = True
    min_rollouts: ClassVar[int] = 4

    def __init__(self, *, alpha: float = 1.0, n_folds: int = 5) -> None:
        if n_folds < 2:
            raise ValueError(
                f"n_folds must be >= 2 (cross-validation needs held-out data), "
                f"got {n_folds}"
            )
        self._alpha = alpha
        self._n_folds = n_folds

    @staticmethod
    def _latent_trajectory(rollout: Rollout) -> np.ndarray:
        """``(horizon + 1, latent_dim)``: last context latent + mean predictions."""
        horizon = rollout.horizon
        mean_prediction = np.mean(
            [p.astype(np.float64) for p in rollout.predictions], axis=0
        ).reshape(horizon, -1)
        last_context = rollout.context[-1].astype(np.float64).reshape(1, -1)
        return np.concatenate([last_context, mean_prediction], axis=0)

    def compute(self, rollouts: Sequence[Rollout], *, seed: int = 0) -> AggregateResult:
        latent_rollouts = [r for r in rollouts if r.modality == "latents"]
        if not latent_rollouts:
            raise MeasureSkipped(
                "action_recoverability skipped: no latent rollouts (the probe "
                "diagnoses a latent world model's space; generate latent rollouts "
                "with a latent adapter, e.g. swm:<checkpoint>)."
            )
        if len(latent_rollouts) < self.min_rollouts:
            raise MeasureSkipped(
                f"action_recoverability skipped: it needs >= {self.min_rollouts} "
                "latent rollouts to fit the probe without overfitting, and it has "
                f"{len(latent_rollouts)}. Evaluate more rollouts."
            )

        horizon = latent_rollouts[0].horizon
        feature_dim = latent_rollouts[0].context[-1].size
        if any(
            r.horizon != horizon or r.context[-1].size != feature_dim
            for r in latent_rollouts
        ):
            raise MeasureSkipped(
                "action_recoverability skipped: the latent rollouts have different "
                "horizons or latent sizes, so they cannot be pooled into one "
                "probe. Evaluate a set from a single model."
            )

        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        steps: list[int] = []
        groups: list[int] = []
        for group, rollout in enumerate(latent_rollouts):
            trajectory = self._latent_trajectory(rollout)
            actions = rollout.actions.astype(np.float64).reshape(horizon, -1)
            for step in range(horizon):
                features.append(
                    np.concatenate([trajectory[step], trajectory[step + 1]])
                )
                targets.append(actions[step])
                steps.append(step)
                groups.append(group)

        feature_matrix = np.asarray(features)
        target_matrix = np.asarray(targets)
        step_index = np.asarray(steps)
        group_index = np.asarray(groups)

        n_folds = min(self._n_folds, len(latent_rollouts))
        predictions = _ridge_predict_out_of_fold(
            feature_matrix,
            target_matrix,
            group_index,
            alpha=self._alpha,
            n_folds=n_folds,
            seed=seed,
        )

        curve = np.full(horizon, np.nan)
        for step in range(horizon):
            mask = step_index == step
            y_true = target_matrix[mask]
            y_pred = predictions[mask]
            baseline = y_true.mean(axis=0)
            ss_total = float(((y_true - baseline) ** 2).sum())
            ss_residual = float(((y_true - y_pred) ** 2).sum())
            if ss_total > 0.0:
                curve[step] = 1.0 - ss_residual / ss_total

        if not np.isfinite(curve).any():
            raise MeasureSkipped(
                "action_recoverability skipped: the actions have no variance "
                "across the set, so there is nothing for the probe to fit. Use "
                "rollouts with varying action sequences."
            )

        summary = float(np.nanmean(curve))
        return AggregateResult(
            name=self.name,
            version=self.version,
            higher_is_better=self.higher_is_better,
            summary=summary,
            curve=tuple(float(v) for v in curve),
            n_items=len(latent_rollouts),
            extra={
                "probe": "ridge-linear",
                "alpha": self._alpha,
                "n_folds": int(n_folds),
                "n_transitions": int(feature_matrix.shape[0]),
                "latent_dim": int(feature_dim),
            },
        )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def default_aggregates(
    *, fvd_extractor: VideoFeatureExtractor | None = None
) -> list[AggregateMeasure]:
    """The v0.1 set-level measures, ready for :func:`run_aggregates`."""
    return [ActionRecoverabilityProbe(), FrechetVideoDistance(fvd_extractor)]


def run_aggregates(
    rollouts: Sequence[Rollout],
    measures: Sequence[AggregateMeasure],
    *,
    seed: int = 0,
) -> tuple[list[AggregateResult], list[MetricSkip]]:
    """Run set-level measures, skipping (and reporting) the inapplicable ones.

    Mirrors the per-rollout runner (SPEC §6.1): a :class:`MeasureSkipped` becomes
    a reported :class:`~worldproof.metrics.runner.MetricSkip` (the one skip
    record shared by every metric family) with its actionable reason; any other
    exception degrades to a reported runtime-failure skip — the report never
    crashes on a bad measure.
    """
    results: list[AggregateResult] = []
    skips: list[MetricSkip] = []
    for measure in measures:
        try:
            results.append(measure.compute(rollouts, seed=seed))
        except MeasureSkipped as skip:
            skips.append(MetricSkip(metric=measure.name, reason=str(skip)))
        except Exception as exc:  # graceful degradation — never crash the report
            skips.append(
                MetricSkip(
                    metric=measure.name,
                    reason=(
                        f"{measure.name} skipped: failed at runtime "
                        f"({type(exc).__name__}: {exc})."
                    ),
                )
            )
    return results, skips

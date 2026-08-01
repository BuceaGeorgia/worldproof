"""Assemble signature-metric inputs from a sim oracle + a pixel model adapter.

The oracle supplies true futures (and failure labels); the adapter supplies the
model's predictions. These helpers stitch them into ready-to-score rollouts and
into the frozen :class:`~worldproof.core.CounterfactualPair` /
:class:`~worldproof.core.VOEPair` that the signature metrics consume — closing
the loop the sim oracle exists for.

The oracle renders pixels. A pixel adapter scores against them directly; a latent
adapter (with :meth:`~worldproof.adapters.base.WorldModelAdapter.encode`) has the
oracle's true frames encoded into its latent space for the ground truth.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from worldproof.adapters.base import WorldModelAdapter
from worldproof.adapters.folder import save_rollout
from worldproof.core import CounterfactualPair, Rollout, VOEPair
from worldproof.sim.base import OracleRollout, SimOracle

__all__ = [
    "predict_with_truth",
    "make_rollout",
    "make_counterfactual_pair",
    "make_voe_pair",
    "generate_rollouts",
]


def _require_generatable_adapter(adapter: WorldModelAdapter) -> None:
    """A pixel adapter is fine; a latent one must be able to ``encode`` frames.

    The sim oracle renders pixels. A pixel model scores against those directly; a
    latent model needs to encode the oracle's true future through its own encoder
    to get a latent ground truth. A latent adapter that can't encode (e.g. a
    latent baseline) can't be paired with a pixel oracle.
    """
    if adapter.modality == "pixels":
        return
    if not adapter.can_encode:
        raise ValueError(
            "a latent adapter must implement encode(frames) to attach latent "
            "ground truth from a pixel sim oracle. The latent baselines cannot do "
            "this because they do not encode pixels. Use a real latent world model."
        )


def _ground_truth_for(adapter: WorldModelAdapter, frames: np.ndarray) -> np.ndarray:
    """Pixel adapters use the frames as-is; latent adapters encode them."""
    if adapter.modality == "pixels":
        return frames
    return adapter.encode(frames)


def predict_with_truth(
    adapter: WorldModelAdapter,
    truth: OracleRollout,
    *,
    n_samples: int,
    predict_seed: int | None,
) -> Rollout:
    """Predict from the oracle's context, then attach the oracle's true future.

    For a latent model the true future is encoded through the model's encoder so
    it lives in the same latent space as the predictions.
    """
    predicted = adapter.predict(
        truth.context,
        truth.actions,
        truth.horizon,
        n_samples=n_samples,
        seed=predict_seed,
    )
    metadata = dataclasses.replace(
        predicted.metadata, context_id=truth.context_id, is_failure=truth.is_failure
    )
    return dataclasses.replace(
        predicted,
        metadata=metadata,
        ground_truth=_ground_truth_for(adapter, truth.future),
    )


def make_rollout(
    oracle: SimOracle,
    adapter: WorldModelAdapter,
    *,
    seed: int,
    actions: np.ndarray,
    context_steps: int = 3,
    n_samples: int = 1,
    predict_seed: int | None = None,
) -> Rollout:
    """One evaluation rollout: the oracle's true future + the adapter's prediction.

    The oracle generates the true future for ``actions`` (attached as
    ``ground_truth``, with ``context_id``/``is_failure`` labels); the adapter
    predicts from the same context. The result is a ready-to-score ``Rollout``.
    """
    _require_generatable_adapter(adapter)
    truth = oracle.rollout(actions, seed=seed, context_steps=context_steps)
    return predict_with_truth(
        adapter, truth, n_samples=n_samples, predict_seed=predict_seed
    )


def make_counterfactual_pair(
    oracle: SimOracle,
    adapter: WorldModelAdapter,
    *,
    seed: int,
    actions_a: np.ndarray,
    actions_b: np.ndarray,
    context_steps: int = 3,
    n_samples: int = 1,
    predict_seed: int | None = None,
) -> CounterfactualPair:
    """Build a counterfactual pair: one context (``seed``), two action sequences.

    The oracle generates the true future for each action sequence (branching from
    the same seed → shared context), the adapter predicts each, and the results
    are paired for :class:`~worldproof.metrics.CounterfactualDivergence`.
    """
    _require_generatable_adapter(adapter)
    truth_a = oracle.rollout(actions_a, seed=seed, context_steps=context_steps)
    truth_b = oracle.rollout(actions_b, seed=seed, context_steps=context_steps)
    rollout_a = predict_with_truth(
        adapter, truth_a, n_samples=n_samples, predict_seed=predict_seed
    )
    rollout_b = predict_with_truth(
        adapter, truth_b, n_samples=n_samples, predict_seed=predict_seed
    )
    return CounterfactualPair(rollout_a, rollout_b)


def make_voe_pair(
    oracle: SimOracle,
    adapter: WorldModelAdapter,
    *,
    seed: int,
    failure_actions: np.ndarray,
    success_actions: np.ndarray,
    context_steps: int = 3,
    n_samples: int = 1,
    predict_seed: int | None = None,
) -> VOEPair:
    """Build a VOE episode: the real (failure) outcome + a success alternative.

    The oracle generates the true failure future (what happened) and the true
    success future (the plausible alternative under ``success_actions``); the
    adapter predicts under ``failure_actions``. Feeds
    :class:`~worldproof.metrics.FailureFaithfulness`.
    """
    _require_generatable_adapter(adapter)
    if len(failure_actions) != len(success_actions):
        raise ValueError(
            "failure_actions and success_actions must share a horizon so their "
            f"futures line up; got {len(failure_actions)} and {len(success_actions)}"
        )
    failure = oracle.rollout(failure_actions, seed=seed, context_steps=context_steps)
    success = oracle.rollout(success_actions, seed=seed, context_steps=context_steps)
    rollout = predict_with_truth(
        adapter, failure, n_samples=n_samples, predict_seed=predict_seed
    )
    return VOEPair(
        rollout, success_reference=_ground_truth_for(adapter, success.future)
    )


def generate_rollouts(
    oracle: SimOracle,
    adapter: WorldModelAdapter,
    out_dir: str | Path,
    *,
    n: int,
    horizon: int,
    n_samples: int = 1,
    context_steps: int = 3,
    seed: int = 0,
) -> list[Path]:
    """Generate ``n`` rollouts (oracle truth + model predictions) into ``out_dir``.

    Each rollout uses a fresh seed (``seed + i``) for both the oracle's initial
    state and the random action sequence (via ``oracle.sample_actions``), so runs
    are reproducible. Returns the written rollout directories. This is what the
    ``worldproof generate`` CLI verb wraps.
    """
    out_dir = Path(out_dir)
    written: list[Path] = []
    for i in range(n):
        rollout_seed = seed + i
        actions = oracle.sample_actions(horizon, seed=rollout_seed)
        rollout = make_rollout(
            oracle,
            adapter,
            seed=rollout_seed,
            actions=actions,
            context_steps=context_steps,
            n_samples=n_samples,
            predict_seed=rollout_seed,
        )
        written.append(save_rollout(rollout, out_dir / f"ep_{i:04d}"))
    return written

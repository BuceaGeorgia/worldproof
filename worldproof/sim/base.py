"""Sim-oracle protocol: generate *true* futures for arbitrary action sequences.

A :class:`SimOracle` is a ground-truth provider (SPEC §5). Given a seed (a
deterministic initial state) and an action sequence, it produces the real future
by stepping a simulator — the thing no fixed dataset can do, because it answers
"what *would* have happened under these actions?". This is what unblocks the
signature metrics: two action sequences from one seed give a true counterfactual
divergence, and the env's own outcome (reward / termination) auto-labels failure.

The oracle deals in **pixels**: it renders true future frames comparable to a
pixel model's predictions. (Latent-space oracles — encoding rendered frames
through a model's encoder — are a v0.2 follow-on.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from worldproof.core import Modality

__all__ = ["OracleRollout", "SimOracle"]


@dataclass(frozen=True)
class OracleRollout:
    """One true rollout from a sim oracle (before any model predicts).

    Attributes:
        context: Initial observed frames ``(T_context, H, W, C)`` uint8 — the
            shared starting point a counterfactual branches from.
        actions: The applied action sequence ``(horizon, action_dim)``.
        future: The **true** future frames ``(horizon, H, W, C)`` uint8.
        context_id: Identifier of the initial state (seed-derived); two rollouts
            from the same seed share it, forming a counterfactual pair.
        is_failure: The env's outcome label for this rollout (drop / miss /
            collision / no-success).
        info: Raw per-rollout outcome details (rewards, termination, etc.).
        fps: Sampling rate of ``context``/``future``, when the source knows it
            (a recorded dataset does; a sim oracle generally does not). ``None``
            leaves the model adapter's ``fps`` in the resulting rollout metadata.
    """

    context: np.ndarray
    actions: np.ndarray
    future: np.ndarray
    context_id: str
    is_failure: bool
    info: Mapping[str, object] = field(default_factory=dict)
    fps: float | None = None

    def __post_init__(self) -> None:
        for name, arr in (("context", self.context), ("future", self.future)):
            if not isinstance(arr, np.ndarray) or arr.ndim != 4:
                raise ValueError(f"{name} must be (T, H, W, C) pixel frames")
            if arr.dtype != np.uint8:
                raise ValueError(f"{name} must be uint8, got {arr.dtype}")
        if self.actions.ndim < 1 or self.actions.shape[0] != self.future.shape[0]:
            raise ValueError(
                "actions must carry one entry per future step: expected leading "
                f"axis {self.future.shape[0]}, got shape {self.actions.shape}"
            )
        if self.context.shape[1:] != self.future.shape[1:]:
            raise ValueError(
                f"context frame dims {self.context.shape[1:]} do not match future "
                f"{self.future.shape[1:]}"
            )
        if not isinstance(self.is_failure, bool):
            raise TypeError(f"is_failure must be a bool, got {type(self.is_failure)}")
        if not isinstance(self.context_id, str) or not self.context_id:
            raise ValueError("context_id must be a non-empty string")
        if self.fps is not None and self.fps <= 0:
            raise ValueError(f"fps must be positive when given, got {self.fps!r}")
        object.__setattr__(self, "info", MappingProxyType(dict(self.info)))

    @property
    def horizon(self) -> int:
        return self.future.shape[0]

    @property
    def modality(self) -> Modality:
        return "pixels"


class SimOracle(ABC):
    """A deterministic generator of true futures for arbitrary action sequences."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of one action."""

    @abstractmethod
    def sample_actions(self, horizon: int, *, seed: int) -> np.ndarray:
        """A random, seed-reproducible action sequence ``(horizon, action_dim)``.

        Drawn from the environment's action space; used by generators to produce
        rollouts without a policy (policies are test drivers only, SPEC §3).
        """

    @abstractmethod
    def rollout(
        self, actions: np.ndarray, *, seed: int, context_steps: int = 3
    ) -> OracleRollout:
        """Reset to the seed's initial state, then apply ``actions``.

        Records ``context_steps`` frames of the initial state as context, steps
        the simulator through ``actions`` (``horizon = len(actions)``) recording
        the true future frames, and labels the outcome. Must be deterministic in
        ``(seed, actions)`` so two action sequences from one seed branch from an
        identical context.
        """

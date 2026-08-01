"""Sim oracles backed by gymnasium environments (pixel-rendering, CPU-runnable).

``GymSimOracle`` wraps any gymnasium environment that renders ``rgb_array``
frames and steps deterministically in its seed — a real ground-truth provider for
arbitrary action sequences. It imports ``gymnasium`` lazily (behind a backend
extra) and best-effort registers optional env namespaces:

- ``swm/*`` (e.g. ``swm/TwoRoom-v1``, ``swm/PushT-v1``) via the ``swm`` extra.
- ``ALE/*`` Atari games via the ``atari`` extra (``ale-py``), wrapped by the
  :class:`AtariSimOracle` convenience subclass.

A real emulator is the thing no fixed dataset can do: two action sequences from
one seed give a true counterfactual divergence, and the env's own reward /
termination auto-labels failure. Atari in particular is deterministic,
visually varied, and laptop-runnable, so it unlocks the full signature suite on
CPU. ManiSkill3 / Isaac Sim (CUDA/Linux-heavy) are deferred behind the same
:class:`~worldproof.sim.base.SimOracle` interface, skipping gracefully
off-platform.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from worldproof.sim.base import OracleRollout, SimOracle

__all__ = ["GymSimOracle", "AtariSimOracle"]

# One per applied step: reward, terminated, truncated, info.
StepOutcome = dict


def _default_is_failure(outcomes: Sequence[StepOutcome]) -> bool:
    """Heuristic: the episode failed if it never earned a positive reward.

    A sensible default for goal-reaching / score-based envs; override via
    ``failure_fn`` for task-specific labels (e.g. querying ``info`` for
    ``is_grasped``, or a game-specific score delta).
    """
    return sum(float(o["reward"]) for o in outcomes) <= 0.0


def _register_backends(gym: object) -> None:
    """Best-effort: register optional env namespaces if their packages are present.

    ``stable_worldmodel`` registers ``swm/*`` on import; ``ale_py`` needs an
    explicit ``gymnasium.register_envs``. Neither is required — a backend the
    user did not install simply stays unregistered, and ``gym.make`` then raises
    a clear, actionable error naming the extra to install.
    """
    with contextlib.suppress(ImportError):
        importlib.import_module("stable_worldmodel")  # registers swm/* on import

    ale_py = None
    with contextlib.suppress(ImportError):
        ale_py = importlib.import_module("ale_py")
    if ale_py is not None:
        register_envs = getattr(gym, "register_envs", None)
        if register_envs is not None:
            # defensive across gymnasium versions
            with contextlib.suppress(Exception):
                register_envs(ale_py)  # registers the ALE/* namespace


def _make_hint(env_id: str) -> str:
    if env_id.startswith("ALE/"):
        return "install it with `pip install worldproof[atari]`"
    if env_id.startswith("swm/"):
        return "install it with `pip install worldproof[swm]`"
    return "install the backend package that provides this env's namespace"


class GymSimOracle(SimOracle):
    """A sim oracle over a pixel-rendering gymnasium environment.

    Args:
        env_id: Registered environment id (default ``"swm/TwoRoom-v1"``).
        failure_fn: Maps the per-step outcomes to an ``is_failure`` label;
            defaults to :func:`_default_is_failure`.
        env_kwargs: Extra keyword arguments forwarded to ``gymnasium.make`` (e.g.
            ``repeat_action_probability=0.0`` for faithful Atari action
            conditioning, ``frameskip=...``). ``render_mode`` is always set to
            ``"rgb_array"`` and must not be overridden here.
    """

    def __init__(
        self,
        env_id: str = "swm/TwoRoom-v1",
        *,
        failure_fn: Callable[[Sequence[StepOutcome]], bool] | None = None,
        env_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "GymSimOracle requires gymnasium; install a backend extra, e.g. "
                "`pip install worldproof[swm]` (swm/* envs) or "
                "`pip install worldproof[atari]` (ALE/* envs)."
            ) from exc

        _register_backends(gym)
        make_kwargs = dict(env_kwargs or {})
        make_kwargs.pop("render_mode", None)
        try:
            self._env = gym.make(env_id, render_mode="rgb_array", **make_kwargs)
        except Exception as exc:
            raise ImportError(
                f"could not create gym env {env_id!r} ({type(exc).__name__}: "
                f"{exc}); {_make_hint(env_id)}"
            ) from exc

        self._env_id = env_id
        space = self._env.action_space
        self._action_dim = int(np.prod(space.shape))
        self._action_dtype = space.dtype
        self._failure_fn = failure_fn or _default_is_failure

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def sample_actions(self, horizon: int, *, seed: int) -> np.ndarray:
        self._env.action_space.seed(seed)  # seed-reproducible samples
        actions = [np.asarray(self._env.action_space.sample()) for _ in range(horizon)]
        return np.stack(actions, axis=0).astype(np.float32)

    def _frame(self) -> np.ndarray:
        frame = np.asarray(self._env.render(), dtype=np.uint8)
        if frame.ndim == 3 and frame.shape[-1] == 4:  # RGBA -> RGB
            frame = frame[..., :3]
        return np.ascontiguousarray(frame)

    def rollout(
        self, actions: np.ndarray, *, seed: int, context_steps: int = 3
    ) -> OracleRollout:
        actions = np.asarray(actions)
        self._env.reset(seed=seed)
        first = self._frame()
        context = np.repeat(first[None], context_steps, axis=0)

        future = []
        outcomes: list[StepOutcome] = []
        last = first
        done = False
        for action in actions:
            if done:
                future.append(last)  # pad after the episode ends
                continue
            _, reward, terminated, truncated, info = self._env.step(
                np.asarray(action, dtype=self._action_dtype)
            )
            last = self._frame()
            future.append(last)
            outcomes.append(
                {
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": info,
                }
            )
            done = bool(terminated or truncated)

        return OracleRollout(
            context=context,
            actions=actions.astype(np.float32),
            future=np.stack(future, axis=0),
            context_id=f"{self._env_id}:seed={seed}",
            is_failure=bool(self._failure_fn(outcomes)),
            info={"env_id": self._env_id, "outcomes": tuple(outcomes)},
        )

    def close(self) -> None:
        self._env.close()


class AtariSimOracle(GymSimOracle):
    """A sim oracle over an Arcade Learning Environment game (``ALE/*``, ale-py).

    A real, deterministic, CPU-runnable pixel emulator — visually varied ground
    truth that unlocks the full signature suite (counterfactual divergence,
    failure faithfulness, fidelity, invariants) on a laptop. Needs
    ``pip install worldproof[atari]`` (ale-py bundles the ROMs; no CUDA).

    Defaults ``repeat_action_probability=0.0`` so the *requested* action is
    applied faithfully at every step. The rollout is deterministic in
    ``(seed, actions)`` regardless — ``reset(seed=...)`` reseeds the emulator, so
    even sticky-action resampling is reproduced — but disabling sticky actions
    keeps counterfactual contrasts clean (the divergence reflects the action
    difference, not the emulator's RNG).

    Args:
        env_id: An ``ALE/*`` game id (default ``"ALE/Pong-v5"``).
        repeat_action_probability: Sticky-action probability; ``0.0`` (faithful)
            by default.
        failure_fn / env_kwargs: As :class:`GymSimOracle`.
    """

    def __init__(
        self,
        env_id: str = "ALE/Pong-v5",
        *,
        repeat_action_probability: float = 0.0,
        failure_fn: Callable[[Sequence[StepOutcome]], bool] | None = None,
        env_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        merged = dict(env_kwargs or {})
        merged.setdefault("repeat_action_probability", repeat_action_probability)
        super().__init__(env_id, failure_fn=failure_fn, env_kwargs=merged)

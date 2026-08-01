"""A tiny, dependency-free reference sim oracle.

``ToySimOracle`` is a deterministic 2-D point-mass on a grid with an obstacle:
actions displace the point, and the true future is the rendered trajectory. The
point colliding with the obstacle is a failure. It is pure numpy — it runs
everywhere with the core install — so the whole oracle → pair → signature-metric
loop is testable without the optional simulator extra. It is the sim-oracle
analog of the deliberately-simple baselines: real (if trivial) dynamics with a
known ground truth.
"""

from __future__ import annotations

import numpy as np

from worldproof.sim.base import OracleRollout, SimOracle

__all__ = ["ToySimOracle"]


class ToySimOracle(SimOracle):
    """Deterministic point-mass world; collision with the obstacle = failure.

    Args:
        size: Frame height/width in pixels.
        obstacle: ``(y0, y1, x0, x1)`` obstacle box; defaults to a centre block.
    """

    def __init__(
        self, *, size: int = 32, obstacle: tuple[int, int, int, int] | None = None
    ) -> None:
        self._size = size
        if obstacle is None:
            lo, hi = size * 3 // 8, size * 5 // 8
            obstacle = (lo, hi, lo, hi)
        self._obstacle = obstacle

    @property
    def action_dim(self) -> int:
        return 2

    def sample_actions(self, horizon: int, *, seed: int) -> np.ndarray:
        # per-step displacements in roughly [-quarter-frame, +quarter-frame]
        rng = np.random.default_rng(seed)
        scale = self._size / 8.0
        return rng.uniform(-scale, scale, (horizon, 2)).astype(np.float32)

    def _start_position(self, seed: int) -> np.ndarray:
        # Deterministic start in the top strip, clear of the centre obstacle.
        rng = np.random.default_rng(seed)
        margin = 2.0
        return rng.uniform(margin, self._size / 4.0, size=2)

    def _collides(self, pos: np.ndarray) -> bool:
        y, x = int(round(pos[0])), int(round(pos[1]))
        y0, y1, x0, x1 = self._obstacle
        return y0 <= y < y1 and x0 <= x < x1

    def _render(self, pos: np.ndarray) -> np.ndarray:
        frame = np.full((self._size, self._size, 3), 240, dtype=np.uint8)
        y0, y1, x0, x1 = self._obstacle
        frame[y0:y1, x0:x1] = (80, 80, 80)  # obstacle
        y = int(np.clip(round(pos[0]), 1, self._size - 2))
        x = int(np.clip(round(pos[1]), 1, self._size - 2))
        frame[y - 1 : y + 2, x - 1 : x + 2] = (220, 30, 30)  # the point (3x3)
        return frame

    def rollout(
        self, actions: np.ndarray, *, seed: int, context_steps: int = 3
    ) -> OracleRollout:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError(
                f"ToySimOracle expects actions shaped (horizon, 2), got {actions.shape}"
            )
        pos = self._start_position(seed)
        context = np.repeat(self._render(pos)[None], context_steps, axis=0)

        future = []
        is_failure = False
        limit = self._size - 2
        for action in actions:
            pos = np.clip(pos + action, 1.0, limit)
            if self._collides(pos):
                is_failure = True
            future.append(self._render(pos))

        return OracleRollout(
            context=context,
            actions=actions.astype(np.float32),
            future=np.stack(future, axis=0),
            context_id=f"toy:seed={seed}",
            is_failure=is_failure,
            info={"collided": is_failure},
        )

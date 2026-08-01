"""Dataset-futures ground-truth provider (SPEC §5).

The second ground-truth provider alongside the sim oracle. A :class:`DatasetSource`
yields windows of **recorded** trajectories — real observed frames as the future,
with the trajectory's own actions — as :class:`~worldproof.sim.base.OracleRollout`
(a dataset window *is* a true rollout: context + actions + true future + labels).

Unlike a :class:`~worldproof.sim.base.SimOracle`, a dataset cannot answer "what
would happen under *arbitrary* actions?" (its actions are fixed by the
recording), so it is a distinct provider, not a ``SimOracle``. But its output
feeds the *same* builder: :func:`rollouts_from_dataset` runs a model adapter on
each window and attaches the real future as ground truth, reusing the latent
bridge (a latent world model has the real future encoded into its own latent
space via ``adapter.encode``). So one dataset drives pixel *and* latent models.

Concrete sources (e.g. a stable-worldmodel / LeRobot dataset loader) subclass
:class:`DatasetSource`; they live behind their data dependency's extra and are
imported lazily, so the core install is unaffected.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from worldproof.adapters.base import WorldModelAdapter
from worldproof.core import Rollout
from worldproof.sim.base import OracleRollout
from worldproof.sim.build import predict_with_truth

__all__ = ["DatasetSource", "rollouts_from_dataset"]


class DatasetSource(ABC):
    """A source of recorded trajectory windows as true rollouts.

    One implementation per dataset format. :meth:`truths` yields up to ``n``
    windows, each an :class:`OracleRollout` with ``context_steps`` context frames,
    a ``horizon``-step true future, and the window's own actions. ``seed`` makes
    window selection reproducible.
    """

    @abstractmethod
    def truths(
        self, *, n: int, context_steps: int, horizon: int, seed: int = 0
    ) -> Iterator[OracleRollout]:
        """Yield up to ``n`` recorded windows as :class:`OracleRollout`s."""


def rollouts_from_dataset(
    source: DatasetSource,
    adapter: WorldModelAdapter,
    *,
    n: int,
    context_steps: int,
    horizon: int,
    n_samples: int = 1,
    seed: int = 0,
    predict_seed: int | None = None,
) -> list[Rollout]:
    """Run ``adapter`` on each dataset window, attaching the real future as truth.

    For each recorded window the adapter predicts from the window's context under
    its recorded actions; the real observed future is attached as ground truth
    (encoded into latent space for a latent adapter). The result is a list of
    ready-to-``evaluate`` :class:`Rollout`s — the dataset counterpart of
    :func:`worldproof.sim.build.generate_rollouts`.
    """
    return [
        predict_with_truth(
            adapter, truth, n_samples=n_samples, predict_seed=predict_seed
        )
        for truth in source.truths(
            n=n, context_steps=context_steps, horizon=horizon, seed=seed
        )
    ]

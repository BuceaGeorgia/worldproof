"""Tests for the dataset-futures ground-truth provider (SPEC §5).

The core tests use a synthetic in-memory :class:`DatasetSource` (no download), so
they run on the core install: they prove one dataset drives both a pixel model
(fidelity vs the real future) and a latent model (via the encode bridge). A
final, gated test proves the provider drives the *real* swm checkpoint end to
end — it skips cleanly when the ``swm`` stack / checkpoint is absent.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch

from worldproof.adapters.base import WorldModelAdapter
from worldproof.baselines import CopyLastFrameBaseline
from worldproof.metrics import PSNR
from worldproof.report import evaluate
from worldproof.sim import DatasetSource, rollouts_from_dataset
from worldproof.sim.base import OracleRollout


class SyntheticDataset(DatasetSource):
    """Deterministic in-memory dataset of random-frame windows (a test double)."""

    def __init__(self, *, size=16, action_dim=2, channels=3, n_available=8):
        self._size = size
        self._action_dim = action_dim
        self._channels = channels
        self._n_available = n_available

    def truths(
        self, *, n: int, context_steps: int, horizon: int, seed: int = 0
    ) -> Iterator[OracleRollout]:
        rng = np.random.default_rng(seed)
        h = w = self._size
        c = self._channels
        for i in range(min(n, self._n_available)):
            yield OracleRollout(
                context=rng.integers(0, 256, (context_steps, h, w, c), np.uint8),
                actions=rng.uniform(-1, 1, (horizon, self._action_dim)).astype(
                    np.float32
                ),
                future=rng.integers(0, 256, (horizon, h, w, c), np.uint8),
                context_id=f"synthetic:traj={i}",
                is_failure=(i % 2 == 0),
            )


class MockLatentAdapter(WorldModelAdapter):
    """A latent world model: encodes pixel frames to a D-vector, copies the last."""

    def __init__(self, dim=32):
        super().__init__(model_id="mock-latent", modality="latents", seed=0)
        self._dim = dim

    def encode(self, frames):
        flat = frames.reshape(frames.shape[0], -1).astype(np.float32) / 255.0
        return flat[:, : self._dim]

    def _predict_futures(self, context, actions, horizon, n_samples, rng):
        last = self.encode(context)[-1]
        return [np.repeat(last[None], horizon, axis=0).copy() for _ in range(n_samples)]


def test_dataset_pixel_rollouts_are_evaluable():
    rollouts = rollouts_from_dataset(
        SyntheticDataset(), CopyLastFrameBaseline(), n=4, context_steps=3, horizon=5
    )
    assert len(rollouts) == 4
    r = rollouts[0]
    assert r.modality == "pixels" and r.has_ground_truth  # real future attached
    assert r.horizon == 5 and r.context.shape[0] == 3
    assert r.metadata.context_id == "synthetic:traj=0"
    result = PSNR().compute(r, device=torch.device("cpu"))
    assert result.horizon == 5 and np.isfinite(result.summary)


def test_dataset_latent_bridge_encodes_real_future():
    rollouts = rollouts_from_dataset(
        SyntheticDataset(), MockLatentAdapter(), n=3, context_steps=3, horizon=5
    )
    r = rollouts[0]
    assert r.modality == "latents" and r.has_ground_truth
    # the real future is encoded into the model's latent space (the bridge)
    assert r.ground_truth.shape == r.predictions[0].shape
    report, _ = evaluate(rollouts)
    assert any(m.name == "latent_prediction_error" for m in report.metrics)


def test_dataset_respects_n_and_availability():
    source = SyntheticDataset(n_available=2)
    rollouts = rollouts_from_dataset(
        source, CopyLastFrameBaseline(), n=5, context_steps=2, horizon=3
    )
    assert len(rollouts) == 2  # capped by what the dataset has


def test_dataset_drives_real_swm_checkpoint():
    pytest.importorskip("stable_worldmodel")
    from worldproof.adapters.swm import SWMAdapter

    try:
        adapter = SWMAdapter("quentinll/lewm-pusht", device="cpu", seed=0)
    except Exception as exc:  # network / cache miss offline
        pytest.skip(f"swm checkpoint unavailable: {exc}")

    # the pusht checkpoint expects 224x224 frames and 10-dim actions
    source = SyntheticDataset(size=224, action_dim=10, n_available=3)
    rollouts = rollouts_from_dataset(source, adapter, n=3, context_steps=3, horizon=4)
    assert len(rollouts) == 3
    assert all(r.modality == "latents" and r.has_ground_truth for r in rollouts)
    assert rollouts[0].ground_truth.shape == rollouts[0].predictions[0].shape
    report, _ = evaluate(rollouts)
    assert any(m.name == "latent_prediction_error" for m in report.metrics)

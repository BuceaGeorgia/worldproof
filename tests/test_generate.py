"""Tests for `generate`: producing rollout folders from a sim oracle + a model."""

from __future__ import annotations

import numpy as np
import pytest

from worldproof import evaluate, iter_rollouts
from worldproof.adapters.base import WorldModelAdapter
from worldproof.baselines import CopyLastFrameBaseline
from worldproof.cli import main
from worldproof.sim import ToySimOracle, generate_rollouts


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
        future = np.repeat(last[None], horizon, axis=0)
        return [future.copy() for _ in range(n_samples)]


# --------------------------------------------------------------------------- #
# generate_rollouts — pixel and latent
# --------------------------------------------------------------------------- #


def test_generate_pixel_rollouts_are_evaluable(tmp_path):
    paths = generate_rollouts(
        ToySimOracle(size=32), CopyLastFrameBaseline(), tmp_path, n=4, horizon=5
    )
    assert len(paths) == 4
    rollouts = list(iter_rollouts(tmp_path))
    assert len(rollouts) == 4
    assert rollouts[0].modality == "pixels"
    assert rollouts[0].has_ground_truth
    assert rollouts[0].horizon == 5
    report, _ = evaluate(rollouts)
    assert any(m.name == "psnr" for m in report.metrics)  # fidelity ran on gt


def test_generate_latent_rollouts_via_encode_bridge(tmp_path):
    generate_rollouts(
        ToySimOracle(size=32), MockLatentAdapter(), tmp_path, n=3, horizon=5
    )
    rollouts = list(iter_rollouts(tmp_path))
    result = rollouts[0]
    assert result.modality == "latents"
    # the ground truth is the oracle's true future encoded into latent space
    assert result.ground_truth.shape == result.predictions[0].shape
    report, _ = evaluate(rollouts)
    assert any(m.name == "latent_prediction_error" for m in report.metrics)


def test_generate_is_reproducible(tmp_path):
    kw = dict(n=2, horizon=4, seed=7)
    generate_rollouts(
        ToySimOracle(size=32), CopyLastFrameBaseline(), tmp_path / "a", **kw
    )
    generate_rollouts(
        ToySimOracle(size=32), CopyLastFrameBaseline(), tmp_path / "b", **kw
    )
    a = list(iter_rollouts(tmp_path / "a"))
    b = list(iter_rollouts(tmp_path / "b"))
    np.testing.assert_array_equal(a[0].ground_truth, b[0].ground_truth)
    np.testing.assert_array_equal(a[0].actions, b[0].actions)


def test_sample_actions_is_deterministic():
    oracle = ToySimOracle(size=32)
    np.testing.assert_array_equal(
        oracle.sample_actions(5, seed=1), oracle.sample_actions(5, seed=1)
    )
    assert oracle.sample_actions(5, seed=1).shape == (5, 2)


def test_pixel_adapter_encode_raises():
    with pytest.raises(NotImplementedError, match="does not encode"):
        CopyLastFrameBaseline().encode(np.zeros((3, 8, 8, 3), np.uint8))


def test_latent_adapter_without_encode_is_rejected(tmp_path):
    latent_baseline = CopyLastFrameBaseline(modality="latents")  # no encode
    with pytest.raises(ValueError, match="must implement encode"):
        generate_rollouts(
            ToySimOracle(size=32), latent_baseline, tmp_path, n=1, horizon=4
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_generate_then_evaluate(tmp_path, capsys):
    folder = tmp_path / "rollouts"
    code = main(
        [
            "generate",
            "--sim",
            "toy",
            "--model",
            "action-blind",
            "--n",
            "3",
            "--horizon",
            "4",
            "--out",
            str(folder),
        ]
    )
    assert code == 0
    assert "generated 3 rollout(s)" in capsys.readouterr().out
    assert len(list(iter_rollouts(folder))) == 3
    assert main(["evaluate", str(folder)]) == 0  # the produced folder is scorable


def test_cli_generate_rejects_unknown_model(tmp_path, capsys):
    code = main(["generate", "--model", "bogus", "--out", str(tmp_path / "x")])
    assert code == 1
    assert "unknown model" in capsys.readouterr().err

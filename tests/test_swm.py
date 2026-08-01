"""Real-checkpoint tests for the stable-worldmodel adapter.

These require the optional ``swm`` stack and a (small, ~72 MB) pretrained
checkpoint. The whole module skips when ``stable_worldmodel`` is not installed,
so the core suite / core-only CI stays green. After the checkpoint's first
download it is cached under ``~/.stable_worldmodel`` and no network is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_worldmodel")

from conformance import assert_adapter_conforms  # noqa: E402

from worldproof.adapters.swm import SWMAdapter  # noqa: E402

CHECKPOINT = "quentinll/lewm-pusht"
ACTION_DIM = 10  # this checkpoint's action encoder expects 10-dim actions
IMAGE_SIZE = 224
GOLDEN = Path(__file__).parent / "data" / "swm_lewm_pusht_golden.npy"


def _fixed_context() -> np.ndarray:
    # A deterministic, non-uniform context so the encoder produces varied latents.
    grad = np.linspace(0, 255, IMAGE_SIZE, dtype=np.uint8)
    frame = np.broadcast_to(grad[None, :, None], (IMAGE_SIZE, IMAGE_SIZE, 3))
    return np.stack([frame, frame, frame], axis=0).astype(np.uint8)


def _fixed_actions(horizon: int = 4) -> np.ndarray:
    return np.full((horizon, ACTION_DIM), 0.1, dtype=np.float32)


@pytest.fixture(scope="module")
def adapter() -> SWMAdapter:
    try:
        return SWMAdapter(CHECKPOINT, device="cpu", seed=0)
    except Exception as exc:  # network / cache miss offline
        pytest.skip(f"swm checkpoint {CHECKPOINT!r} unavailable: {exc}")


def test_swm_adapter_conforms(adapter):
    context = np.full((3, IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.uint8)
    assert_adapter_conforms(adapter, context=context, action_dim=ACTION_DIM)


def test_swm_adapter_produces_latent_rollout(adapter):
    rollout = adapter.predict(
        _fixed_context(), _fixed_actions(), 4, n_samples=2, seed=0
    )
    assert rollout.modality == "latents"
    assert rollout.metadata.resolution is None
    # latent context and predictions share the feature dim
    assert rollout.context.shape[1] == rollout.predictions[0].shape[1]
    for prediction in rollout.predictions:
        assert prediction.shape[0] == 4
        assert prediction.dtype == np.float32


def test_swm_golden_file(adapter):
    """Deterministic latent output matches the committed golden (no network)."""
    rollout = adapter.predict(
        _fixed_context(), _fixed_actions(), 4, n_samples=1, seed=0
    )
    got = rollout.predictions[0]
    if not GOLDEN.exists():
        pytest.fail(
            f"golden file missing: {GOLDEN}. Regenerate with "
            "tests/data/generate_swm_golden.py."
        )
    expected = np.load(GOLDEN)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)

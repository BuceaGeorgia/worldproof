"""Reusable adapter conformance suite.

``assert_adapter_conforms`` exercises any :class:`WorldModelAdapter` against the
frozen contract: modality/identity, prediction shape+dtype, ``n_samples``
handling, the ``actions`` leading-axis rule, and seed reproducibility. Both
baselines pass it; real adapters (e.g. the swm adapter) call it from their own
tests, supplying a ``context``/``action_dim`` that suits the real model.

Frame dims are read from what the adapter *produces*, not from the input
context — a latent adapter whose input is pixels (DINO-WM) returns a latent
rollout whose stored context differs in shape from the pixel input.

Not a ``test_*`` module, so pytest does not collect it directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from worldproof.adapters.base import WorldModelAdapter
from worldproof.core import Modality, Rollout


def make_context(modality: Modality) -> np.ndarray:
    if modality == "pixels":
        return np.full((3, 8, 8, 3), 128, dtype=np.uint8)
    return np.linspace(0.0, 1.0, 3 * 16, dtype=np.float32).reshape(3, 16)


def make_actions(horizon: int, action_dim: int = 4) -> np.ndarray:
    size = horizon * action_dim
    return np.arange(size, dtype=np.float32).reshape(horizon, action_dim)


def assert_adapter_conforms(
    adapter: WorldModelAdapter,
    *,
    context: np.ndarray | None = None,
    action_dim: int = 4,
) -> None:
    """Assert ``adapter`` honours the WorldModelAdapter contract.

    Args:
        adapter: The adapter under test.
        context: Input context to feed. Defaults to a modality-appropriate
            synthetic array; pass an explicit one for adapters whose input
            modality differs from their output (pixel-in, latent-out).
        action_dim: Action dimension the model expects.
    """
    modality = adapter.modality
    horizon = 5
    n_samples = 3
    if context is None:
        context = make_context(modality)
    actions = make_actions(horizon, action_dim)

    rollout = adapter.predict(context, actions, horizon, n_samples=n_samples, seed=123)

    assert isinstance(rollout, Rollout)
    assert rollout.modality == modality
    assert rollout.metadata.model_id == adapter.model_id
    assert rollout.metadata.seed == 123
    assert rollout.n_samples == n_samples
    assert rollout.horizon == horizon

    # Frame dims come from what the adapter produced; the Rollout contract
    # already guarantees context and predictions share them.
    frame_dims = rollout.predictions[0].shape[1:]
    assert rollout.context.shape[1:] == frame_dims
    expected_dtype = np.uint8 if modality == "pixels" else np.float32
    for prediction in rollout.predictions:
        assert prediction.shape == (horizon, *frame_dims)
        assert prediction.dtype == expected_dtype

    # n_samples handling
    single = adapter.predict(context, actions, horizon, n_samples=1, seed=1)
    assert single.n_samples == 1

    # actions leading axis must equal horizon
    with pytest.raises(ValueError, match="one entry per predicted step"):
        adapter.predict(context, make_actions(horizon + 1, action_dim), horizon, seed=1)

    # invalid horizon / n_samples
    with pytest.raises(ValueError):
        adapter.predict(context, actions, 0, seed=1)
    with pytest.raises(ValueError):
        adapter.predict(context, actions, horizon, n_samples=0, seed=1)

    # seed reproducibility: same seed -> identical predictions
    first = adapter.predict(context, actions, horizon, n_samples=n_samples, seed=7)
    second = adapter.predict(context, actions, horizon, n_samples=n_samples, seed=7)
    for a, b in zip(first.predictions, second.predictions, strict=True):
        np.testing.assert_array_equal(a, b)

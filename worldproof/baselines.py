"""Built-in baseline adapters — the permanent sanity anchors.

These two adapters are deliberately bad, and that is their job. They are
first-class, importable, documented members of the package: the fixed
low-water marks for the ranking test every metric must pass
(``real model > naive baseline > broken baseline``, SPEC §4). They appear in
docs and demos, not just tests.

- :class:`CopyLastFrameBaseline` — the *naive* anchor: a persistence forecast.
- :class:`ActionBlindBaseline` — the *broken* anchor: provably action-blind.

Both support pixel and latent modalities (copy-last-frame is modality-agnostic;
latent support costs nothing).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from worldproof.adapters.base import WorldModelAdapter
from worldproof.core import Modality

__all__ = ["CopyLastFrameBaseline", "ActionBlindBaseline"]


def _tile_last(context: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the final context frame ``horizon`` times along a new time axis."""
    last = context[-1]
    return np.repeat(last[None, ...], horizon, axis=0)


class CopyLastFrameBaseline(WorldModelAdapter):
    """Naive baseline: repeat the final context frame across the whole horizon.

    A persistence forecast — the "nothing moves" prior. It is deterministic and
    action-independent, so with ``n_samples > 1`` every sample is *identical*
    (zero spread). That degenerate spread is exactly what a calibration metric
    should later flag: an overconfident model that reports no uncertainty.
    """

    def __init__(
        self,
        *,
        modality: Modality = "pixels",
        fps: float = 10.0,
        seed: int | None = None,
        camera_id: str | None = None,
        model_id: str = "baseline.copy_last_frame",
    ) -> None:
        super().__init__(
            model_id=model_id,
            modality=modality,
            fps=fps,
            seed=seed,
            camera_id=camera_id,
        )

    def _predict_futures(
        self,
        context: np.ndarray,
        actions: np.ndarray,
        horizon: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> Sequence[np.ndarray]:
        future = _tile_last(context, horizon)
        if self.modality == "pixels":
            future = future.astype(np.uint8, copy=False)
        else:
            future = future.astype(np.float32, copy=False)
        return [future.copy() for _ in range(n_samples)]


class ActionBlindBaseline(WorldModelAdapter):
    """Broken baseline: futures that provably ignore the action input.

    Implemented as copy-last-frame plus small per-sample seeded noise. The noise
    is drawn only from ``rng`` and the sample index — never from ``actions`` —
    so, for a fixed seed, two calls with the same context but *different actions*
    return byte-identical predictions. That action-invariance is precisely the
    pathology counterfactual divergence must catch (SPEC §4).

    The noise also gives it non-zero sample spread, distinguishing it from
    :class:`CopyLastFrameBaseline`: it *looks* like it is modelling uncertainty
    while remaining blind to what the agent actually did. This duality is
    deliberate — a calibration metric should eventually flag exactly this
    adapter as *confidently wrong* (spread that does not track error). That is a
    planned future test case, not an accident.

    Note:
        The action-blind guarantee is stated for a fixed seed. With no seed the
        noise (but never the action-dependence) varies between calls.

    Args:
        noise_scale: Standard deviation of the per-sample noise. For pixels it
            is applied in integer intensity units and clipped to ``[0, 255]``;
            for latents it is added in float units.
    """

    def __init__(
        self,
        *,
        noise_scale: float = 2.0,
        modality: Modality = "pixels",
        fps: float = 10.0,
        seed: int | None = None,
        camera_id: str | None = None,
        model_id: str = "baseline.action_blind",
    ) -> None:
        if noise_scale < 0:
            raise ValueError(f"noise_scale must be non-negative, got {noise_scale!r}")
        super().__init__(
            model_id=model_id,
            modality=modality,
            fps=fps,
            seed=seed,
            camera_id=camera_id,
        )
        self._noise_scale = float(noise_scale)

    def _predict_futures(
        self,
        context: np.ndarray,
        actions: np.ndarray,  # noqa: ARG002 — deliberately ignored (the whole point)
        horizon: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> Sequence[np.ndarray]:
        base = _tile_last(context, horizon)
        futures: list[np.ndarray] = []
        for _ in range(n_samples):
            noise = rng.standard_normal(base.shape) * self._noise_scale
            if self.modality == "pixels":
                noisy = np.clip(
                    base.astype(np.int16) + np.round(noise).astype(np.int16),
                    0,
                    255,
                ).astype(np.uint8)
            else:
                noisy = (base.astype(np.float32) + noise.astype(np.float32)).astype(
                    np.float32
                )
            futures.append(noisy)
        return futures

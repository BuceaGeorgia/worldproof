"""The model-adapter protocol.

A :class:`WorldModelAdapter` wraps any model implementing the SPEC §3 contract

    predict(context, actions, horizon, n_samples=1) -> future

and turns its output into the frozen :class:`~worldproof.core.Rollout`. The
base class owns input validation, deterministic seeding, and Rollout assembly;
a concrete adapter only implements :meth:`_predict_futures`, the actual future
generation.

Settled conventions (see SPEC §5 decision log):

- Output is the frozen ``Rollout`` contract. Pixels are ``numpy`` ``(T,H,W,C)``
  ``uint8``; latents are ``float32``.
- ``actions`` leading axis always equals ``horizon`` (1:1 with predicted steps).
- Device selection, when an adapter needs one, goes through
  :func:`worldproof.device.get_device` — never ``.cuda()`` directly.
- Seeding is explicit: an adapter may take a base ``seed`` at construction and
  :meth:`predict` accepts a per-call ``seed`` override, so any run is
  reproducible. The seed actually used is recorded in the rollout metadata.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from worldproof.core import Modality, Rollout, RolloutMetadata

__all__ = ["WorldModelAdapter"]


class WorldModelAdapter(ABC):
    """Abstract base for anything that predicts futures as :class:`Rollout`s.

    Concrete adapters implement :meth:`_predict_futures`. Identity is exposed
    via :attr:`model_id` and :attr:`modality`.

    Args:
        model_id: Identifier stamped into every produced rollout's metadata.
        modality: ``"pixels"`` or ``"latents"``.
        fps: Playback rate recorded in metadata.
        seed: Base RNG seed; used when :meth:`predict` is called without a
            per-call seed. ``None`` means "not seeded" (non-reproducible).
        camera_id: Optional camera/view identifier recorded in metadata.
    """

    def __init__(
        self,
        *,
        model_id: str,
        modality: Modality = "pixels",
        fps: float = 10.0,
        seed: int | None = None,
        camera_id: str | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a non-empty string")
        if modality not in ("pixels", "latents"):
            raise ValueError(
                f"modality must be 'pixels' or 'latents', got {modality!r}"
            )
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        self._model_id = model_id
        self._modality: Modality = modality
        self._fps = float(fps)
        self._seed = seed
        self._camera_id = camera_id

    @property
    def model_id(self) -> str:
        """Identifier stamped into produced rollouts."""
        return self._model_id

    @property
    def modality(self) -> Modality:
        """``"pixels"`` or ``"latents"``."""
        return self._modality

    @property
    def seed(self) -> int | None:
        """The adapter's base seed, if any."""
        return self._seed

    @abstractmethod
    def _predict_futures(
        self,
        context: np.ndarray,
        actions: np.ndarray,
        horizon: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> Sequence[np.ndarray]:
        """Generate ``n_samples`` futures, each ``(horizon, *frame_dims)``.

        Implementations must draw all randomness from ``rng`` (so a fixed seed
        reproduces the output) and emit the modality's dtype (``uint8`` for
        pixels, ``float32`` for latents). The base class validates the returned
        arrays by constructing the :class:`Rollout`.

        ``context`` is the raw argument passed to :meth:`predict`. For a latent
        world model whose *input* is pixels (e.g. DINO-WM), this is pixels;
        override :meth:`_prepare_context` to encode them so the stored rollout
        context is latent and matches the predicted futures.
        """

    @property
    def can_encode(self) -> bool:
        """Whether this adapter overrides :meth:`encode` (pixels to latents).

        The single source of truth for "does this adapter encode?"; the base
        class and the generators consult this instead of each re-deriving it by
        reflection.
        """
        return type(self).encode is not WorldModelAdapter.encode

    def _prepare_context(self, context: np.ndarray) -> np.ndarray:
        """Transform the raw input context into the array stored on the rollout.

        A latent adapter whose input is pixels (the JEPA/DINO-WM track) needs the
        stored context to be latent so it matches the predicted futures — if it
        implements :meth:`encode`, that happens automatically here. Everything
        else stores the context unchanged. Override for full control.
        """
        if self._modality == "latents" and self.can_encode:
            return self.encode(context)
        return context

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Encode pixel ``frames`` ``(T, H, W, C)`` into this model's latent space.

        Only latent adapters (whose *input* is pixels but *output* is latents)
        implement this — it lets a generator attach a latent ground truth by
        encoding a sim oracle's true future frames through the model's own
        encoder. Pixel adapters do not encode and raise.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not encode pixels to latents; only a "
            "latent world model (e.g. SWMAdapter) can attach latent ground truth "
            "from a pixel sim oracle"
        )

    def predict(
        self,
        context: np.ndarray,
        actions: np.ndarray,
        horizon: int,
        n_samples: int = 1,
        seed: int | None = None,
    ) -> Rollout:
        """Predict ``n_samples`` futures and return them as a :class:`Rollout`.

        Args:
            context: Past frames/latents, array-like ``(T_context, *frame_dims)``.
            actions: Applied actions, array-like with leading axis ``horizon``.
            horizon: Number of steps to predict (``>= 1``).
            n_samples: Number of samples to draw (``>= 1``).
            seed: Per-call seed overriding the adapter's base seed. When both
                are ``None`` the run is not reproducible.

        Returns:
            A validated :class:`Rollout`.

        Raises:
            ValueError: On non-positive ``horizon``/``n_samples``, or when the
                ``actions`` leading axis does not equal ``horizon``.
        """
        context = np.asarray(context)
        actions = np.asarray(actions)

        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError(f"horizon must be an integer >= 1, got {horizon!r}")
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError(f"n_samples must be an integer >= 1, got {n_samples!r}")
        horizon = int(horizon)
        n_samples = int(n_samples)

        if actions.ndim < 1 or actions.shape[0] != horizon:
            raise ValueError(
                "actions must carry one entry per predicted step: expected "
                f"leading axis {horizon}, got shape {actions.shape}"
            )

        if self._modality == "pixels" and context.ndim != 4:
            raise ValueError(
                "a pixel adapter expects context shaped (T, H, W, C); got "
                f"shape {context.shape}"
            )

        effective_seed = self._seed if seed is None else seed
        rng = np.random.default_rng(effective_seed)

        start = time.perf_counter()
        futures = list(self._predict_futures(context, actions, horizon, n_samples, rng))
        if len(futures) != n_samples:
            raise RuntimeError(
                f"{type(self).__name__}._predict_futures returned {len(futures)} "
                f"futures but n_samples={n_samples}"
            )
        stored_context = np.asarray(self._prepare_context(context))
        inference_latency_s = time.perf_counter() - start

        if self._modality == "pixels":
            resolution: tuple[int, int] | None = (
                int(stored_context.shape[1]),
                int(stored_context.shape[2]),
            )
        else:
            resolution = None

        metadata = RolloutMetadata(
            fps=self._fps,
            model_id=self._model_id,
            seed=effective_seed,
            resolution=resolution,
            camera_id=self._camera_id,
            inference_latency_s=inference_latency_s,
        )
        return Rollout(
            modality=self._modality,
            context=stored_context,
            actions=actions,
            predictions=tuple(futures),
            metadata=metadata,
        )

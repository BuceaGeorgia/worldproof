"""Adapter for stable-worldmodel (LeWM / DINO-WM-style) latent world models.

This is a *thin* wrapper: it imports ``stable_worldmodel`` lazily (only when an
adapter is constructed), converts its rollout output into worldproof's frozen
:class:`~worldproof.core.Rollout`, and does nothing else — no training, no
architecture code, no checkpoint remapping. Install the dependency stack with
``pip install worldproof[swm]``.

Modality: **latents**. stable-worldmodel world models take pixel observations
and predict in latent (encoded) space. Per the SPEC §5 decision log ("one
modality per rollout"), the adapter encodes the pixel context into latents via
the model's own encoder, so the whole rollout — context and predicted futures —
is latent and internally consistent.

Shapes (from ``LeWM.rollout``):

- input pixels ``(B, S, T_ctx, C, H, W)`` in ``[0, 1]``; ``S`` = n_samples
- input actions ``(B, S, T_ctx + horizon, action_dim)`` — the context frames
  get zero-padded actions (worldproof's contract supplies only the ``horizon``
  future actions), the future slots get the caller's actions
- output ``predicted_emb`` ``(B, S, T_ctx + horizon + 1, D)``; the last
  ``horizon`` latents per sample are the predicted future

``action_dim`` is model-specific (e.g. 10 for the pusht checkpoint) — the
caller must supply actions of the dimension the checkpoint expects; the adapter
passes them through unchanged.

Note: stable-worldmodel's published checkpoints require ``transformers<5`` (the
5.x ViT weight-name refactor breaks ``load_state_dict``); this pin lives in the
``swm`` extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from worldproof.adapters.base import WorldModelAdapter
from worldproof.device import get_device

__all__ = ["SWMAdapter"]

_EXTRA_HINT = "install it with `pip install worldproof[swm]`"


class SWMAdapter(WorldModelAdapter):
    """Wrap a stable-worldmodel latent world model as a worldproof adapter.

    Args:
        checkpoint: Either a checkpoint name resolvable by
            ``stable_worldmodel.wm.utils.load_pretrained`` (e.g.
            ``"quentinll/lewm-pusht"``) or an already-loaded model instance.
        device: Optional device override (``"cpu"`` | ``"mps"`` | ``"cuda"``);
            autodetected via :func:`worldproof.device.get_device` otherwise.
        history_size: Optional rollout history window; defaults to the model's
            ``predictor.num_frames``.
        fps, seed, camera_id, model_id: Passed to metadata / seeding as usual.
    """

    def __init__(
        self,
        checkpoint: str | Any,
        *,
        device: str | None = None,
        history_size: int | None = None,
        fps: float = 10.0,
        seed: int | None = None,
        camera_id: str | None = None,
        model_id: str | None = None,
    ) -> None:
        model = self._load_model(checkpoint)
        self._torch_device = get_device(device)
        model = model.to(self._torch_device)
        model.eval()
        self._model = model
        self._history_size = history_size

        if model_id is None:
            model_id = (
                f"swm:{checkpoint}" if isinstance(checkpoint, str) else "swm:model"
            )
        super().__init__(
            model_id=model_id,
            modality="latents",
            fps=fps,
            seed=seed,
            camera_id=camera_id,
        )

    @staticmethod
    def _load_model(checkpoint: str | Any) -> Any:
        if not isinstance(checkpoint, str):
            return checkpoint
        try:
            from stable_worldmodel.wm import utils as swm_utils
        except ImportError as exc:
            raise ImportError(
                f"SWMAdapter requires the 'stable-worldmodel' stack; {_EXTRA_HINT}"
            ) from exc
        return swm_utils.load_pretrained(checkpoint)

    def _pixels_to_tensor(self, context: np.ndarray) -> Any:
        """``(T_ctx, H, W, C)`` uint8 -> torch ``(1, T_ctx, C, H, W)`` in [0, 1]."""
        import torch

        arr = np.asarray(context)
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(self._torch_device)
        tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
        return tensor.unsqueeze(0)

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Encode pixel ``frames`` ``(T, H, W, C)`` -> latent ``(T, D)`` float32.

        Used both for the stored latent context (via the base ``_prepare_context``)
        and to encode a sim oracle's true future into latent ground truth.
        """
        import torch

        pixels = self._pixels_to_tensor(frames)
        with torch.no_grad():
            info = self._model.encode({"pixels": pixels})
        return info["emb"][0].detach().cpu().numpy().astype(np.float32)

    def _predict_futures(
        self,
        context: np.ndarray,
        actions: np.ndarray,
        horizon: int,
        n_samples: int,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        # NOTE: LeWM's rollout is deterministic, so ``rng`` is unused and the N
        # samples are identical (zero spread). A stochastic checkpoint would need
        # explicit torch seeding from the effective seed to honour reproducibility
        # and to give meaningful per-sample spread — wire that in if one lands.
        import torch

        actions = np.asarray(actions)
        action_dim = int(actions.shape[1]) if actions.ndim >= 2 else 1
        t_ctx = int(context.shape[0])

        pixels = self._pixels_to_tensor(context)  # (1, T_ctx, C, H, W)
        pixels = pixels.unsqueeze(1).expand(1, n_samples, *pixels.shape[1:])

        action_seq = torch.zeros(
            1, n_samples, t_ctx + horizon, action_dim, device=self._torch_device
        )
        future = torch.from_numpy(
            np.ascontiguousarray(actions.reshape(horizon, action_dim))
        ).float()
        action_seq[:, :, t_ctx:, :] = future.to(self._torch_device)

        with torch.no_grad():
            out = self._model.rollout(
                {"pixels": pixels}, action_seq, history_size=self._history_size
            )
        predicted = out["predicted_emb"]  # (1, n_samples, T_ctx + horizon + 1, D)
        future_latents = predicted[0, :, -horizon:, :]
        return [
            future_latents[s].detach().cpu().numpy().astype(np.float32)
            for s in range(n_samples)
        ]

"""LPIPS perceptual distance — the optional, lazily-imported fidelity metric.

Unlike PSNR/SSIM (pure numpy, always available), LPIPS (Zhang et al. 2018) needs
a pretrained network, so it lives behind the ``fidelity`` extra
(``pip install worldproof[fidelity]``) and imports ``lpips`` lazily — only when
the metric actually runs. If the extra is absent the runner catches the
ImportError and reports a skip, never a crash.

LPIPS is a distance, so **lower is better**. The dynamic variant uses LPIPS
spatial maps, averaged over the moving-region mask.
"""

from __future__ import annotations

import numpy as np
import torch

from worldproof.core import Rollout
from worldproof.masking import dynamic_region_mask
from worldproof.metrics.base import Metric
from worldproof.metrics.registry import register

__all__ = ["LPIPS", "LPIPSDynamic"]

_EXTRA_HINT = "install it with `pip install worldproof[fidelity]`"


@register
class LPIPS(Metric):
    """LPIPS perceptual distance (AlexNet backbone), lower is better."""

    name = "lpips"
    version = "1.0.0"
    higher_is_better = False
    modality = "pixels"
    needs_reference = True
    net = "alex"
    dynamic: bool = False

    def __init__(self) -> None:
        self._model: object | None = None

    def _get_model(self, device: torch.device) -> object:
        if self._model is None:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError(
                    f"the {self.name!r} metric requires the 'lpips' package; "
                    f"{_EXTRA_HINT}"
                ) from exc
            model = lpips.LPIPS(net=self.net, spatial=self.dynamic, verbose=False)
            self._model = model.to(device).eval()
        return self._model

    @staticmethod
    def _to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
        # (T, H, W, C) uint8 -> (T, C, H, W) float in [-1, 1] (LPIPS convention)
        tensor = torch.from_numpy(np.ascontiguousarray(frames)).to(device)
        return tensor.permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0

    def _curve(self, rollout: Rollout, device: torch.device) -> np.ndarray:
        model = self._get_model(device)
        target = rollout.ground_truth
        assert target is not None  # runner gates needs_reference
        target_t = self._to_tensor(target, device)
        masks = dynamic_region_mask(target) if self.dynamic else None
        horizon = rollout.horizon
        curve = np.empty((rollout.n_samples, horizon), dtype=np.float64)
        with torch.no_grad():
            for sample_idx, prediction in enumerate(rollout.predictions):
                distances = model(self._to_tensor(prediction, device), target_t)
                if self.dynamic:
                    per_pixel = distances[:, 0].cpu().numpy()  # (T, H, W)
                    for step in range(horizon):
                        mask = masks[step]
                        frame = per_pixel[step]
                        curve[sample_idx, step] = float(
                            frame[mask].mean() if mask.any() else frame.mean()
                        )
                else:
                    curve[sample_idx] = distances.reshape(horizon).cpu().numpy()
        return curve


@register
class LPIPSDynamic(LPIPS):
    """LPIPS on moving-region pixels only (spatial LPIPS map, masked)."""

    name = "lpips@dynamic"
    dynamic = True

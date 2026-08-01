"""Regenerate the swm adapter golden file.

Run inside an environment with the ``swm`` extra installed:

    python tests/data/generate_swm_golden.py

Produces ``swm_lewm_pusht_golden.npy`` next to this script from a fixed
context/action pair on CPU (device-deterministic). Keep the inputs in sync with
``tests/test_swm.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldproof.adapters.swm import SWMAdapter

CHECKPOINT = "quentinll/lewm-pusht"
ACTION_DIM = 10
IMAGE_SIZE = 224
OUT = Path(__file__).parent / "swm_lewm_pusht_golden.npy"


def fixed_context() -> np.ndarray:
    grad = np.linspace(0, 255, IMAGE_SIZE, dtype=np.uint8)
    frame = np.broadcast_to(grad[None, :, None], (IMAGE_SIZE, IMAGE_SIZE, 3))
    return np.stack([frame, frame, frame], axis=0).astype(np.uint8)


def fixed_actions(horizon: int = 4) -> np.ndarray:
    return np.full((horizon, ACTION_DIM), 0.1, dtype=np.float32)


def main() -> None:
    adapter = SWMAdapter(CHECKPOINT, device="cpu", seed=0)
    rollout = adapter.predict(fixed_context(), fixed_actions(), 4, n_samples=1, seed=0)
    np.save(OUT, rollout.predictions[0])
    print(f"wrote {OUT} shape={rollout.predictions[0].shape}")


if __name__ == "__main__":
    main()

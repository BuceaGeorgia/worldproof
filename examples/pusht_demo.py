"""Real-model PushT demo: drive the dataset-futures provider with a world model.

Runs the report-card pipeline through :func:`worldproof.rollouts_from_dataset` —
the SPEC §5 dataset-futures ground-truth provider — on the **real**
``quentinll/lewm-pusht`` latent world model (a LeWM / DINO-WM checkpoint). Each
window's real future is encoded into the model's latent space (the latent
bridge), so the latent metrics — latent prediction error and the
action-recoverability probe — score the actual model.

Requires the swm extra:  ``pip install worldproof[swm]``  (first run downloads
the ~72 MB checkpoint, then caches it). Run:  ``python examples/pusht_demo.py``.

--------------------------------------------------------------------------------
Ground-truth data: what runs here vs. what needs a real dataset
--------------------------------------------------------------------------------
The provider takes any ``DatasetSource`` — a stream of recorded windows
(``context`` frames + the window's ``actions`` + the true ``future`` frames).
This script uses a small **synthetic** source so it runs end-to-end anywhere and
produces a real report card *from the real model*. To point it at the **real
PushT dataset** instead, implement a ``DatasetSource`` over swm's data layer:

    from stable_worldmodel import data as swm_data
    ds = swm_data.load_dataset("<swm-native lance/hdf5 pusht dataset>")
    #   then window each episode into OracleRollout(context, actions, future, ...)

Known frictions (recon, session 15) when sourcing the real PushT trajectories:
  * ``lerobot/pusht`` is raw LeRobot format — swm's reader wants a top-level
    ``*.lance`` / ``*.h5``; it does not load directly.
  * The ``lerobot://lerobot/pusht`` scheme *does* read LeRobot format, but pulls
    in the ``lerobot`` package, which requires **Python 3.12+** (worldproof
    targets 3.10) — the same lerobot/3.12 friction that blocks the lerobot
    adapter.
  * The checkpoint consumes **10-dim** actions (= frameskip-stacked 2-dim PushT
    actions; ``action_encoder.input_dim = 10``). A real source must reproduce
    swm's frameskip action encoding, or predictions are meaningless.
So the correct real source is a swm-native **Lance/HDF5** PushT dataset loaded on
Python 3.10; converting one once (on a 3.12 box, via swm's LeRobot→Lance tools)
is the clean hand-off. Until that dataset id is confirmed, this demo ships with
the synthetic source and the recipe above.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from worldproof import (
    Capabilities,
    default_video_extractor,
    evaluate,
    report_html,
    report_json,
    rollouts_from_dataset,
)
from worldproof.sim import DatasetSource
from worldproof.sim.base import OracleRollout

CHECKPOINT = "quentinll/lewm-pusht"
IMAGE_SIZE = 224
ACTION_DIM = 10  # the checkpoint's action_encoder.input_dim
CONTEXT_STEPS = 3  # the checkpoint's predictor.num_frames
HORIZON = 6


class _SyntheticPushT(DatasetSource):
    """Stand-in dataset: correctly-shaped windows so the demo runs anywhere.

    Replace with a swm-native Lance/HDF5 PushT source (see module docstring) for a
    real-data report card. Frame *content* is synthetic; the shapes and the
    10-dim action encoding match what the checkpoint expects.
    """

    def truths(
        self, *, n: int, context_steps: int, horizon: int, seed: int = 0
    ) -> Iterator[OracleRollout]:
        rng = np.random.default_rng(seed)
        for i in range(n):
            yield OracleRollout(
                context=rng.integers(
                    0, 256, (context_steps, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8
                ),
                actions=rng.uniform(-1, 1, (horizon, ACTION_DIM)).astype(np.float32),
                future=rng.integers(
                    0, 256, (horizon, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8
                ),
                context_id=f"pusht:traj={i}",
                is_failure=False,
            )


def main(out_dir: str = "./pusht-demo-output", n: int = 12) -> None:
    try:
        from worldproof.adapters import SWMAdapter
    except ImportError as exc:  # pragma: no cover - example script
        raise SystemExit(
            "this demo needs the swm extra: pip install worldproof[swm]"
        ) from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    adapter = SWMAdapter(CHECKPOINT, device="cpu", seed=0)
    rollouts = rollouts_from_dataset(
        _SyntheticPushT(),
        adapter,
        n=n,
        context_steps=CONTEXT_STEPS,
        horizon=HORIZON,
        n_samples=1,
        seed=0,
    )
    # This is a latent model, so the pixel metrics (including FVD) do not apply
    # and are skipped-and-reported; latent_prediction_error and the
    # action-recoverability probe are what run. Passing the extractor just makes
    # the FVD skip reason accurate ("not pixels") rather than "no extractor".
    capabilities = Capabilities.detect(fvd_extractor=default_video_extractor())
    report, run_report = evaluate(rollouts, capabilities=capabilities)

    (out / "report.json").write_text(
        json.dumps(report_json(report), indent=2), encoding="utf-8"
    )
    (out / "report.html").write_text(
        report_html(report, rollouts, run_report), encoding="utf-8"
    )
    print(report.verdict)
    print(f"\nwrote {out / 'report.json'} and {out / 'report.html'}")


if __name__ == "__main__":
    main()

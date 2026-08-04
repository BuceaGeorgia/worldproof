"""Use worldproof with your own model and your own data.

Three journeys, all with core dependencies only, no downloads:

1. Your own model: subclass WorldModelAdapter, implement _predict_futures.
2. Your own recorded data: subclass DatasetSource so recorded trajectories
   become the ground truth your model is scored against.
3. No model wrapper at all: you already have predictions and ground truth as
   arrays (computed anywhere, any framework), so you build Rollout objects,
   save them to a folder, and evaluate the folder. The same folder also works
   from the command line: worldproof evaluate <folder>.

Note on the CLI: `worldproof generate` only wires the built-in models and
simulators. Your own model runs through the Python API below; anything you
save with save_rollout is then scored by `worldproof evaluate` like any other
folder.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from worldproof import (
    DatasetSource,
    OracleRollout,
    Rollout,
    RolloutMetadata,
    WorldModelAdapter,
    evaluate,
    iter_rollouts,
    rollouts_from_dataset,
    save_rollout,
)


class DriftModel(WorldModelAdapter):
    """Journey 1: your own model.

    Implement _predict_futures: given context frames and the action sequence,
    return n_samples predicted futures, each (horizon, H, W, C) uint8 for a
    pixel model. This toy one copies the last frame and drifts it brighter;
    replace the body with a call into your real model. A latent model would
    pass modality="latents" and also implement encode(frames).
    """

    def __init__(self) -> None:
        super().__init__(model_id="my-drift-model", modality="pixels", seed=0)

    def _predict_futures(self, context, actions, horizon, n_samples, rng):
        last = context[-1].astype(np.int16)
        futures = []
        for _ in range(n_samples):
            frames = [
                np.clip(last + 5 * (t + 1) + rng.integers(-2, 3, last.shape), 0, 255)
                for t in range(horizon)
            ]
            futures.append(np.stack(frames).astype(np.uint8))
        return futures


class MyRecordings(DatasetSource):
    """Journey 2: your own recorded data.

    Yield windows of recorded trajectories as OracleRollout: the context
    frames, the actions that were actually taken, and the frames that actually
    followed (the ground truth). Here the arrays are random stand-ins; replace
    them with reads from your storage. If your data is a LeRobotDataset v3.0,
    LeRobotDatasetSource already does this for you.
    """

    def truths(
        self, *, n: int, context_steps: int, horizon: int, seed: int = 0
    ) -> Iterator[OracleRollout]:
        rng = np.random.default_rng(seed)
        for i in range(n):
            video = rng.integers(
                0, 256, (context_steps + horizon, 32, 32, 3), np.uint8
            )  # your camera frames
            actions = rng.uniform(-1, 1, (horizon, 4)).astype(
                np.float32
            )  # your action log
            yield OracleRollout(
                context=video[:context_steps],
                actions=actions,
                future=video[context_steps:],
                context_id=f"recording-{i}",
                is_failure=False,
            )


def main(out_dir: str | Path = "./bring-your-own-output") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Journeys 1 + 2 together: score your model against your recordings.
    rollouts = rollouts_from_dataset(
        MyRecordings(), DriftModel(), n=4, context_steps=3, horizon=5
    )
    report, _ = evaluate(rollouts)
    print("your model vs your recordings:")
    print(report.verdict)

    # Journey 3: no model wrapper. You already have the predicted frames and
    # the true frames as arrays; build Rollout objects and save them.
    rng = np.random.default_rng(0)
    for i in range(4):
        truth = rng.integers(0, 256, (5, 32, 32, 3), np.uint8)  # what happened
        predicted = np.clip(
            truth.astype(np.int16) + rng.integers(-20, 21, truth.shape), 0, 255
        ).astype(np.uint8)  # what your model predicted, computed anywhere
        rollout = Rollout(
            modality="pixels",
            context=rng.integers(0, 256, (3, 32, 32, 3), np.uint8),
            actions=rng.uniform(-1, 1, (5, 4)).astype(np.float32),
            predictions=(predicted,),
            metadata=RolloutMetadata(
                fps=10.0, model_id="precomputed", resolution=(32, 32)
            ),
            ground_truth=truth,
        )
        save_rollout(rollout, out / "rollouts" / f"ep_{i:02d}")

    folder_rollouts = list(iter_rollouts(out / "rollouts"))
    report2, _ = evaluate(folder_rollouts)
    print("\nprecomputed rollout folder:")
    print(report2.verdict)
    print(f"\nthe same folder works from the CLI:  worldproof evaluate {out}/rollouts")
    return out


if __name__ == "__main__":
    main()

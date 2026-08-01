"""worldproof quickstart — generate rollouts, evaluate them, write a report card.

Self-contained: no download, core dependencies only, runs in well under a minute
on a laptop. It uses the built-in toy sim oracle as ground truth and the naive
copy-last-frame baseline as the "model", then scores the rollouts and writes a
JSON blob (for CI) and a single self-contained HTML report card.

    python examples/quickstart.py [output_dir]

Swap ``ToySimOracle`` for ``GymSimOracle("swm/TwoRoom-v1")`` (needs the ``swm``
extra) and ``CopyLastFrameBaseline`` for ``SWMAdapter(...)`` to run a real
environment and a real world model.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from worldproof import (
    Capabilities,
    CopyLastFrameBaseline,
    ToySimOracle,
    evaluate,
    iter_rollouts,
    make_rollout,
    report_html,
    report_json,
    save_rollout,
)

N_ROLLOUTS = 6
HORIZON = 6


def generate_rollouts(folder: Path) -> None:
    """Write a few toy rollouts (oracle truth + baseline prediction) to ``folder``."""
    oracle = ToySimOracle(size=48)
    model = CopyLastFrameBaseline()  # the naive "nothing moves" baseline
    rng = np.random.default_rng(0)
    for i in range(N_ROLLOUTS):
        actions = rng.uniform(-2.0, 2.0, (HORIZON, 2)).astype(np.float32)
        rollout = make_rollout(oracle, model, seed=i, actions=actions, n_samples=2)
        save_rollout(rollout, folder / f"ep_{i:02d}")


def main(output_dir: str | Path | None = None) -> Path:
    output_dir = Path(output_dir or tempfile.mkdtemp(prefix="worldproof-quickstart-"))
    rollouts_dir = output_dir / "rollouts"

    print(f"[1/3] generating {N_ROLLOUTS} toy rollouts in {rollouts_dir} ...")
    generate_rollouts(rollouts_dir)

    print("[2/3] evaluating (this is what `worldproof evaluate` does) ...")
    rollouts = list(iter_rollouts(rollouts_dir))
    report, run_report = evaluate(
        rollouts, capabilities=Capabilities.detect(has_tracker=True)
    )

    print("[3/3] writing report card ...\n")
    (output_dir / "report.json").write_text(
        json.dumps(report_json(report), indent=2), encoding="utf-8"
    )
    (output_dir / "report.html").write_text(
        report_html(report, rollouts, run_report), encoding="utf-8"
    )

    print(report.verdict)
    print(f"\nWrote {output_dir / 'report.json'} and {output_dir / 'report.html'}.")
    print("Open the HTML file in a browser to see the report card.")
    return output_dir


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

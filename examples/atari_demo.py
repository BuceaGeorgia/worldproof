"""Atari demo: a report card for a model on a real Atari game.

Uses the AtariSimOracle (a deterministic ALE emulator) as ground truth, runs a
model against real game frames, and writes a report card. Real pixels, no
downloads, runs on a laptop.

Requires:  pip install worldproof[atari]   (ale-py bundles the ROMs).
Run:       python examples/atari_demo.py

The default model is the copy-last-frame baseline. On a fast game like Pong it
scores poorly (the ball and paddles move every frame), which is the point: the
report shows a model failing where the scene is dynamic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldproof import (
    Capabilities,
    default_video_extractor,
    evaluate,
    report_html,
    report_json,
)
from worldproof.baselines import CopyLastFrameBaseline
from worldproof.sim import AtariSimOracle, make_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ALE/Pong-v5", help="an ALE game id")
    parser.add_argument("--out", default="./atari-demo-output")
    parser.add_argument("--n", type=int, default=8, help="number of rollouts")
    parser.add_argument("--horizon", type=int, default=6)
    args = parser.parse_args()

    try:
        oracle = AtariSimOracle(args.game)
    except ImportError as exc:
        raise SystemExit(
            "needs the atari extra: pip install worldproof[atari]"
        ) from exc

    model = CopyLastFrameBaseline()
    rollouts = [
        make_rollout(
            oracle,
            model,
            seed=i,
            actions=oracle.sample_actions(args.horizon, seed=i),
            context_steps=3,
        )
        for i in range(args.n)
    ]
    # Atari sprites on a plain background are a clean scene, so the built-in
    # numpy tracker is appropriate here and the object invariants run. FVD runs
    # too when worldproof[fvd] is installed, and skips otherwise.
    capabilities = Capabilities.detect(
        has_tracker=True, fvd_extractor=default_video_extractor()
    )
    report, run_report = evaluate(rollouts, capabilities=capabilities)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
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

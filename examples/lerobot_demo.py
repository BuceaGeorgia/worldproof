"""Real-data demo: a report card for a model on a real LeRobotDataset.

Reads a real LeRobotDataset v3.0 (`lerobot/pusht` by default) *directly* from
parquet + mp4 — no `lerobot` package, so it runs on Python 3.10 — windows its
episodes into rollouts, scores a model, and writes a report card. Swap `--repo`
for any LeRobotDataset on the Hub (thousands, including Open-X mirrors).

Requires:  pip install worldproof[lerobot-data,fidelity]   (first run downloads
the dataset, then caches it). Run:  python examples/lerobot_demo.py

The default "model" is the copy-last-frame baseline — a static predictor. Its
report is instructive precisely because it fails the right way: high global PSNR
(the mostly-static scene) but a collapsing dynamic-region score (it cannot
predict the moving object). Point `--repo` at your own dataset and plug in a real
model adapter to diagnose it.
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
    rollouts_from_dataset,
)
from worldproof.baselines import CopyLastFrameBaseline
from worldproof.sim import LeRobotDatasetSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="lerobot/pusht",
        help="LeRobotDataset repo id or local path",
    )
    parser.add_argument("--out", default="./lerobot-demo-output")
    parser.add_argument("--n", type=int, default=12, help="number of rollouts")
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument(
        "--tracker",
        action="store_true",
        help="enable the object invariants (only sensible on a clean scene)",
    )
    args = parser.parse_args()

    try:
        source = LeRobotDatasetSource(args.repo)
    except ImportError as exc:
        raise SystemExit(
            "needs the lerobot-data extra: pip install worldproof[lerobot-data]"
        ) from exc

    rollouts = rollouts_from_dataset(
        source,
        CopyLastFrameBaseline(),
        n=args.n,
        context_steps=args.context_steps,
        horizon=args.horizon,
    )
    # FVD runs when worldproof[fvd] is installed, and skips otherwise. The
    # tracker is off by default because real footage is usually too cluttered
    # for the clean-scene tracker; turn it on with --tracker for a simple scene.
    capabilities = Capabilities.detect(
        has_tracker=args.tracker, fvd_extractor=default_video_extractor()
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

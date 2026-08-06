"""Regenerate the README media from real runs.

Dev-only helper. `docs/` is excluded from the sdist, so this never ships.

Two jobs, either or both per invocation:

    # prediction-vs-truth GIF from a dataset, picking the worst-predicted episode
    python docs/make_media.py --repo lerobot/droid_1.0.1 --horizon 48 \
        --gif docs/img/droid-pred-vs-true.gif

    # report-card PNG from an existing report.html (needs a Chromium binary)
    python docs/make_media.py --report reports/droid-h48/report.html \
        --png docs/img/droid-h48.png

Requires the same extras as the example it mirrors:
`pip install worldproof[lerobot-data,fidelity]`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

CHROMIUM_CANDIDATES = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def find_chromium(explicit: str | None = None) -> Path:
    """Locate a Chromium-family binary for headless screenshots."""
    candidates = (explicit,) if explicit else CHROMIUM_CANDIDATES
    for candidate in candidates:
        if candidate is None:
            continue
        found = shutil.which(candidate) or (
            candidate if Path(candidate).exists() else None
        )
        if found:
            return Path(found)
    raise SystemExit(
        "no Chromium binary found: install one (`brew install chromium`, "
        "`apt install chromium`) or pass --chromium /path/to/binary"
    )


def shoot(report: Path, png: Path, *, chromium: str | None, width: int) -> None:
    """Render a report card HTML file to a PNG via headless Chromium."""
    binary = find_chromium(chromium)
    png.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(binary),
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            f"--window-size={width},1600",
            f"--screenshot={png}",
            report.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    print(f"wrote {png} ({png.stat().st_size // 1024} KB)")


def _worst_rollout(rollouts: list) -> object:
    scored = [
        (
            float(
                np.mean(
                    np.abs(
                        r.predictions[0].astype(np.float32)
                        - r.ground_truth.astype(np.float32)
                    )
                )
            ),
            i,
        )
        for i, r in enumerate(rollouts)
        if r.ground_truth is not None
    ]
    if not scored:
        raise SystemExit(
            "no rollout carried ground truth; the GIF needs a dataset source "
            "(pass --repo pointing at a LeRobotDataset v3.0)"
        )
    return rollouts[max(scored)[1]]


def build_gif(
    *,
    repo: str,
    n: int,
    context_steps: int,
    horizon: int,
    stride: int,
    panel_width: int,
    gif: Path,
    duration_ms: int,
    colors: int,
) -> None:
    """Write a side-by-side prediction-vs-truth GIF for the worst rollout."""
    from PIL import Image

    from worldproof import rollouts_from_dataset
    from worldproof.baselines import CopyLastFrameBaseline
    from worldproof.sim import LeRobotDatasetSource

    rollouts = rollouts_from_dataset(
        LeRobotDatasetSource(repo),
        CopyLastFrameBaseline(),
        n=n,
        context_steps=context_steps,
        horizon=horizon,
    )
    rollout = _worst_rollout(rollouts)
    predicted, truth = rollout.predictions[0], rollout.ground_truth

    frames = []
    for step in range(0, len(truth), stride):
        pair = np.concatenate(
            [predicted[step], np.full_like(predicted[step][:, :2], 255), truth[step]],
            axis=1,
        )
        image = Image.fromarray(
            pair.squeeze(), mode="L" if pair.shape[-1] == 1 else "RGB"
        )
        scale = (panel_width * 2) / image.width
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)), Image.LANCZOS
        )
        frames.append(image.convert("P", palette=Image.ADAPTIVE, colors=colors))

    gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(
        f"wrote {gif} ({gif.stat().st_size // 1024} KB, {len(frames)} frames, "
        f"left=prediction right=truth)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="LeRobotDataset repo id or local path")
    parser.add_argument("--gif", type=Path, help="output GIF path")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--context-steps", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument(
        "--stride", type=int, default=2, help="keep every Nth predicted step"
    )
    parser.add_argument(
        "--panel-width", type=int, default=240, help="width of each side, in pixels"
    )
    parser.add_argument("--duration-ms", type=int, default=200)
    parser.add_argument(
        "--colors", type=int, default=128, help="GIF palette size; lower is smaller"
    )
    parser.add_argument("--report", type=Path, help="report.html to screenshot")
    parser.add_argument("--png", type=Path, help="output PNG path")
    parser.add_argument("--chromium", help="explicit Chromium binary path")
    parser.add_argument("--shot-width", type=int, default=1200)
    args = parser.parse_args()

    if not args.gif and not args.png:
        parser.error(
            "nothing to do: pass --gif (with --repo) and/or --png (with --report)"
        )
    if args.gif and not args.repo:
        parser.error("--gif needs --repo to read frames from")
    if args.png and not args.report:
        parser.error("--png needs --report pointing at a report.html")

    if args.gif:
        build_gif(
            repo=args.repo,
            n=args.n,
            context_steps=args.context_steps,
            horizon=args.horizon,
            stride=args.stride,
            panel_width=args.panel_width,
            gif=args.gif,
            duration_ms=args.duration_ms,
            colors=args.colors,
        )
    if args.png:
        shoot(args.report, args.png, chromium=args.chromium, width=args.shot_width)


if __name__ == "__main__":
    sys.exit(main())

"""The ``worldproof`` command — two deliberately-separated verbs (SPEC §5).

- ``worldproof evaluate <folder>`` — the laptop-always verb: load a folder of
  rollouts, run the registered metrics (skip-and-report the unsupported ones),
  and write a JSON blob (for CI) and/or a self-contained HTML report card. It
  never runs model inference (CLAUDE.md) — it only scores stored rollouts.
- ``worldproof generate`` — the heavy/occasional verb: run a model + a sim
  oracle to *produce* a folder of rollouts (true futures + predictions), ready
  for ``evaluate``. Pixel models score against the oracle's pixels; latent
  models have the true future encoded into their latent space.

Uses stdlib ``argparse`` only, so it always runs with the core install.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["main"]


def _evaluate(args: argparse.Namespace) -> int:
    from worldproof.adapters import iter_rollouts
    from worldproof.metrics import Capabilities, default_video_extractor
    from worldproof.report import evaluate, report_html, report_json

    folder = Path(args.folder)
    if not folder.exists():
        print(f"no such folder: {folder}", file=sys.stderr)
        return 1
    rollouts = list(iter_rollouts(folder))
    if not rollouts:
        print(f"no rollouts found under {folder}", file=sys.stderr)
        return 1

    capabilities = Capabilities.detect(
        has_tracker=args.tracker,
        fvd_extractor=default_video_extractor() if args.fvd else None,
    )
    report, run_report = evaluate(rollouts, capabilities=capabilities, seed=args.seed)
    print(report.verdict)

    if args.json:
        Path(args.json).write_text(
            json.dumps(report_json(report), indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json}")
    if args.html:
        Path(args.html).write_text(
            report_html(report, rollouts, run_report), encoding="utf-8"
        )
        print(f"wrote {args.html}")
    return 0


def _build_oracle(spec: str):
    """`toy` or `gym:<env-id>` -> a SimOracle."""
    from worldproof.sim import GymSimOracle, ToySimOracle

    if spec == "toy":
        return ToySimOracle()
    if spec.startswith("gym:"):
        return GymSimOracle(spec[len("gym:") :])
    raise ValueError(f"unknown sim {spec!r}. Use 'toy' or 'gym:<env-id>'.")


def _build_model(spec: str):
    """`copy-last-frame`, `action-blind`, or `swm:<checkpoint>` -> an adapter."""
    from worldproof.baselines import ActionBlindBaseline, CopyLastFrameBaseline

    if spec == "copy-last-frame":
        return CopyLastFrameBaseline()
    if spec == "action-blind":
        return ActionBlindBaseline(seed=0)
    if spec.startswith("swm:"):
        from worldproof.adapters import SWMAdapter

        return SWMAdapter(spec[len("swm:") :])
    raise ValueError(
        f"unknown model {spec!r}. Use 'copy-last-frame', 'action-blind', "
        "or 'swm:<checkpoint>'."
    )


def _generate(args: argparse.Namespace) -> int:
    from worldproof.sim import generate_rollouts

    try:
        oracle = _build_oracle(args.sim)
        model = _build_model(args.model)
    except (ValueError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out = Path(args.out)
    written = generate_rollouts(
        oracle,
        model,
        out,
        n=args.n,
        horizon=args.horizon,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(
        f"generated {len(written)} rollout(s) in {out} "
        f"(sim={args.sim}, model={args.model}).\n"
        f"score them with:  worldproof evaluate {out}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worldproof", description="A reality check for world models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="score a folder of rollouts (laptop-runnable)"
    )
    evaluate.add_argument("folder", help="directory of rollouts (see save_rollout)")
    evaluate.add_argument("--json", help="write the JSON report to this path")
    evaluate.add_argument("--html", help="write the HTML report card to this path")
    evaluate.add_argument(
        "--tracker",
        action="store_true",
        help="enable tracker-based invariant metrics",
    )
    evaluate.add_argument(
        "--fvd",
        action="store_true",
        help="compute FVD with the default extractor (needs worldproof[fvd])",
    )
    evaluate.add_argument(
        "--seed", type=int, default=0, help="bootstrap-CI seed (reproducible)"
    )
    evaluate.set_defaults(func=_evaluate)

    generate = subparsers.add_parser(
        "generate", help="produce a folder of rollouts from a sim oracle + a model"
    )
    generate.add_argument(
        "--sim", default="toy", help="ground-truth source: 'toy' or 'gym:<env-id>'"
    )
    generate.add_argument(
        "--model",
        default="copy-last-frame",
        help="model: 'copy-last-frame', 'action-blind', or 'swm:<checkpoint>'",
    )
    generate.add_argument("--out", required=True, help="output directory for rollouts")
    generate.add_argument("--n", type=int, default=8, help="number of rollouts")
    generate.add_argument("--horizon", type=int, default=6, help="predicted steps")
    generate.add_argument(
        "--n-samples", type=int, default=2, dest="n_samples", help="samples per rollout"
    )
    generate.add_argument(
        "--seed", type=int, default=0, help="base seed (reproducible)"
    )
    generate.set_defaults(func=_generate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

# CLAUDE.md — worldproof

## Project context

Read SPEC.md first. It is the scope contract. Do not implement anything
listed under NON-GOALS or the roadmap parking lot unless the user
explicitly asks AND SPEC.md is updated in the same commit.

The one-liner: a diagnosis tool for world models — probes models that
predict the future against ground truth and physical invariants, and
reports where and why the imagination breaks. It does NOT evaluate
policies, planning utility, or aesthetics.

## Environment & commands

- Python 3.10, `uv` for environment management
- Install dev env: `uv sync --all-extras --group dev`
- Run fast tests: `uv run pytest`
- Run metric validation suite (slow): `uv run pytest -m slow`
- Lint & format: `ruff check --fix && ruff format`
- Docs build: `mkdocs serve`

## Packaging & portability

- **uv is the DEV tool only, not a user requirement.** The library ships
  as a standard wheel on PyPI; consumers use plain `pip install
  worldproof` (or uv, or Poetry — their choice) and never know or care
  what we develop with. Never require uv at runtime, never reference uv
  in runtime code, error messages, or user-facing docs beyond the
  contributing guide.
- torch is a loose dependency; never pin a CUDA-specific build.
  Per-platform install instructions live in README, not in deps.
- All video I/O goes through pyav / imageio-ffmpeg (reliable wheels on
  Linux, macOS, and Windows). decord and other Linux-only readers are
  banned.
- Use pathlib everywhere; no `shell=True`, no POSIX-only paths, no
  OS-specific subprocess assumptions.
- CI runs the core test suite on ubuntu-latest, macos-latest, and
  windows-latest. A feature that can't pass on a platform must declare
  a requirement and skip gracefully there (sim oracle: Linux-preferred).

## Hardware rules (this machine is Apple Silicon)

- NO `.cuda()` calls anywhere. Device selection goes exclusively through
  `worldproof/device.py` (cpu / mps / cuda autodetect).
- Core package must import and run with numpy + torch + pillow only.
- Heavy dependencies (trackers, segmenters, judges) are optional extras
  (`pip install worldproof[trackers]`), imported lazily, never at
  module top level.
- If a feature genuinely requires CUDA, it belongs behind a declared
  metric requirement, not an import error. The runner skips-and-reports;
  it never crashes on missing hardware.

## Architecture rules

- New metrics: subclass `Metric` in `metrics/base.py`, register in the
  registry, declare requirements (needs_reference? needs_tracker?
  needs_gpu? needs_sim_oracle?).
- Every new metric MUST ship in the same PR with:
  1. a corruption test (the metric responds to its targeted corruption)
  2. inclusion in the ranking test (real > naive > broken baselines)
  A metric without both is not mergeable.
- Metrics return per-step horizon curves. Scalars are derived in
  `MetricResult`, never stored alone.
- `n_samples` is a first-class runner parameter: metrics must handle
  multi-sample rollouts (best-of-N, mean, spread). Never assume a single
  prediction per context.
- Fidelity metrics must also produce their dynamic-region-masked
  variant when a mask is available; masking logic lives in
  `worldproof/masking.py`, not inside individual metrics.
- `Rollout` and `MetricResult` dataclasses are frozen contracts.
  Changing them requires updating SPEC.md §5 in the same commit.
- The two CLI verbs stay separated: `generate` may be heavy;
  `evaluate` must run on a laptop. Never leak model inference into the
  evaluate path.

## Pinned versions — do not bump casually

- `lerobot==<PINNED_VERSION>` (adapter tested against this only; bumping
  requires re-running the adapter integration tests and a CHANGELOG note)
- Tracker/segmenter checkpoints: <PINNED_TRACKER>, <PINNED_SAM_VARIANT>
  (chosen for confirmed CPU/MPS paths — replacements must prove the same)

## Style

- No comments narrating obvious code. Docstrings on public API only.
- Type hints everywhere on public interfaces.
- Errors must say what to do next ("metric X skipped: requires
  sim oracle; run `worldproof generate --sim ...` first"), not just
  what failed.
- README claims are tested claims: anything stated as working ("runs on
  a MacBook", "10-minute quickstart") has a corresponding CI check or
  scripted verification.

## Don'ts

- Don't add dependencies without flagging it explicitly in the response.
- Don't create new top-level modules; extend the existing structure.
- Don't touch pinned versions (see above).
- Don't implement VLM-judge functionality in core (plugin only, v0.2).
- Don't add policy evaluation, MPC scoring, or leaderboard code —
  SPEC.md non-goals. If asked, point at SPEC.md §3 first.
- Don't "improve" metric math silently: any change to a metric's
  computation bumps its version string in the registry and gets a
  CHANGELOG entry (numbers people cite must be reproducible).

## Working conventions

- New related projects discovered during work go into RELATED.md as one
  line (name, lane, steal/skip) — do not stop to analyze them inline.
- When a coding session goes wrong because of missing context, the fix
  ends with a new line in this file.
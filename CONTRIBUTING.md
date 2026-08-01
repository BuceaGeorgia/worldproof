# Contributing to worldproof

Thanks for looking at this. The short version: the library is small on purpose,
the rules below keep it correct, and every claim in the README is backed by a
test.

## Development setup

Development uses [uv](https://docs.astral.sh/uv/). Consumers of the package
never need it; contributors do.

```bash
uv sync --all-extras --group dev   # install everything, editable
uv run pytest                      # fast tests (slow ones are skipped by default)
uv run pytest -m slow              # the slow suite (downloads models/datasets)
uv run ruff check --fix && uv run ruff format
```

The core package must import and run with numpy, torch, and pillow only.
CI enforces this with a core-only job, so keep heavy imports lazy and behind
extras.

## Rules that are not negotiable

These come from [SPEC.md](SPEC.md) and CLAUDE.md; changes that break them will
not merge.

- **Every new metric ships in the same PR with two tests**: a corruption test
  (the metric responds to its targeted corruption) and inclusion in the ranking
  test (real model beats naive baseline beats broken baseline).
- **Metric math never changes silently.** Any change to a metric's computation
  bumps its `version` string and gets a CHANGELOG entry. Numbers people cite
  must be reproducible.
- **Frozen contracts.** `Rollout`, `MetricResult`, `CounterfactualPair`,
  `VOEPair`, and the on-disk folder format are contracts; changing them
  requires updating SPEC.md section 5 in the same commit.
- **No `.cuda()` calls.** Device selection goes through `worldproof/device.py`
  (cpu, mps, cuda autodetect). A feature that genuinely needs CUDA declares a
  requirement and skips gracefully; it never crashes on missing hardware.
- **Two commands stay separated.** `generate` may be heavy; `evaluate` must run
  on a laptop and never runs a model.
- **Portability.** pathlib everywhere, no `shell=True`, no OS-specific paths.
  CI runs on Ubuntu, macOS, and Windows.
- **Errors say what to do next**, not just what failed.

## Adding a metric

Subclass `Metric` in `worldproof/metrics/base.py`, declare its requirements
(`needs_reference`, `needs_tracker`, ...) and `modality`, register it with
`@register`, and add the two required tests. Set-level measures subclass
`MetricIdentity` and are run by `run_aggregates` instead.

## Before you open a PR

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

Both must be clean. If you touched anything user-visible, update the README;
if you changed behavior, add a CHANGELOG entry.

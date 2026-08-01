# Changelog

Metric computation changes bump the metric's `version` string and are recorded
here (numbers people cite must be reproducible — CLAUDE.md).

## Unreleased

### Changed

- **Pre-packaging design review (session 18).** All API-shape findings fixed
  before first release:
  - Every environment capability now travels through `Capabilities`:
    `Capabilities.detect(has_tracker=..., fvd_extractor=...)`. The separate
    `evaluate(..., fvd_extractor=...)` keyword is gone; `build_report` takes
    `capabilities` instead.
  - One skip record for every metric family: `run_aggregates` returns
    `MetricSkip` and `AggregateSkip` is removed. One identity base for every
    family: `MetricIdentity` (name, version, higher_is_better), shared by
    per-rollout metrics, signature metrics, and set-level measures.
  - Report provenance is now one rule for both layers: `metric_versions`
    records every *configured* measure (set-level included), whether or not it
    produced a result; skip counts merge under one rule with no silent drops.
  - `predict_with_truth` is public (was `_predict_with_truth`, used across
    modules); adapters expose `can_encode` instead of two call sites
    re-deriving it by reflection.
  - Worst-clip selection ranks by a fidelity metric the pixel rollouts
    actually have (psnr, then ssim, then lpips, then any), instead of a
    brittle name heuristic that could silently select no clips.
  - The action-recoverability probe rejects `n_folds < 2` and its
    cross-validation buffer is NaN-filled with a loud internal error instead
    of silently emitting uninitialized values on an impossible path.
  - Removed the dead `worldproof[report]` extra (jinja2 was never used; the
    HTML report is deliberately dependency-free).

### Added

- **FVD default extractor** (`KineticsVideoExtractor`, `default_video_extractor`,
  `worldproof[fvd]`) — a torchvision r3d_18 (Kinetics-400) feature extractor,
  verified to load and run on CPU and MPS, so FVD produces a number instead of
  skipping. Opt in with `worldproof evaluate --fvd` or by passing
  `default_video_extractor()`. It is not the I3D of published FVD (numbers are
  not comparable to those); pass your own extractor for a paper-comparable
  backbone. The model loads lazily and caches. `uv run pytest` now skips slow
  tests by default; the real-download FVD test is marked slow and gated on
  torchvision.
- **`LeRobotDatasetSource`** (`worldproof/sim/lerobot.py`, `worldproof[lerobot-data]`
  extra) — reads **LeRobotDataset v3.0 directly** (per-chunk parquet + mp4, via
  `pyarrow` + `imageio-ffmpeg`) **without the `lerobot` package**, so the entire
  LeRobot Hub (thousands of action-conditioned datasets, incl. Open-X mirrors) is
  usable ground truth on Python 3.10 — sidestepping lerobot's Python-3.12
  requirement. Windows episodes into rollouts for `rollouts_from_dataset`;
  verified end-to-end on real `lerobot/pusht` (`examples/lerobot_demo.py`).
- **Dataset-futures ground-truth provider** (`worldproof/sim/dataset.py`) —
  `DatasetSource` + `rollouts_from_dataset`, the SPEC §5 dataset provider
  alongside the sim oracle. A recorded trajectory window is an `OracleRollout`
  (context + actions + true future), so one dataset drives both pixel and latent
  models through the existing latent bridge. Verified driving the real
  `quentinll/lewm-pusht` checkpoint (`examples/pusht_demo.py`). A bundled real
  PushT dataset source is deferred — loading the trajectories on Python 3.10 hits
  the swm-Lance-format / lerobot-needs-3.12 frictions (see the example's notes).
- **`AtariSimOracle`** (`worldproof/sim/gym.py`, `worldproof[atari]` extra) — a
  sim oracle over `ALE/*` Atari games via `ale-py`: a real, deterministic,
  CPU-runnable pixel emulator (ROMs bundled, no CUDA) that unlocks the signature
  suite (counterfactual divergence, failure faithfulness, fidelity, invariants)
  on visually varied ground truth. Enabling it decoupled `GymSimOracle` from the
  `swm` stack — it now needs only `gymnasium`, best-effort registers the `swm/*`
  and `ALE/*` namespaces if their packages are present, and forwards `env_kwargs`
  to `gymnasium.make` (`AtariSimOracle` defaults `repeat_action_probability=0.0`).
- **Report-layer set-level measures** — `action_recoverability` and `fvd`
  (`worldproof/metrics/aggregate.py`), run over the whole eval set and shown in
  the report card's "Set-level metrics" section and the JSON `aggregates` block.
  - **`action_recoverability` 1.0.0** — the primary latent diagnostic: a ridge
    inverse-dynamics probe fit across the set with K-fold CV (split by rollout),
    reporting the out-of-fold action R² per horizon step from consecutive
    predicted latents. Pure numpy, deterministic, core; higher is better.
  - **`fvd` 1.0.0** — Fréchet Video Distance as a labeled *weak reference*
    (quality, not dynamics). The Fréchet math is pure numpy (no scipy) and
    fully tested; the feature extractor is a plug-point supplied via
    `Capabilities.detect(fvd_extractor=...)`. FVD skips-and-reports when no
    extractor is configured.
- **`worldproof generate`** — the second CLI verb: run a model + a sim oracle to
  produce a folder of rollouts, ready for `evaluate`. Model-agnostic — pixel
  models score against the oracle's pixels; latent models (e.g. `SWMAdapter`)
  get a **latent-ground-truth bridge** (the oracle's true future is encoded
  through the model's own `encode`). Also `generate_rollouts()`,
  `SimOracle.sample_actions()`, and `WorldModelAdapter.encode()`.

### Fixed

- **`calibration_ece` / `calibration_mce` → version 1.1.0** — the calibration
  error compared the ensemble spread (≈σ) against the error of the ensemble
  *mean* (≈σ/√n), two different scales, so a genuinely calibrated stochastic
  model was scored as *increasingly miscalibrated as `n_samples` grew*. Now
  compares the spread against the **per-sample** RMS error (both at the
  single-sample scale); a calibrated ensemble scores ~0 and converges toward 0
  with more samples. (Found in the session-11 review pass.)
- The report card's HTML is now written as UTF-8, and the plain-language verdict
  is ASCII, so `worldproof evaluate --html` no longer crashes on Windows /
  non-UTF-8 locales.
- `RolloutMetadata` accepts numpy-integer `resolution` values (a loader passing
  `frames.shape`-derived numpy ints no longer hits a spurious `ValueError`).
- Rollout folders reload predictions/frames by numeric index, so `n_samples > 99`
  or very long horizons no longer risk lexicographic mis-ordering.

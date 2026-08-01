# worldproof SPEC

> A reality check for world models.

## 1. One line and thesis

worldproof is a diagnosis tool for world models. It probes models that predict
the future of a scene against ground truth and physical rules, and reports where
and why the prediction breaks.

Where it sits:

| Lane | Question | Owned by |
|---|---|---|
| Policy benchmarks | Does the acting model succeed at tasks? | VLABench, LIBERO |
| World-model platforms | Can the model be trained and used for planning? | stable-worldmodel |
| **Rollout faithfulness** | **Is the prediction correct? Where does it fail, and why?** | **worldproof** |

Why this lane is real:

- PSNR, FVD, and LPIPS correlate poorly with control utility. Pixel quality and
  action relevance are largely separate (arXiv 2606.07687).
- "The evaluation landscape for world models remains fragmented," from the
  definition and roadmap survey (arXiv 2607.06401), which also names uncertainty
  calibration as essential for safety-critical use.
- Existing protocols "fail to assess coherent causal dynamics and consistency
  across long temporal spans" (PAN, arXiv 2511.09057).
- Labs hand-roll their own versions of these metrics per paper (for example
  World-VLA-Loop's success/failure alignment rate, which is our failure
  faithfulness).
- Benchmark repos keep adding "evaluate your own videos" modes with partial
  support (VBench custom_input). People want a tool. Benchmarks can only half
  become one.

## 2. Users and use cases

1. Researcher: is my world model learning dynamics, or just rendering plausible
   pixels?
2. Practitioner in CI: which checkpoint regressed on physics? Nightly runs on
   every checkpoint, JSON output, no GPU.
3. Reviewer or content loop: a report card for model X within 48 hours of a
   release.

## 3. Scope contract

worldproof evaluates any model that implements:

```python
predict(context, actions, horizon, n_samples=1) -> future
```

where `future` is pixels or latents.

In scope (v0.1):

- Pixel-space action-conditioned world models
- Latent (JEPA-family) world models, through the latent track
- Pre-generated rollout folders (no model required)

Non-goals (v0.1). Do not implement without changing this file first:

- Policies and action-outputting models (that is VLABench's lane). Policies show
  up here only as test drivers that supply realistic action sequences.
- MPC and planning-utility scoring (stable-worldmodel's lane)
- Aesthetic or visual-appeal scoring
- Hosted leaderboards
- Text-to-video without action conditioning (a v0.2 wedge)

## 4. Metrics (v0.1)

Every metric returns a per-step curve over the horizon, a derived scalar, and
artifact paths. Every metric ships with (a) a corruption test it responds to and
(b) a ranking test it passes: real model beats naive baseline beats broken
baseline. The corruption suite includes temporal corruptions (local and global
frame swap, frame interleaving, from Unterthiner et al. 2018), not just spatial
noise, so a metric has to prove it tracks dynamics, not just frame quality.

Stochasticity is first class. The runner supports `n_samples` and reports
best-of-N and spread (the spread feeds calibration). Scores are aggregated across
rollouts with the interquartile mean (IQM) and stratified bootstrap confidence
intervals, never a bare mean and standard deviation (Agarwal et al. 2021). See
section 6.

Two layers. Per-rollout metrics produce `MetricResult` curves. Set-level
measures (IQM, performance profiles, FVD, the action-recoverability probe)
summarize whole sets of rollouts and are not per-rollout `MetricResult`s. Each
metric declares a `modality` (pixels, latents, or any), and the runner skips a
metric on a mismatched rollout with a clear reason.

### Tier 1: fidelity against ground truth

- PSNR, SSIM, LPIPS as curves over the horizon.
- FVD (a weak reference. v0.1: shipped, with a default extractor). The
  plug-point and the Frechet math ship, plus a default extractor (torchvision
  r3d_18, Kinetics-400) behind the `fvd` extra, opt in with `evaluate --fvd`.
  Reported for continuity with standard video metrics. It matches
  human quality judgment better than PSNR or SSIM (Unterthiner et al. 2018) but
  is action-blind and separate from control utility (arXiv 2606.07687). It is a
  distribution-level scalar over a set of rollouts, so it lives at the set-level
  layer, not as a per-step curve. The report labels it "quality, not dynamics."
- Dynamic-region variants. Every fidelity metric is also reported on the moving
  regions only, so a static background cannot dominate the score.
- Latent track. Action relevance is primary, not decodability alone:
  - Action-recoverability probe (the main latent check. v0.1: shipped). Fit a
    linear or MLP inverse-dynamics probe to recover the actions from consecutive
    predicted latents, and report action R2 over the horizon. This tests whether
    the latent space is organized around the variables you can control.
    Perceptual quality, and even state decodability, can be high while action
    recoverability is near zero (Yeom et al. 2026, arXiv 2606.07687). The probe
    is fit across the whole eval set at the set-level layer, since one rollout
    has too few transitions and overfits, so it ships with the report card, like
    FVD, not as a per-rollout metric.
  - Latent prediction error (the per-rollout latent metric. v0.1: shipped).
    Per-step mean squared error between predicted and ground-truth latents.
    Probing the decodability of task-relevant state needs a state-label source
    that the frozen `Rollout` does not carry, so it is deferred.

### Signature: the moat (needs the sim oracle or curated episodes)

- Calibration (v0.1: shipped). Does the spread across samples predict the actual
  error? Reported as expected and maximum calibration error (ECE and MCE) per
  step. Guo et al. 2017 defines ECE and MCE for classification. Rollouts are
  continuous, so we use the regression-uncertainty version (Levi et al. 2019):
  bin elements by predicted uncertainty, then compare root-mean variance to RMSE
  per bin. Needs `n_samples >= 2` (a `needs_multisample` requirement) and a
  reference. Quantile and coverage calibration (Kuleshov et al. 2018) is the
  alternative reading, parked.
- Counterfactual divergence (v0.1: shipped, curated path). Same context, two
  action sequences. Consumes a `CounterfactualPair` (section 5): two rollouts
  matched by `context_id`, with different actions and ground-truth futures on
  both sides. The predicted contrast (the difference of the two sample-mean
  predictions) is compared to the true contrast as the mean absolute error per
  step (lower is better). This checks both the size and the direction of the
  causal effect, so an action-blind model, whose predicted contrast is about
  zero, gets caught. Pairs come from curated data or from the sim oracle (below,
  shipped) through `make_counterfactual_pair`, which produces true futures for
  any action sequence.
- Failure faithfulness (v0.1: shipped, VOE mode). On curated failure episodes,
  does the model reproduce the failure or imagine success? Consumes a `VOEPair`
  (section 5): a rollout whose ground truth is the real (failure) outcome, plus a
  physically plausible `success_reference`. The score is error(pred, success)
  minus error(pred, failure) per step (mean absolute, higher is better). A
  positive value means the model imagines the real failure more accurately than
  the plausible alternative (violation of expectation, IntPhys). VOE episodes
  come from curated data or the sim oracle through `make_voe_pair`, which
  generates the failure and success futures and labels `is_failure`. The binary
  reproduce-or-hallucinate classifier mode (which queries oracle state directly)
  is deferred to v0.2.

### Invariants: deterministic, tracker-based

v0.1: shipped, on a core reference tracker (`worldproof/tracking.py`,
`BlobTracker`, a numpy connected-components tracker, deterministic, for clean
scenes). A stronger segmenter and point tracker for messy real video slots in
behind the same `Tracker` interface through a `trackers` extra (the pins in
CLAUDE.md are still TBD). Both declare `needs_tracker` and assume a closed scene.

- Object-count conservation (reference-free). How far the predicted per-frame
  object count drifts from the starting count over the horizon (lower is better).
- Object permanence through occlusion (reference-based). The fraction of
  ground-truth objects the prediction keeps per step, matched by position. An
  object that reappears in the truth after occlusion has to reappear in the
  prediction (higher is better).

Left out of core on purpose: a VLM as judge (an optional plugin, v0.2). It is
expensive, slow, and not reproducible, which makes it bad for CI.

## 5. Interfaces

Two commands, kept separate on purpose. Both use the standard library argparse,
so they run on the core install.

- `worldproof evaluate <folder>` (v0.1: shipped). Scores a folder of rollouts.
  Runs on CPU or MPS, always, and "runs on a laptop" is a tested claim. Runs the
  registered metrics, skips and reports the ones it cannot run, and writes
  `--json` (for CI) and `--html` (the report card). Never runs a model.
- `worldproof generate --sim <toy|gym:ENV> --model <copy-last-frame|action-blind|swm:CKPT> --out FOLDER`
  (v0.1: shipped). Runs a model and a sim oracle to produce a folder of rollouts
  (true futures, predictions, random seeded actions), ready for `evaluate`. Pixel
  models score against the oracle's pixels. Latent models have the true future
  encoded into their latent space through `adapter.encode` (the latent bridge).
  This one may be heavy and is run occasionally. Generating counterfactual-pair
  and VOE folders (for the signature metrics) and a dataset source are follow-ons.

Frozen data contracts. Changing any of these requires editing this file in the
same commit:

- `Rollout`: context frames or latents, the action sequence, one or more
  predictions, an optional ground truth, and metadata (fps, resolution, model id,
  seed, camera_id, lossy_source, inference_latency_s, context_id, is_failure).
- `MetricResult`: the per-step curve, a summary scalar, artifact paths, and
  requirements-met flags.
- `CounterfactualPair`: two rollouts matched by `context_id`, with different
  actions and ground truth on both. The input to counterfactual divergence.
- `VOEPair`: a rollout (its ground truth is the real outcome) plus a
  `success_reference` future. The input to failure faithfulness.

Adapters (v0.1):

- stable-worldmodel-trained models (swm as the substrate, datasets through its
  format registry).
- LeRobot world models: not supported as model adapters (see the decision log
  and RELATED.md). The LeRobot data path is supported through
  `LeRobotDatasetSource`.
- Folder of rollouts (no model needed).

Ground-truth providers:

- Dataset futures. Two provider pieces are shipped: `DatasetSource` plus
  `rollouts_from_dataset` (the general provider), and `LeRobotDatasetSource`,
  which reads LeRobotDataset v3.0 directly (parquet and mp4, no `lerobot`
  package, runs on Python 3.10).
- Sim oracle (v0.1: shipped), in `worldproof/sim/`. A `SimOracle` steps a
  simulator to produce true futures for any action sequence, is deterministic in
  `(seed, actions)`, and labels outcomes. Implementations: `ToySimOracle` (a pure
  numpy point-mass, core, runs everywhere) and `GymSimOracle` (any
  pixel-rendering gym env: `swm/*` behind the `swm` extra, for example
  `swm/TwoRoom-v1`, and `ALE/*` Atari games through `AtariSimOracle` behind the
  `atari` extra, a deterministic CPU emulator with bundled ROMs). Both are real
  and laptop-runnable, and imported lazily. `make_counterfactual_pair` and
  `make_voe_pair` build the signature-metric inputs from an oracle and a pixel
  adapter. This is the thing no fixed dataset can do. ManiSkill and Isaac Sim
  (CUDA and Linux) are deferred behind the same interface to a later extra, and
  skip and report off-platform. The v0.2 failure-faithfulness classifier mode
  will use the oracle's richer state queries (for example `is_grasped`).

Outputs (v0.1: shipped, `worldproof/report.py`). One self-contained HTML report
card (inline SVG curves, base64 worst-N clips, a plain-language verdict, a
latency table, and a "skipped and why" table) plus a JSON blob for CI. Both are
produced on the evaluate path with core dependencies only. Cross-rollout
aggregation is robust (IQM, seeded bootstrap CIs, performance profiles), never a
bare mean and standard deviation. Latency is a reported column, since in dynamic
control inference speed matters, with an optional user-set budget flag that CI
can enforce. Latency is a reported signal and an opt-in gate, never a hard-coded
pass or fail (that would break for people profiling unoptimized prototypes, and
certifying policies or safety is a non-goal, section 3).

### Decision log for the frozen section 5 contracts

Recorded 2026-07-18 (session 1: contracts and package skeleton). These resolve
ambiguities left open above. Changing any of them is a contract change that needs
an edit here in the same commit.

- Array storage. Pixels are numpy `(T, H, W, C)` uint8 with C in {1, 3}. Latents
  are numpy float32 `(T, *latent_dims)`. The leading axis is always time.
- Modality. One `context`, `predictions`, and `ground_truth` field per role,
  discriminated by a `modality` flag (`"pixels"` or `"latents"`), not separate
  pixel and latent fields.
- Latent track: pixel-in models store latent context (recorded 2026-07-25,
  session 2). A latent world model (JEPA or DINO-WM family) takes pixels but
  predicts latents. Its adapter encodes the pixel context with the model's own
  encoder and emits a uniformly latent `Rollout` (context, predictions, and
  ground truth all latent). Raw pixels are not stored, so a `Rollout` never mixes
  modalities and the single `modality` flag holds. The reason: evaluate a latent
  model in the space it predicts in, which is decoder-independent (most
  checkpoints ship no decoder) and cheap (no decode in `evaluate`). This upholds
  the thesis that pixel fidelity is not the same as dynamics correctness (section
  1) and is the rule every future latent adapter follows. Guardrail: decoding
  latents to pixels for visual reports or pixel metrics is an opt-in bridge that
  produces a separate pixel-modality `Rollout` or report artifact, never a mixed
  `Rollout` and never a change to this contract. Accepted cost: latent report
  cards are curves and numbers, not embedded clips, until that bridge exists.
- Samples. `predictions` is a sequence, one array per sample. `n_samples` is
  first class and at least 1.
- Metric curves. `MetricResult.curve` is per-sample, shape
  `(n_samples, horizon)`. Best-of-N, mean, spread, and the summary scalar are
  derived, never stored alone.
- Action alignment. The `actions` leading axis always equals `horizon` (one entry
  per predicted step). Sub-stepped control is expressed as a richer per-step
  shape, for example `(horizon, substeps, action_dim)`. The length equality is
  never relaxed, because counterfactual divergence relies on 1:1 step alignment
  with the sim oracle.
- Summary scalar. The default is the mean over the horizon of the sample-mean
  curve (kept as the reproducible default so cited numbers do not drift). v0.1
  also offers a selectable progressive-penalty reduction that up-weights
  late-horizon steps, since late quality matters most for planning under
  compounding error (PAN, arXiv 2511.09057, and consistency metrics per Duan et
  al. 2025, WorldScore). Other reductions (final step, area under curve, worst
  step) stay parked for v0.2.
- Score direction. `MetricResult.higher_is_better` keeps serialized results self
  describing. The authority is the `Metric` subclass (a required class attribute
  from session 2, in `metrics/base.py`), and the runner stamps it onto each
  `MetricResult`.
- Channels. Pixel frames have 1 or 3 channels.
- Metadata. `RolloutMetadata` carries `fps`, `model_id`, `seed`, `resolution`
  (required for pixels, validated against the frames), `camera_id` (v0.1: one
  camera per `Rollout`, many rollouts per episode, the field that distinguishes
  two views of one episode), and `lossy_source` (bool, default False. True when
  frames were decoded from a lossily re-encoded source, so reports can note that
  fidelity scores are polluted by codec artifacts), plus `inference_latency_s`
  (float or None, default None. Wall-clock seconds the model took to produce the
  predictions, stamped by the adapter at generate time, so the report card can
  show latency without re-running the model on the laptop-only evaluate path),
  `context_id` (str or None, the counterfactual-pair matching key), and
  `is_failure` (bool or None, which marks curated failure episodes and lets
  reports slice failure-only fidelity).
- Research-pass scope (recorded 2026-07-25, session 3). The v0.1 metric set was
  extended after a 16-source review. No frozen-contract change (`Rollout` and
  `MetricResult` untouched, the additions are new metrics and set-level
  aggregation). Adopted into v0.1: the action-recoverability probe as the main
  latent check (section 4), FVD as a weak reference (set-level, action-blind),
  calibration as ECE and MCE, robust aggregation (IQM, stratified bootstrap CIs,
  performance profiles) as the standard (section 6), temporal corruptions in the
  suite (section 4), and a selectable progressive-penalty summary. Kept off the
  headline on purpose: FVD stays a weak reference (it contradicts the thesis as a
  primary), and latency is a reported column plus optional budget, never a hard
  gate.
- Tier-1 metrics shipped (recorded 2026-07-25, session 4). No frozen-contract
  change (new metrics plus a `Metric.modality` flag on the metric base class, not
  on `Rollout` or `MetricResult`). Registered per-rollout: `psnr`, `ssim`, and
  their `@dynamic` variants (pure numpy, core, always run; SSIM cross-checked
  against scikit-image to machine precision), `lpips` and `lpips@dynamic`
  (optional `fidelity` extra, `lpips` imported lazily so it skips cleanly when
  absent), and `latent_prediction_error` (latent modality). Dynamic-region
  masking lives in `worldproof/masking.py` (a temporal-difference mask from the
  reference). Each metric ships a corruption test (spatial noise plus temporal
  frame swap) and passes the ranking test. The action-recoverability probe and
  FVD stay at the set-level layer, built with the report card.
- Signature: calibration shipped (recorded 2026-07-25, session 5). No
  frozen-contract change (a new metric plus a `needs_multisample` flag on the
  metric base class). `calibration_ece` and `calibration_mce` (regression
  uncertainty, Levi et al. 2019, with `needs_reference` and `needs_multisample`),
  with corruption (induced overconfidence) and ranking tests. Counterfactual
  divergence and failure faithfulness were pending here, since they needed
  infrastructure that did not exist yet (a pair abstraction, the sim oracle, and
  a failure-detection mechanism), so they were left out rather than faked.
- Signature: counterfactual and failure faithfulness shipped (recorded
  2026-07-25, session 6). Frozen-contract change: two new input objects,
  `CounterfactualPair` and `VOEPair` (section 5), and two `RolloutMetadata`
  fields (`context_id`, `is_failure`). These are multi-arity metrics (pair or VOE
  input), so they sit outside the per-rollout `Metric` hierarchy and are driven by
  a `SignatureRunner` (skip and report, reusing the same report types).
  `counterfactual_divergence` is the mean absolute difference of the contrasts per
  step (curated pairs, lower is better). `failure_faithfulness` is error(pred,
  success) minus error(pred, failure) per step (VOE mode, higher is better). Both
  use the sample-mean prediction, and both ship corruption and ranking tests
  (synthetic perfect, partial, and broken, since the fidelity baselines are all
  action-blind here). The sim-oracle and classifier modes stayed for later. VOE
  moved from the v0.2 parking lot into this v0.1 mechanism.
- Sim oracle shipped (recorded 2026-07-25, session 7). No frozen-contract change
  (a new `worldproof/sim/` subpackage, the `Rollout`, `MetricResult`, and pair
  contracts untouched). A `SimOracle` protocol and `OracleRollout`. `ToySimOracle`
  (a pure numpy point-mass, core, deterministic, collision equals failure) and
  `GymSimOracle` (stable-worldmodel gym envs, behind the `swm` extra, imported
  lazily, verified end to end on `swm/TwoRoom-v1` on Apple Silicon, CPU,
  deterministic). `make_counterfactual_pair` and `make_voe_pair` close the loop
  from an oracle and pixel adapter to the pair contracts. Verified that an
  action-blind model gets flagged by counterfactual divergence on both the toy
  and the real env. The backend chosen was swm gym envs (run and verified on this
  machine) over ManiSkill or SAPIEN (CUDA and Linux, unverifiable here).
  ManiSkill deferred behind the same interface.
- Invariants shipped (recorded 2026-07-25, session 8). No frozen-contract change
  (new metrics plus `worldproof/tracking.py`, `Rollout` and `MetricResult`
  untouched). `object_count_conservation` (reference-free, count drift from the
  starting count) and `object_permanence` (reference-based, the fraction of
  ground-truth objects kept per step). Both are `needs_tracker`, pixel modality,
  closed scene, with corruption and ranking tests. The backend is a core numpy
  `BlobTracker` (connected components) so both run and are verified on the core
  install. The stronger tracker for messy video is deferred behind a `trackers`
  extra (the CLAUDE.md pins are still placeholders, not invented). Permanence was
  chosen reference-based (ground truth defines which objects persist) over a
  noisier reference-free proxy.
- Report card and evaluate CLI shipped (recorded 2026-07-25, session 9). No
  frozen-contract change (`worldproof/report.py` and `worldproof/cli.py`). The
  set-level layer: IQM, seeded bootstrap CIs, and performance profiles over a
  folder's per-rollout results, plus latency columns, skip transparency, and a
  plain-language verdict, producing JSON (for CI) and one self-contained HTML
  report card (inline SVG curves and base64 worst-N clips, no external resources,
  core deps only). `worldproof evaluate <folder> --json --html` loads rollouts and
  scores them, never running a model (the evaluate and generate split). At this
  point `worldproof generate` was a documented stub. The HTML was chosen core and
  self-contained (no jinja2) so the report always renders on a laptop.
- Last mile: README and a verified quickstart (recorded 2026-07-25, session 10).
  No frozen-contract change. `README.md` plus the pyproject `readme` field.
  `examples/quickstart.py` is self-contained (toy oracle and baseline, generate,
  evaluate, JSON and HTML, core deps, no download) and is exercised by
  `tests/test_quickstart.py`, so "runs on a laptop" is a CI-tested claim on all
  three operating systems. Added `make_rollout` (the single-rollout analog of the
  pair builders). Chose the self-contained toy quickstart over a real-checkpoint
  or bundled-dataset one so it stays core, fast, and offline.
- Review-pass fixes (recorded 2026-07-25, session 11). A 5-agent correctness
  review. Findings verified and fixed. High: (1) `calibration_ece` and
  `calibration_mce` compared the ensemble spread against the error of the
  ensemble mean (two different scales), so a calibrated model scored worse as
  `n_samples` grew; now per-sample RMSE (metric version 1.1.0, see CHANGELOG).
  (2) The HTML report was written without `encoding="utf-8"` and crashed on
  Windows and non-UTF-8 locales (the up and down arrow glyphs); writes are now
  UTF-8 and the printed verdict is ASCII. Medium: `RolloutMetadata` accepts numpy
  integer `resolution`. Low: rollout folders reload by numeric index (for more
  than 99 samples), `dynamic_region_mask` rejects non-uint8, and `bootstrap_ci`
  is computed once. No frozen-contract semantics changed.
- The generate command shipped (recorded 2026-07-26, session 12). No
  frozen-contract change (`worldproof/cli.py` and `sim/build.py`). Model
  agnostic: a new optional `WorldModelAdapter.encode(frames)` lets a latent model
  attach latent ground truth by encoding the oracle's true future (the latent
  bridge, chosen over a pixel-only first cut). The base `_prepare_context` now
  auto-encodes for latent adapters, so a latent model implements just `encode`.
  `SimOracle.sample_actions` (random seeded actions) and `generate_rollouts` back
  the command. Counterfactual-pair and VOE-folder generation, plus a dataset
  source, were deferred.
- Set-level measures shipped: FVD and action recoverability (recorded 2026-07-26,
  session 13). No frozen-contract change (`Rollout` and `MetricResult` untouched,
  a new `worldproof/metrics/aggregate.py` set-level layer plus an additive
  `Report.aggregates` field on the non-frozen report model). These consume the
  whole eval set, not one rollout, so they sit outside the per-rollout `Metric`
  hierarchy and are run by `run_aggregates` (skip and report, mirroring
  `MetricRunner`) as a report-layer consumer in `report.py`.
  `action_recoverability` (the main latent check, section 4) is a ridge inverse
  dynamics probe fit across the set with K-fold cross-validation (folds split by
  rollout so nothing leaks), reporting out-of-fold action R2 per step from
  consecutive predicted latents. Pure numpy, deterministic, core (a linear probe
  is the default, MLP deferred). It ships with corruption (shuffle actions, R2
  drops to zero) and ranking (structured beats noise) tests. `fvd` (a weak
  reference, section 4): the Frechet-distance math is pure numpy (a symmetric
  eigendecomposition matrix square root, no scipy) and fully tested; the I3D
  feature extractor is a plug-point (`VideoFeatureExtractor`), passed through
  `evaluate(..., fvd_extractor=...)`. The hybrid choice: ship the plug-point and
  math now, defer the pinned default I3D behind a `worldproof[fvd]` extra, since
  its checkpoint pin has to prove a CPU or MPS path on target hardware first, so
  it is not invented here. With no extractor, FVD skips and reports rather than
  emit an un-anchored number. Both show up in the report card's "Set-level
  metrics" section and the JSON `aggregates` block.
- Atari sim-oracle backend shipped (recorded 2026-07-26, session 14). No
  frozen-contract change (`worldproof/sim/gym.py` plus a `worldproof[atari]`
  extra, the `SimOracle` and `OracleRollout` contracts untouched).
  `AtariSimOracle` wraps `ALE/*` gym games through `ale-py`: a real,
  deterministic, CPU-runnable pixel emulator (ROMs bundled, no CUDA), far more
  varied than PushT, which unlocks the full signature suite on a laptop. A recon
  spike confirmed a clean drop-in and, correcting the pre-recon assumption, that
  rollouts are deterministic in `(seed, actions)` even with default sticky
  actions, because `reset(seed=...)` reseeds the emulator. The backend still
  defaults `repeat_action_probability=0.0` so the requested action is applied
  faithfully (which keeps counterfactual contrasts clean). This decoupled
  `GymSimOracle` from the swm stack: it now requires only `gymnasium`, best-effort
  registers the `swm/*` and `ALE/*` namespaces if present, and forwards
  `env_kwargs` to `gymnasium.make`. ManiSkill and Isaac Sim stay deferred behind
  the same interface.
- Dataset-futures provider shipped (recorded 2026-07-26, session 15). No
  frozen-contract change (a new `worldproof/sim/dataset.py`, the `Rollout` and
  `OracleRollout` contracts untouched). The second ground-truth provider
  alongside the sim oracle (section 5): a `DatasetSource` yields recorded
  trajectory windows as `OracleRollout`s (a dataset window is a true rollout:
  context, actions, true future), and `rollouts_from_dataset(source, adapter, ...)`
  runs a model on each window and attaches the real future as ground truth
  through the existing latent bridge, so one dataset drives pixel and latent
  models. Verified end to end driving the real `quentinll/lewm-pusht` checkpoint
  (a gated test plus `examples/pusht_demo.py`, producing a real report card). The
  reusable provider is what shipped; a bundled real PushT dataset source is not,
  because loading the trajectories on Python 3.10 hit three frictions: (1) raw
  `lerobot/pusht` is not swm's Lance or HDF5 format; (2) the `lerobot://` reader
  needs `lerobot`, which requires Python 3.12; (3) the checkpoint consumes 10-dim
  frameskip-stacked actions, so a source has to reproduce swm's action encoding.
  The clean hand-off is a swm-native Lance or HDF5 PushT dataset loaded on 3.10,
  documented in the example, deferred until that dataset id is confirmed.
- LeRobotDataset reader shipped (recorded 2026-07-26, session 16). No
  frozen-contract change (a new `worldproof/sim/lerobot.py` plus a `lerobot-data`
  extra). `LeRobotDatasetSource` is the concrete `DatasetSource` that unblocks the
  real-data flagship. It reads LeRobotDataset v3.0 directly: per-chunk parquet
  (low-dim columns: `action`, `next.success`, episode index) and per-chunk mp4
  (camera frames, keyed by `from_timestamp`), through `pyarrow` and
  `imageio-ffmpeg`, without the `lerobot` package, so it runs on the supported
  Python 3.10 floor. It windows episodes into `OracleRollout`s that feed
  `rollouts_from_dataset`. This makes the LeRobot Hub (thousands of
  action-conditioned datasets, including Open-X mirrors) usable ground truth.
  Verified end to end on the real `lerobot/pusht` (206 episodes, 96 by 96; the
  copy-last-frame baseline shows the expected signature: high global PSNR, a
  collapsing dynamic-region SSIM). Tested against a synthesized real-v3-layout
  dataset (no network) plus a slow real-download smoke test. Chosen as the launch
  data story over the swm-PushT-checkpoint path, which is blocked by the session
  15 frictions. v2.x LeRobot layouts and non-video (parquet-image) datasets are
  follow-ons. mp4 frames are lossy, which is a fidelity caveat.
- FVD default extractor shipped (recorded 2026-07-31, session 17). No
  frozen-contract change (a new `KineticsVideoExtractor` and
  `default_video_extractor` in `worldproof/metrics/aggregate.py`, a `fvd` extra,
  and an `evaluate --fvd` flag). This promotes FVD's default extractor out of the
  session-13 deferral. A spike verified that a torchvision r3d_18 pretrained on
  Kinetics-400 loads and runs on both CPU and MPS on Apple Silicon, and produces
  a monotonic Frechet distance through the shipped math (identical clips give 0,
  corruption raises it), so the "prove a CPU/MPS path" rule is met. It is opt in,
  not automatic: torchvision also arrives with the `fidelity` extra, and the
  model is a ~120 MB download, so running it on the light evaluate path only
  happens when the user passes `--fvd` or `default_video_extractor()`. The
  extractor loads the model lazily on first use and caches it. Honest caveat:
  r3d_18 is not the I3D of most published FVD, so numbers are not comparable to
  those; a user can pass their own extractor for a paper-comparable backbone, and
  FVD stays labeled a weak reference regardless. Testing: light wiring tests run
  in core CI (no download, lazy construction), and the real-extraction test is
  marked slow and gated on torchvision. `uv run pytest` now skips slow tests by
  default (matching the fast/slow split), so neither core CI nor a normal local
  run downloads the model.
- Pre-packaging design review fixed (recorded 2026-08-01, session 18). No
  frozen-contract change (`Rollout`, `MetricResult`, and the pair contracts are
  untouched); this hardens the not-yet-released API shape before packaging. A
  whole-project review found eight design problems; all fixed. (1) The dead
  `worldproof[report]` extra (jinja2, never used) is deleted. (2) Every
  environment capability now travels through `Capabilities`
  (`detect(has_tracker=..., fvd_extractor=...)`); the separate
  `evaluate(..., fvd_extractor=...)` keyword is removed and `build_report` takes
  `capabilities`. The typing-only import in the runner avoids a cycle with the
  aggregate module. (3) One identity base, `MetricIdentity` (name, version,
  higher_is_better, validated at class creation), is shared by all three metric
  families, and `run_aggregates` reports skips as `MetricSkip` (`AggregateSkip`
  deleted), so the skip record is one type everywhere. (4) Report provenance is
  one rule for both layers: `metric_versions` records every configured measure
  whether or not it produced a result, skip counts merge under one rule with no
  silent drops, and the verdict is derived from the built `Report` rather than a
  six-argument helper. (5) `predict_with_truth` is public (it was a
  private-by-name function imported across modules). (6) Adapters expose
  `can_encode`, replacing duplicated reflection in the base class and the
  builders. (7) Worst-clip selection ranks by a fidelity metric the pixel
  rollouts actually have (psnr, ssim, lpips, then any), fixing a silent
  no-clips corner on mixed-modality sets (regression test added). (8) The
  action-recoverability probe rejects `n_folds < 2` and its cross-validation
  buffer is NaN-filled with a loud internal error on the impossible empty-fold
  path, instead of silently emitting uninitialized memory.

### Folder-of-rollouts on-disk format (a public contract)

Recorded 2026-07-18 (session 2: adapters). The folder source
(`worldproof/adapters/folder.py`, `format_version` `"0.1-provisional"`) loads and
saves one rollout per directory. Changing this layout is a contract change that
needs an edit here.

- One directory is one rollout is one camera. Many rollouts per episode are
  sibling directories. `iter_rollouts(root)` walks them and skips any without a
  manifest.
- `manifest.json` (required) records `format_version`, `modality`, `storage`
  (`"png_sequence"`, `"npy"`, or `"video"`), `fps`, `resolution` (`[H, W]` or
  null), `channels` (pixels only), `model_id`, `seed`, `camera_id`,
  `lossy_source`, `inference_latency_s`, `horizon`, `n_samples`, and
  `has_ground_truth`. `load_rollout` rejects a `format_version` mismatch.
- Lossless canonical storage (a metric-correctness decision, not just
  convenience). Pixels are zero-padded PNG sequences (`context/`,
  `predictions/sample_NN/`, optional `ground_truth/`). Latents are `.npy`
  (`context.npy`, `predictions/sample_NN.npy`, optional `ground_truth.npy`).
  `actions.npy` is always present. This guarantees an exact round trip, so
  fidelity metrics score the generated numbers, not codec artifacts.
- Video is read-only, lossy, and opt-in (`storage == "video"`), decoded behind the
  `io` extra. Such rollouts load with `lossy_source = True`.

## 6. Design principles

1. Device-aware graceful degradation. Each metric declares its requirements, the
   runner runs what the machine supports, and it reports what it skipped and why.
   A MacBook user gets a partial but real report, never an install error.
2. Deterministic core. Reproducible numbers. Anything judge-based is an optional
   plugin.
3. Reports are the product. Scores are inputs to reports.
4. Curves, not scalars. Degradation over the horizon is the practical question.
   Scalars are derived, never stored alone.
5. Wrap benchmarks, do not reimplement them. Published metric suites are allies
   to wrap, and their papers are the citation trail.
6. Robust aggregation, not point estimates. Across rollouts, report the
   interquartile mean, stratified bootstrap confidence intervals, and performance
   profiles, never a bare mean and standard deviation, which outliers dominate on
   the heavy-tailed distributions typical of world-model tasks (Agarwal et al.
   2021). This is a report-layer decision; per-rollout curves are unchanged.

## 7. Roadmap parking lot

Everything tempting goes here, not into v0.1.

- v0.2: a VLM-judge plugin; a video-only (action-free) wedge with the
  reference-free subset; more invariants (contact consistency, size constancy,
  object-relationship preservation); alternative summary reductions (final step,
  area under curve, worst step, best sample then mean); SSC and FSC conformal
  coverage diagnostics and a temperature-scaling calibration utility (Guo et al.
  2017); a transition-smoothness temporal invariant (optical-flow acceleration);
  task-grounded unit-test datasets that isolate long-term memory and relational
  reasoning (StarCraft2-Videos style, Unterthiner et al. 2018). The
  action-recoverability probe and IDM probing were promoted to v0.1 (section 4,
  latent track). Violation of expectation was promoted to v0.1 as the
  failure-faithfulness mechanism (section 4, signature).
- v0.3: a policy-in-imagination correlation track (success rate inside the world
  model versus inside the simulator, WorldArena's protocol, productized); a safety
  probe track (instruction injection, distribution-shift stress tests,
  reachability analysis proving a model never predicts states that violate safety
  boundaries); Isaac Sim as an extra, heavy, Linux and CUDA sim-oracle provider;
  the action-conditioning study (frozen backbones with and without training-time
  action conditioning, Yeom et al. 2026, Appendix B) as a documented recipe, not a
  core metric, since worldproof scores rollouts and does not train.
- v0.x: multi-view evaluation (cross-camera consistency, keyed on
  `RolloutMetadata.camera_id`); a concrete metric, MEt3R (a DUSt3R warp of one
  generated view into another for pose-free 3D consistency).
- v0.2: consider a `RolloutSource` protocol if the runner grows source-streaming
  needs (today the runner consumes `Rollout` objects, not sources, so the
  abstraction has no consumer yet).
- Infra (any version): an in-memory tensor fast path for in-training-loop
  evaluation (the storage contract is unchanged); when the first GPU adapter
  lands, consider an optional batched hook (`_predict_futures_batched`) the
  adapter base prefers if implemented. Do not add it speculatively.
- Someday: an swm success-rate panel in the report card; MechVerse-style kinematic
  checks; driving-domain invariants.

## 8. Related work map

See RELATED.md, one line per project: name, lane, steal or skip. New finds go
there and are reviewed in a batch. They do not interrupt the build.

# RELATED.md

One line per related project: name, lane, and whether to steal from it or skip
it. New finds get appended to the log and reviewed in a batch. They do not
interrupt the build. See the working conventions in CLAUDE.md.

## The three lanes (SPEC.md section 1)

| Lane | Question | Owned by |
|---|---|---|
| Policy benchmarks | Does the acting model succeed at the task? | VLABench, LIBERO |
| World-model platforms | Can the model be trained and used for planning? | stable-worldmodel |
| Rollout faithfulness | Is the prediction correct, and where does it break? | **worldproof** |

## Log

| Project | Lane | Steal or skip |
|---|---|---|
| VLABench | Policy benchmarks | Skip. Task success, not rollout fidelity. Different lane. |
| LIBERO | Policy benchmarks | Skip. Policy eval, not prediction faithfulness. |
| stable-worldmodel (swm) | World-model platform | Steal. Adapter shipped in session 2 (`worldproof[swm]`). Its LeWM and DINO-WM latent models map cleanly onto our latent rollout and run on MPS. Pins: `stable-worldmodel==0.1.1`, `transformers<5` (checkpoints predate the 5.x weight rename). |
| LeRobot | World-model platform | Skip the model adapter, take the data. Re-checked in session 15. Its v0.6 world models are policies (they choose their own actions through `select_action`), not forward models you can drive with an external action sequence, and they need a lot of CUDA memory (a 5B video backbone, eval on a 140 GB GPU). The data path is shipped: `LeRobotDatasetSource` reads LeRobotDataset v3.0 directly (parquet and mp4, no `lerobot` package) on Python 3.10. Revisit the models if a CPU or MPS forward model with an external-action API lands. |
| Atari / ALE (ale-py) | Ground-truth provider | Steal. `AtariSimOracle` shipped in session 14 (`worldproof[atari]`). A deterministic pixel emulator, CPU only, ROMs bundled, no license step. It unlocks the signature checks on varied game frames, and cheaper than ManiSkill or CARLA. Atari sprites on flat backgrounds also suit the clean-scene tracker. |
| VBench (custom_input) | Rollout faithfulness (partial) | Steal. Wrap it, do not reimplement. Part of the citation trail. |
| PAN (arXiv 2511.09057) | Rollout faithfulness | Steal. The long-horizon causal-consistency framing and the progressive penalty. |

## References

The sources behind the metric set and the roadmap (SPEC.md sections 4 to 7).
One line each: what it is, what we took, and where it lands. Marked "(confirm)"
where the exact citation still needs checking.

| Source | Taken for | SPEC |
|---|---|---|
| arXiv 2606.07687, action-fidelity dissociation (PSNR and FVD are close to orthogonal to control utility and action recoverability) | The founding thesis. Motivates the action-recoverability probe and the "FVD is weak" label. | 1, 4 |
| arXiv 2607.06401, a definition and roadmap survey (fragmented eval, calibration for safety) | The lane justification, and calibration as a first-class metric. | 1, 4 |
| Yeom et al. 2026, what makes video world model latents action-relevant | Action recoverability promoted to the main latent check. The action-conditioning study is parked for v0.3. | 4, 7 |
| Unterthiner et al. 2018, a new metric for video generation | FVD as a weak reference, the temporal corruptions (frame swap, interleaving), and the StarCraft2-Videos unit tests (parked). | 4, 7 |
| PAN, arXiv 2511.09057 | The progressive-penalty summary and the long-horizon consistency framing. | 4, 5 |
| Duan et al. 2025, WorldScore | The consistency metrics that PAN's penalties weight. | 5 |
| Agarwal et al. 2021, deep RL at the edge of the statistical precipice | Interquartile mean, stratified bootstrap CIs, and performance profiles as the aggregation standard. | 5, 6 |
| Guo et al. 2017, on calibration of modern neural networks | The ECE and MCE calibration metrics. Temperature scaling parked. | 4, 7 |
| Angelopoulos and Bates 2021, conformal prediction (confirm) | The SSC and FSC coverage diagnostics (parked). | 7 |
| IntPhys, Riochet et al. (confirm) | Violation-of-expectation plausibility scoring (parked). | 7 |
| MEt3R 2025 and DUSt3R 2024 (confirm) | A pose-free multi-view 3D consistency metric (parked). | 7 |

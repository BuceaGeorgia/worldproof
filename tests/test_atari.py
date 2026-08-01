"""Atari sim-oracle tests (gated on the ``atari`` extra).

These require ``ale-py`` (`pip install worldproof[atari]`); the whole module
skips when it is absent, so the core suite / core-only CI stays green. ale-py
bundles the ROMs, so no download or license step is needed.

They confirm the two things the recon spike promised: an ``ALE/*`` game drops
into the existing ``GymSimOracle`` loop, and it satisfies the SimOracle
contract — pixel frames, deterministic in ``(seed, actions)``, and a real
counterfactual (shared context, action-driven divergence).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("ale_py")

from worldproof.baselines import CopyLastFrameBaseline  # noqa: E402
from worldproof.metrics import PSNR, Capabilities  # noqa: E402
from worldproof.report import evaluate, report_json  # noqa: E402
from worldproof.sim import AtariSimOracle, make_rollout  # noqa: E402

FRAME = (210, 160, 3)
SEQ_A = np.array([2, 3, 2, 3, 5, 0], np.float32)
SEQ_B = np.array([5, 5, 4, 4, 0, 0], np.float32)


@pytest.fixture(scope="module")
def oracle() -> AtariSimOracle:
    return AtariSimOracle("ALE/Pong-v5")


def test_atari_oracle_basics(oracle):
    assert oracle.action_dim == 1  # Discrete action space
    actions = oracle.sample_actions(5, seed=0)
    assert actions.shape == (5,) and actions.dtype == np.float32
    rollout = oracle.rollout(actions, seed=0, context_steps=2)
    assert rollout.future.shape == (5, *FRAME) and rollout.future.dtype == np.uint8
    assert rollout.context.shape == (2, *FRAME)
    assert isinstance(rollout.is_failure, bool)
    assert rollout.context_id == "ALE/Pong-v5:seed=0"


def test_atari_defaults_to_faithful_actions_no_sticky(oracle):
    # repeat_action_probability=0.0 so the *requested* action is applied.
    assert oracle._env.spec.kwargs.get("repeat_action_probability") == 0.0


def test_atari_rollout_is_deterministic_in_seed_and_actions(oracle):
    r1 = oracle.rollout(SEQ_A, seed=7, context_steps=3)
    r2 = oracle.rollout(SEQ_A, seed=7, context_steps=3)
    assert np.array_equal(r1.context, r2.context)
    assert np.array_equal(r1.future, r2.future)


def test_atari_counterfactual_shares_context_and_diverges(oracle):
    ra = oracle.rollout(SEQ_A, seed=5, context_steps=3)
    rb = oracle.rollout(SEQ_B, seed=5, context_steps=3)
    assert np.array_equal(ra.context, rb.context)  # same seed -> shared start
    assert not np.array_equal(ra.future, rb.future)  # actions drive divergence


def test_atari_rollout_is_evaluable(oracle):
    actions = np.array([2, 3, 2, 3], np.float32)
    rollout = make_rollout(
        oracle, CopyLastFrameBaseline(), seed=0, actions=actions, context_steps=3
    )
    assert rollout.modality == "pixels" and rollout.has_ground_truth
    result = PSNR().compute(rollout, device=torch.device("cpu"))
    assert result.horizon == len(actions)
    assert np.isfinite(result.summary)


def test_atari_set_runs_full_evaluate_report(oracle):
    """The whole evaluate() -> report-card pipeline over a set of real rollouts.

    With the tracker enabled (Atari is a clean scene), the object invariants run
    too, so the report covers fidelity and invariants together.
    """
    model = CopyLastFrameBaseline()
    rollouts = [
        make_rollout(
            oracle,
            model,
            seed=i,
            actions=oracle.sample_actions(6, seed=i),
            context_steps=3,
        )
        for i in range(6)
    ]
    report, _ = evaluate(rollouts, capabilities=Capabilities.detect(has_tracker=True))
    assert report.n_rollouts == 6
    names = {m.name for m in report.metrics}
    assert {"psnr", "ssim"} <= names  # fidelity ran on Pong frames
    assert {"object_count_conservation", "object_permanence"} <= names  # invariants ran
    assert all(np.isfinite(m.iqm) for m in report.metrics)
    assert "Evaluated 6 rollout(s)" in report.verdict
    import json

    json.dumps(report_json(report))  # report serializes cleanly

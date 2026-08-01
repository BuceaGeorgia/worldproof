"""Metric base-class enforcement, registry, and device-aware runner."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from worldproof.baselines import CopyLastFrameBaseline
from worldproof.core import Rollout, RolloutMetadata
from worldproof.metrics import (
    Capabilities,
    Metric,
    MetricRegistry,
    MetricRunner,
    register,
)

CPU = Capabilities(device=torch.device("cpu"))
GPU = Capabilities(device=torch.device("mps"))


# --------------------------------------------------------------------------- #
# Test-local dummy metrics (never registered globally)
# --------------------------------------------------------------------------- #


class OnesMetric(Metric):
    name = "dummy.ones"
    version = "1.0"
    higher_is_better = True

    def _curve(self, rollout, device):
        return np.ones((rollout.n_samples, rollout.horizon), dtype=np.float64)


class NeedsReferenceMetric(Metric):
    name = "dummy.needs_reference"
    version = "1.0"
    higher_is_better = False
    needs_reference = True

    def _curve(self, rollout, device):
        return np.zeros((rollout.n_samples, rollout.horizon))


class NeedsGPUMetric(Metric):
    name = "dummy.needs_gpu"
    version = "1.0"
    higher_is_better = True
    needs_gpu = True

    def _curve(self, rollout, device):
        return np.ones((rollout.n_samples, rollout.horizon))


class NeedsSimOracleMetric(Metric):
    name = "dummy.needs_sim_oracle"
    version = "1.0"
    higher_is_better = True
    needs_sim_oracle = True

    def _curve(self, rollout, device):
        return np.ones((rollout.n_samples, rollout.horizon))


class NeedsTrackerMetric(Metric):
    name = "dummy.needs_tracker"
    version = "1.0"
    higher_is_better = True
    needs_tracker = True

    def _curve(self, rollout, device):
        return np.ones((rollout.n_samples, rollout.horizon))


class ExplodingMetric(Metric):
    name = "dummy.exploding"
    version = "1.0"
    higher_is_better = True

    def _curve(self, rollout, device):
        raise RuntimeError("kaboom")


def _rollout():
    return CopyLastFrameBaseline().predict(
        np.zeros((3, 8, 8, 3), np.uint8), np.zeros((4, 2), np.float32), 4, n_samples=2
    )


def _rollout_with_ground_truth():
    context = np.zeros((3, 8, 8, 3), np.uint8)
    predictions = (
        np.zeros((4, 8, 8, 3), np.uint8),
        np.zeros((4, 8, 8, 3), np.uint8),
    )
    ground_truth = np.zeros((4, 8, 8, 3), np.uint8)
    metadata = RolloutMetadata(fps=10.0, model_id="m", resolution=(8, 8))
    return Rollout(
        "pixels",
        context,
        np.zeros((4, 2), np.float32),
        predictions,
        metadata,
        ground_truth,
    )


# --------------------------------------------------------------------------- #
# Metric base-class enforcement
# --------------------------------------------------------------------------- #


def test_metric_requires_name():
    with pytest.raises(TypeError, match="non-empty `name`"):

        class _NoName(Metric):
            version = "1.0"
            higher_is_better = True

            def _curve(self, rollout, device):
                return np.ones((1, 1))


def test_metric_requires_version():
    with pytest.raises(TypeError, match="non-empty `version`"):

        class _NoVersion(Metric):
            name = "x"
            higher_is_better = True

            def _curve(self, rollout, device):
                return np.ones((1, 1))


def test_metric_requires_direction():
    with pytest.raises(TypeError, match="higher_is_better"):

        class _NoDirection(Metric):
            name = "x"
            version = "1.0"

            def _curve(self, rollout, device):
                return np.ones((1, 1))


def test_metric_direction_must_be_bool():
    with pytest.raises(TypeError, match="higher_is_better"):

        class _BadDirection(Metric):
            name = "x"
            version = "1.0"
            higher_is_better = "yes"  # type: ignore[assignment]

            def _curve(self, rollout, device):
                return np.ones((1, 1))


def test_requirements_declares_only_needed_flags():
    assert OnesMetric.requirements() == {}
    assert NeedsReferenceMetric.requirements() == {"needs_reference": True}


def test_compute_stamps_direction_and_requirements():
    result = NeedsReferenceMetric().compute(
        _rollout_with_ground_truth(), device=torch.device("cpu")
    )
    assert result.name == "dummy.needs_reference"
    assert result.higher_is_better is False
    assert result.requirements_met["needs_reference"] is True
    assert result.curve.shape == (2, 4)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_register_get_create():
    reg = MetricRegistry()
    reg.register(OnesMetric)
    assert "dummy.ones" in reg
    assert len(reg) == 1
    assert reg.get("dummy.ones") is OnesMetric
    assert reg.create("dummy.ones").name == "dummy.ones"
    assert reg.names() == ["dummy.ones"]


def test_registry_reregister_same_class_is_idempotent():
    reg = MetricRegistry()
    reg.register(OnesMetric)
    reg.register(OnesMetric)
    assert len(reg) == 1


def test_registry_rejects_duplicate_name():
    reg = MetricRegistry()
    reg.register(OnesMetric)

    class _OnesClone(Metric):
        name = "dummy.ones"
        version = "2.0"
        higher_is_better = True

        def _curve(self, rollout, device):
            return np.ones((1, 1))

    with pytest.raises(ValueError, match="already registered"):
        reg.register(_OnesClone)


def test_registry_get_unknown_raises():
    with pytest.raises(KeyError, match="no metric registered"):
        MetricRegistry().get("nope")


def test_register_decorator_uses_global_registry():
    @register
    class _Registered(Metric):
        name = "dummy.registered_via_decorator"
        version = "1.0"
        higher_is_better = True

        def _curve(self, rollout, device):
            return np.ones((1, 1))

    from worldproof.metrics.registry import REGISTRY

    assert REGISTRY.get("dummy.registered_via_decorator") is _Registered


# --------------------------------------------------------------------------- #
# Runner — run, skip-and-report, graceful failure
# --------------------------------------------------------------------------- #


def test_runner_runs_metric_and_stamps_result():
    run = MetricRunner([OnesMetric()], capabilities=CPU).run(_rollout())
    assert not run.skips
    assert len(run.results) == 1
    result = run.results[0]
    assert result.name == "dummy.ones"
    assert result.higher_is_better is True
    assert result.curve.shape == (2, 4)


def test_runner_skips_needs_reference_without_ground_truth():
    run = MetricRunner([NeedsReferenceMetric()], capabilities=CPU).run(_rollout())
    assert not run.results
    assert len(run.skips) == 1
    assert "ground truth" in run.skips[0].reason


def test_runner_runs_needs_reference_with_ground_truth():
    run = MetricRunner([NeedsReferenceMetric()], capabilities=CPU).run(
        _rollout_with_ground_truth()
    )
    assert len(run.results) == 1
    assert not run.skips


def test_runner_skips_sim_oracle_and_tracker_with_actionable_reasons():
    run = MetricRunner(
        [NeedsSimOracleMetric(), NeedsTrackerMetric()], capabilities=CPU
    ).run(_rollout())
    assert not run.results
    reasons = " ".join(skip.reason for skip in run.skips)
    assert "sim oracle" in reasons
    assert "worldproof[trackers]" in reasons


def test_runner_gpu_gate():
    assert MetricRunner([NeedsGPUMetric()], capabilities=CPU).run(_rollout()).skips
    gpu_run = MetricRunner([NeedsGPUMetric()], capabilities=GPU).run(_rollout())
    assert len(gpu_run.results) == 1
    assert not gpu_run.skips


def test_runner_catches_metric_failure_instead_of_crashing():
    run = MetricRunner([ExplodingMetric()], capabilities=CPU).run(_rollout())
    assert not run.results
    assert len(run.skips) == 1
    assert "failed at runtime" in run.skips[0].reason
    assert "RuntimeError" in run.skips[0].reason


def test_runner_rejects_non_metric_instances():
    with pytest.raises(TypeError, match="Metric instances"):
        MetricRunner([OnesMetric])  # class, not instance


def test_run_many_produces_report_with_provenance():
    runner = MetricRunner([OnesMetric()], capabilities=CPU)
    report = runner.run_many([_rollout(), _rollout()])
    assert len(report.runs) == 2
    assert [run.index for run in report.runs] == [0, 1]
    assert len(report.all_results) == 2
    assert not report.all_skips
    assert report.device == "cpu"
    assert report.metric_versions["dummy.ones"] == "1.0"


def test_from_registry_builds_named_subset():
    reg = MetricRegistry()
    reg.register(OnesMetric)
    runner = MetricRunner.from_registry(["dummy.ones"], registry=reg, capabilities=CPU)
    assert len(runner.metrics) == 1
    assert runner.metrics[0].name == "dummy.ones"


def test_capabilities_detect_defaults_to_autodetected_device():
    caps = Capabilities.detect()
    assert caps.device.type in ("cpu", "mps", "cuda")
    assert caps.has_sim_oracle is False
    assert caps.has_tracker is False

"""Metric layer: the :class:`Metric` base, the registry, and the device-aware
runner.

v0.1 ships the infrastructure; concrete metrics (PSNR/SSIM/LPIPS, the latent
action-recoverability probe, calibration, invariants) land in later work, each
subclassing :class:`Metric` and registering itself.
"""

from worldproof.metrics.aggregate import (
    ActionRecoverabilityProbe,
    AggregateMeasure,
    AggregateResult,
    FrechetVideoDistance,
    KineticsVideoExtractor,
    MeasureSkipped,
    VideoFeatureExtractor,
    default_aggregates,
    default_video_extractor,
    frechet_distance,
    run_aggregates,
)
from worldproof.metrics.base import Metric, MetricIdentity
from worldproof.metrics.calibration import CalibrationECE, CalibrationMCE
from worldproof.metrics.fidelity import (
    PSNR,
    SSIM,
    FidelityMetric,
    PSNRDynamic,
    SSIMDynamic,
)
from worldproof.metrics.invariants import ObjectCountConservation, ObjectPermanence
from worldproof.metrics.latent import LatentPredictionError
from worldproof.metrics.perceptual import LPIPS, LPIPSDynamic
from worldproof.metrics.registry import REGISTRY, MetricRegistry, register
from worldproof.metrics.runner import (
    Capabilities,
    MetricRunner,
    MetricSkip,
    RolloutRun,
    RunReport,
)
from worldproof.metrics.signature import (
    CounterfactualDivergence,
    FailureFaithfulness,
    SignatureRunner,
)

__all__ = [
    "Metric",
    "MetricIdentity",
    "MetricRegistry",
    "REGISTRY",
    "register",
    "Capabilities",
    "MetricRunner",
    "MetricSkip",
    "RolloutRun",
    "RunReport",
    "FidelityMetric",
    "PSNR",
    "PSNRDynamic",
    "SSIM",
    "SSIMDynamic",
    "LPIPS",
    "LPIPSDynamic",
    "LatentPredictionError",
    "CalibrationECE",
    "CalibrationMCE",
    "CounterfactualDivergence",
    "FailureFaithfulness",
    "SignatureRunner",
    "ObjectCountConservation",
    "ObjectPermanence",
    "ActionRecoverabilityProbe",
    "FrechetVideoDistance",
    "KineticsVideoExtractor",
    "VideoFeatureExtractor",
    "AggregateMeasure",
    "AggregateResult",
    "MeasureSkipped",
    "default_aggregates",
    "default_video_extractor",
    "run_aggregates",
    "frechet_distance",
]

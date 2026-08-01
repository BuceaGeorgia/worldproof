"""worldproof — a reality check for world models.

Public API surface for v0.1. The frozen data contracts, device selection, the
model-adapter protocol, the built-in baseline anchors, the folder-of-rollouts
source, and the metric layer (base class, registry, device-aware runner) are
exported here; concrete metrics, the report card, and the CLI land in later
work.
"""

from worldproof.adapters import (
    FOLDER_FORMAT_VERSION,
    SWMAdapter,
    WorldModelAdapter,
    iter_rollouts,
    load_rollout,
    save_rollout,
)
from worldproof.baselines import ActionBlindBaseline, CopyLastFrameBaseline
from worldproof.core import (
    CounterfactualPair,
    MetricResult,
    Modality,
    Rollout,
    RolloutMetadata,
    VOEPair,
)
from worldproof.device import available_devices, get_device
from worldproof.metrics import (
    REGISTRY,
    ActionRecoverabilityProbe,
    AggregateResult,
    Capabilities,
    CounterfactualDivergence,
    FailureFaithfulness,
    FrechetVideoDistance,
    KineticsVideoExtractor,
    Metric,
    MetricRegistry,
    MetricRunner,
    MetricSkip,
    ObjectCountConservation,
    ObjectPermanence,
    RolloutRun,
    RunReport,
    SignatureRunner,
    VideoFeatureExtractor,
    default_video_extractor,
    frechet_distance,
    register,
)
from worldproof.report import (
    MetricSummary,
    Report,
    build_report,
    evaluate,
    report_html,
    report_json,
)
from worldproof.sim import (
    AtariSimOracle,
    DatasetSource,
    GymSimOracle,
    LeRobotDatasetSource,
    OracleRollout,
    SimOracle,
    ToySimOracle,
    generate_rollouts,
    make_counterfactual_pair,
    make_rollout,
    make_voe_pair,
    rollouts_from_dataset,
)
from worldproof.tracking import BlobTracker, Detection, Tracker

__version__ = "0.1.0"

__all__ = [
    "MetricResult",
    "Modality",
    "Rollout",
    "RolloutMetadata",
    "CounterfactualPair",
    "VOEPair",
    "available_devices",
    "get_device",
    "WorldModelAdapter",
    "SWMAdapter",
    "CopyLastFrameBaseline",
    "ActionBlindBaseline",
    "FOLDER_FORMAT_VERSION",
    "save_rollout",
    "load_rollout",
    "iter_rollouts",
    "Metric",
    "MetricRegistry",
    "REGISTRY",
    "register",
    "Capabilities",
    "MetricRunner",
    "MetricSkip",
    "RolloutRun",
    "RunReport",
    "CounterfactualDivergence",
    "FailureFaithfulness",
    "SignatureRunner",
    "ObjectCountConservation",
    "ObjectPermanence",
    "ActionRecoverabilityProbe",
    "FrechetVideoDistance",
    "KineticsVideoExtractor",
    "VideoFeatureExtractor",
    "default_video_extractor",
    "AggregateResult",
    "frechet_distance",
    "SimOracle",
    "OracleRollout",
    "ToySimOracle",
    "GymSimOracle",
    "AtariSimOracle",
    "make_rollout",
    "make_counterfactual_pair",
    "make_voe_pair",
    "generate_rollouts",
    "DatasetSource",
    "rollouts_from_dataset",
    "LeRobotDatasetSource",
    "Tracker",
    "BlobTracker",
    "Detection",
    "evaluate",
    "build_report",
    "report_json",
    "report_html",
    "Report",
    "MetricSummary",
    "__version__",
]

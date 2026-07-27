"""Verified low-rank checkpoint transport for Track 1."""

from poetry50m.trajectory.application import apply_accepted_candidate
from poetry50m.trajectory.config import TrajectoryConfig
from poetry50m.trajectory.endpoint_geometry import (
    AngularGeometrySummary,
    AngularTensorMetrics,
    ConsecutiveDeltaGeometry,
    EndpointGeometryReport,
    GeometryMetrics,
    SnapshotProvenance,
    TurningSummary,
    analyze_endpoint_geometry,
    analyze_endpoint_geometry_paths,
)
from poetry50m.trajectory.forecast import (
    LinearForecastConfig,
    LowRankForecastConfig,
    linear_finite_difference,
    low_rank_temporal_forecast,
)
from poetry50m.trajectory.gates import (
    AnchorOutputs,
    CandidateAcceptanceGates,
    decide_candidate,
    fixed_anchor_function_drift,
    forecast_safety_report,
    future_target_distillation_loss,
)
from poetry50m.trajectory.manifest import (
    OperationScope,
    RunManifest,
    SuccessLevel,
    TrajectoryExperimentManifest,
)
from poetry50m.trajectory.preparation import state_dict_hash
from poetry50m.trajectory.snapshots import load_weight_snapshot, save_weight_snapshot
from poetry50m.trajectory.types import SnapshotMetadata, WeightSnapshot
from poetry50m.trajectory.verification import (
    CandidateVerification,
    VerificationBatch,
    verify_candidate,
)

__all__ = [
    "AnchorOutputs",
    "AngularGeometrySummary",
    "AngularTensorMetrics",
    "CandidateAcceptanceGates",
    "CandidateVerification",
    "ConsecutiveDeltaGeometry",
    "EndpointGeometryReport",
    "GeometryMetrics",
    "LinearForecastConfig",
    "LowRankForecastConfig",
    "OperationScope",
    "RunManifest",
    "SnapshotMetadata",
    "SnapshotProvenance",
    "SuccessLevel",
    "TrajectoryConfig",
    "TrajectoryExperimentManifest",
    "TurningSummary",
    "WeightSnapshot",
    "apply_accepted_candidate",
    "analyze_endpoint_geometry",
    "analyze_endpoint_geometry_paths",
    "decide_candidate",
    "fixed_anchor_function_drift",
    "forecast_safety_report",
    "future_target_distillation_loss",
    "linear_finite_difference",
    "load_weight_snapshot",
    "low_rank_temporal_forecast",
    "save_weight_snapshot",
    "state_dict_hash",
    "VerificationBatch",
    "verify_candidate",
]

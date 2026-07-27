"""Public endpoint-informed geometry API.

The schema and numerical kernels are separated so the durable evidence format
does not depend on the implementation details of the streaming calculation.
"""

from poetry50m.trajectory.endpoint_analysis import (
    analyze_endpoint_geometry,
    analyze_endpoint_geometry_paths,
)
from poetry50m.trajectory.endpoint_schema import (
    EVIDENCE_LABEL,
    FORMULA_DEFINITIONS,
    METHOD,
    SCHEMA_VERSION,
    AngularGeometrySummary,
    AngularTensorMetrics,
    ConsecutiveDeltaGeometry,
    EndpointGeometryReport,
    FormulaDefinition,
    GeometryMetrics,
    SnapshotProvenance,
    TurningSummary,
)

__all__ = [
    "EVIDENCE_LABEL",
    "FORMULA_DEFINITIONS",
    "METHOD",
    "SCHEMA_VERSION",
    "AngularGeometrySummary",
    "AngularTensorMetrics",
    "ConsecutiveDeltaGeometry",
    "EndpointGeometryReport",
    "FormulaDefinition",
    "GeometryMetrics",
    "SnapshotProvenance",
    "TurningSummary",
    "analyze_endpoint_geometry",
    "analyze_endpoint_geometry_paths",
]

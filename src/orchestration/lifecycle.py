"""
SATQuery AI — Canonical Execution Lifecycle & Tracing

Defines the 15-stage structured pipeline lifecycle, stage status enums,
stage trace models, and aggregated pipeline metrics for complete, truthful,
and auditable query execution traceability.
"""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    QUERY_RECEIVED = "query_received"
    INPUT_VALIDATION = "input_validation"
    QUERY_UNDERSTANDING = "query_understanding"
    CAPABILITY_SENSOR_ROUTING = "capability_sensor_routing"
    EVIDENCE_PLANNING = "evidence_planning"
    INPUT_SCENE_RESOLUTION = "input_scene_resolution"
    PREPROCESSING = "preprocessing"
    SPECIALIST_EXECUTION = "specialist_execution"
    EVIDENCE_REGISTRATION = "evidence_registration"
    GIS_SPATIAL_PROCESSING = "gis_spatial_processing"
    EVIDENCE_VERIFICATION = "evidence_verification"
    CONFIDENCE_ABSTENTION_DECISION = "confidence_abstention_decision"
    RESPONSE_GENERATION = "response_generation"
    VISUALIZATION_PREPARATION = "visualization_preparation"
    EXECUTION_REPORT = "execution_report"


class StageStatus(str, Enum):
    EXECUTED = "EXECUTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    SKIPPED_WITH_REASON = "SKIPPED_WITH_REASON"
    FAILED_EXECUTION = "FAILED_EXECUTION"


STAGE_DISPLAY_NAMES: dict[PipelineStage, str] = {
    PipelineStage.QUERY_RECEIVED: "Query Received",
    PipelineStage.INPUT_VALIDATION: "Input Validation",
    PipelineStage.QUERY_UNDERSTANDING: "Query Understanding",
    PipelineStage.CAPABILITY_SENSOR_ROUTING: "Capability & Sensor Routing",
    PipelineStage.EVIDENCE_PLANNING: "Evidence Planning",
    PipelineStage.INPUT_SCENE_RESOLUTION: "Input Scene Resolution",
    PipelineStage.PREPROCESSING: "Preprocessing & Alignment",
    PipelineStage.SPECIALIST_EXECUTION: "Specialist Perception Execution",
    PipelineStage.EVIDENCE_REGISTRATION: "Evidence Registration",
    PipelineStage.GIS_SPATIAL_PROCESSING: "GIS Spatial Processing",
    PipelineStage.EVIDENCE_VERIFICATION: "GeoReason Verification",
    PipelineStage.CONFIDENCE_ABSTENTION_DECISION: "Confidence & Abstention Decision",
    PipelineStage.RESPONSE_GENERATION: "Response Generation",
    PipelineStage.VISUALIZATION_PREPARATION: "Visualization Preparation",
    PipelineStage.EXECUTION_REPORT: "Execution Report Packaging",
}


class ExecutionStageTrace(BaseModel):
    """Execution trace record for one canonical pipeline stage."""

    stage_id: str
    stage_name: str
    started_at: str
    completed_at: str
    status: StageStatus = StageStatus.EXECUTED
    execution_success: bool = True
    result_status: str = "completed"
    inputs: list[Any] = Field(default_factory=list)
    outputs: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    reason: str | None = None


class PipelineMetrics(BaseModel):
    """Aggregate metrics across the canonical execution lifecycle."""

    total_stages: int = 15
    executed_stages: int = 0
    successful_executions: int = 0
    execution_errors: int = 0
    not_applicable_stages: int = 0
    skipped_stages: int = 0
    abstentions: int = 0
    evidence_count: int = 0
    final_decision: str = "VERIFIED"
    verification_status: str = "verified"

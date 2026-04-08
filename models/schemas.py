"""Pydantic models and enums for LabGuard protocol analysis.

Single source of truth for all data structures. Every other layer
(prompts, API, UI) imports from this file.
"""

from __future__ import annotations

import re
from enum import Enum
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Canonical category and severity lists — prompts.py imports these
# ---------------------------------------------------------------------------

VALID_CATEGORIES = [
    "contamination_risk",
    "ambiguous_quantity",
    "temperature_sensitivity",
    "missing_control",
    "reproducibility",
    "reagent_incompatibility",
    "timing_dependency",
    "equipment_assumption",
    "safety_hazard",
    "regulatory_compliance",
    "sample_integrity",
    "validation_missing",
]

VALID_SEVERITIES = ["critical", "warning", "info"]

# ---------------------------------------------------------------------------
# Input validation constants — single source of truth for protocol text limits.
# Enforced in both UI (text_area max_chars) and backend (Pydantic schema).
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH = 15_000
MIN_INPUT_LENGTH = 50
# Minimum unique non-whitespace chars — rejects spam (e.g., "AAAA..." x 15K)
MIN_UNIQUE_CHARS = 10
# Rejects null bytes and control chars that could break downstream processing.
# Preserves newline (\n), tab (\t), carriage return (\r) for protocol formatting.
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SeverityLevel(str, Enum):
    """Three-tier severity for protocol flags."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class FailureCategory(str, Enum):
    """Twelve failure categories for protocol analysis."""
    CONTAMINATION_RISK = "contamination_risk"
    AMBIGUOUS_QUANTITY = "ambiguous_quantity"
    TEMPERATURE_SENSITIVITY = "temperature_sensitivity"
    MISSING_CONTROL = "missing_control"
    REPRODUCIBILITY = "reproducibility"
    REAGENT_INCOMPATIBILITY = "reagent_incompatibility"
    TIMING_DEPENDENCY = "timing_dependency"
    EQUIPMENT_ASSUMPTION = "equipment_assumption"
    SAFETY_HAZARD = "safety_hazard"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    SAMPLE_INTEGRITY = "sample_integrity"
    VALIDATION_MISSING = "validation_missing"


# ---------------------------------------------------------------------------
# Input validation — schema-based sanitization per OWASP input validation
# ---------------------------------------------------------------------------


class ProtocolInput(BaseModel):
    """Validated and sanitized protocol text from the user."""
    text: str = Field(..., min_length=MIN_INPUT_LENGTH, max_length=MAX_INPUT_LENGTH)

    @field_validator("text", mode="before")
    @classmethod
    def _strip_whitespace(cls, v: str) -> str:
        """Strip leading/trailing whitespace before length checks."""
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _validate_content(self):
        """Reject control characters and degenerate input."""
        if _CONTROL_CHAR_RE.search(self.text):
            raise ValueError(
                "Input contains invalid characters. "
                "Please paste plain text only."
            )
        # Fewer than MIN_UNIQUE_CHARS distinct non-whitespace characters
        # indicates spam or garbage, not a real protocol
        non_ws = re.sub(r'\s', '', self.text)
        if len(set(non_ws)) < MIN_UNIQUE_CHARS:
            raise ValueError(
                "Input does not appear to be a valid lab protocol."
            )
        return self


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class Flag(BaseModel):
    """A single detected issue within a protocol step."""
    flag_id: str = ""
    category: FailureCategory
    severity: SeverityLevel
    title: str = Field(..., max_length=120)
    description: str
    affected_text: str
    suggested_fix: str

    @model_validator(mode="after")
    def _strings_non_empty(self):
        """Reject empty required string fields."""
        for field_name in ("title", "description", "affected_text", "suggested_fix"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        return self


class QCCheckpoint(BaseModel):
    """A quality-control action the researcher should perform."""
    checkpoint_id: str = ""
    after_step: int = Field(..., ge=1)
    action: str
    expected_result: str
    failure_action: str

    @model_validator(mode="after")
    def _strings_non_empty(self):
        """Reject empty required string fields."""
        for field_name in ("action", "expected_result", "failure_action"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        return self


class ProtocolStep(BaseModel):
    """A single parsed step from the protocol with its flags and checkpoints."""
    step_number: int = Field(..., ge=1)
    original_text: str
    flags: list[Flag] = []
    qc_checkpoints: list[QCCheckpoint] = []


class ProtocolMetadata(BaseModel):
    """High-level information about the protocol as a whole."""
    protocol_title: str = "Untitled Protocol"
    protocol_type: str = "general"
    organism: str | None = None
    estimated_duration: str | None = None
    technique_tags: list[str] = []


class AnalysisSummary(BaseModel):
    """Aggregate statistics computed from the analysis steps."""
    total_steps: int = Field(..., ge=0)
    total_flags: int = Field(..., ge=0)
    critical_count: int = Field(..., ge=0)
    warning_count: int = Field(..., ge=0)
    info_count: int = Field(..., ge=0)
    total_qc_checkpoints: int = Field(..., ge=0)
    categories_detected: list[FailureCategory] = []


# ---------------------------------------------------------------------------
# Claude response payload — internal, used only for parsing
# ---------------------------------------------------------------------------

class ClaudeResponsePayload(BaseModel):
    """Shape of Claude's JSON response. IDs are omitted — added client-side."""
    metadata: ProtocolMetadata
    steps: list[ProtocolStep]
    overall_assessment: str


# ---------------------------------------------------------------------------
# Public model returned to the UI
# ---------------------------------------------------------------------------

class ProtocolAnalysis(BaseModel):
    """Top-level analysis result consumed by the UI layer."""
    analysis_id: str
    input_text: str
    metadata: ProtocolMetadata
    steps: list[ProtocolStep]
    summary: AnalysisSummary
    overall_assessment: str
    timestamp: str
    model_version: str

    @classmethod
    def from_claude_response(
        cls,
        payload: ClaudeResponsePayload,
        input_text: str,
        model_version: str,
    ) -> "ProtocolAnalysis":
        """Build a full ProtocolAnalysis from Claude's parsed response."""
        # Assign client-side IDs to flags and checkpoints
        for step in payload.steps:
            for seq, flag in enumerate(step.flags, start=1):
                flag.flag_id = f"F-{step.step_number}-{seq}"
            for seq, qc in enumerate(step.qc_checkpoints, start=1):
                qc.checkpoint_id = f"QC-{step.step_number}-{seq}"

        # Compute summary from steps
        all_flags = [f for s in payload.steps for f in s.flags]
        categories_seen: list[FailureCategory] = list(
            dict.fromkeys(f.category for f in all_flags)
        )
        summary = AnalysisSummary(
            total_steps=len(payload.steps),
            total_flags=len(all_flags),
            critical_count=sum(
                1 for f in all_flags if f.severity == SeverityLevel.CRITICAL
            ),
            warning_count=sum(
                1 for f in all_flags if f.severity == SeverityLevel.WARNING
            ),
            info_count=sum(
                1 for f in all_flags if f.severity == SeverityLevel.INFO
            ),
            total_qc_checkpoints=sum(
                len(s.qc_checkpoints) for s in payload.steps
            ),
            categories_detected=categories_seen,
        )

        return cls(
            analysis_id=str(uuid4()),
            input_text=input_text,
            metadata=payload.metadata,
            steps=payload.steps,
            summary=summary,
            overall_assessment=payload.overall_assessment,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_version=model_version,
        )

"""Anthropic tool definition for structured protocol-analysis output.

Using forced tool use guarantees the model's output conforms to the
ClaudeResponsePayload schema — no text-to-JSON extraction required.
The tool's input_schema is generated directly from the Pydantic model
so the schema and validator never drift.
"""

from __future__ import annotations

from models.schemas import ClaudeResponsePayload

ANALYSIS_TOOL_NAME = "record_protocol_analysis"

ANALYSIS_TOOL = {
    "name": ANALYSIS_TOOL_NAME,
    "description": (
        "Record the structured failure-point analysis of a laboratory protocol. "
        "Call this tool exactly once per request with the full analysis. "
        "Every flag must cite a specific mechanism, threshold, reagent, or "
        "failure mode — no generic advice."
    ),
    "input_schema": ClaudeResponsePayload.model_json_schema(),
}

ANALYSIS_TOOL_CHOICE = {"type": "tool", "name": ANALYSIS_TOOL_NAME}

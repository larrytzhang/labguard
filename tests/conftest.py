"""Shared fixtures for LabGuard tests.

All API calls are mocked — no ANTHROPIC_API_KEY required to run tests.
"""

from unittest.mock import MagicMock

import pytest

from llm.tools import ANALYSIS_TOOL_NAME


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_PROTOCOL_TEXT = (
    "1. Collect 200 uL of whole blood into a 1.5 mL microcentrifuge tube.\n"
    "2. Add 20 uL of Proteinase K solution and 200 uL of lysis buffer AL.\n"
    "3. Incubate at 56C for 10 minutes.\n"
    "4. Centrifuge at 6,000 x g for 1 minute."
)


def make_claude_response_dict(
    num_steps=2,
    flags_per_step=1,
    qc_per_step=0,
):
    """Build a valid dict matching ClaudeResponsePayload schema."""
    steps = []
    for s in range(1, num_steps + 1):
        flags = [
            {
                "category": "contamination_risk",
                "severity": "critical" if s == 1 else "warning",
                "title": f"Test flag {s}-{f}",
                "description": f"Detailed description for flag {s}-{f}.",
                "affected_text": f"Step {s} affected text",
                "suggested_fix": f"Fix for flag {s}-{f}.",
            }
            for f in range(1, flags_per_step + 1)
        ]
        qcs = [
            {
                "after_step": s,
                "action": f"Verify step {s}",
                "expected_result": "Pass criterion",
                "failure_action": "Repeat step",
            }
            for _ in range(qc_per_step)
        ]
        steps.append(
            {
                "step_number": s,
                "original_text": f"Step {s} original text.",
                "flags": flags,
                "qc_checkpoints": qcs,
            }
        )

    return {
        "metadata": {
            "protocol_title": "Test Protocol",
            "protocol_type": "molecular_biology",
            "organism": "human",
            "estimated_duration": "2 hours",
            "technique_tags": ["extraction", "centrifugation"],
        },
        "steps": steps,
        "overall_assessment": "This protocol has moderate risk.",
    }


@pytest.fixture
def sample_protocol_text():
    """Return a minimal valid protocol string."""
    return SAMPLE_PROTOCOL_TEXT


@pytest.fixture
def sample_claude_response_dict():
    """Return a valid dict matching ClaudeResponsePayload schema."""
    return make_claude_response_dict()


def make_tool_use_block(tool_input: dict, name: str = ANALYSIS_TOOL_NAME):
    """Build a mock tool_use content block with the given input dict."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = tool_input
    return block


@pytest.fixture
def mock_anthropic_message(sample_claude_response_dict):
    """Return a mock anthropic.types.Message with a valid tool_use block."""
    msg = MagicMock()
    msg.stop_reason = "tool_use"
    msg.content = [make_tool_use_block(sample_claude_response_dict)]
    return msg


@pytest.fixture(autouse=True)
def reset_api_client():
    """Reset the module-level API client before each test."""
    import llm.api
    llm.api._client = None
    yield
    llm.api._client = None

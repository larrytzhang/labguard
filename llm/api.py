"""Anthropic API integration for LabGuard protocol analysis.

Public interface: analyze_protocol() and LabGuardAPIError.
All other functions are private implementation details.

Uses forced tool use so the model's output is a structured dict that
already matches ClaudeResponsePayload — no text extraction required.
"""

from __future__ import annotations

import logging
import os

import anthropic
import streamlit as st

from llm.prompts import ANALYSIS_SYSTEM_PROMPT, build_user_message
from llm.tools import ANALYSIS_TOOL, ANALYSIS_TOOL_CHOICE, ANALYSIS_TOOL_NAME
from models.schemas import ClaudeResponsePayload, ProtocolAnalysis, MAX_INPUT_LENGTH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8192
TEMPERATURE = 0.0
API_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# Client initialization (lazy — so Streamlit secrets are loaded before use)
# ---------------------------------------------------------------------------

_client = None


def _get_client() -> anthropic.Anthropic:
    """Return the Anthropic client, initializing on first call."""
    global _client
    if _client is None:
        api_key = os.environ.get(API_KEY_ENV_VAR)
        if not api_key:
            try:
                api_key = st.secrets[API_KEY_ENV_VAR]
            except (KeyError, FileNotFoundError):
                pass
        if not api_key:
            raise LabGuardAPIError(
                "API key is not configured. Please set the "
                "ANTHROPIC_API_KEY environment variable or add it "
                "to .streamlit/secrets.toml."
            )
        _client = anthropic.Anthropic(api_key=api_key, timeout=API_TIMEOUT)
    return _client


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class LabGuardAPIError(Exception):
    """API error with a user-friendly message safe to display in the UI."""

    def __init__(self, user_message: str):
        self.user_message = user_message
        super().__init__(user_message)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _call_api(protocol_text: str) -> anthropic.types.Message:
    """Send protocol text to Claude with forced tool use and return the response."""
    client = _get_client()
    user_message = build_user_message(protocol_text)

    try:
        return client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=ANALYSIS_SYSTEM_PROMPT,
            tools=[ANALYSIS_TOOL],
            tool_choice=ANALYSIS_TOOL_CHOICE,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError:
        raise LabGuardAPIError(
            "The API key is invalid. Please check your "
            "ANTHROPIC_API_KEY environment variable."
        )
    except anthropic.RateLimitError:
        raise LabGuardAPIError(
            "The API is temporarily overloaded. "
            "Please wait a moment and try again."
        )
    except anthropic.APITimeoutError:
        raise LabGuardAPIError("The request timed out. Please try again.")
    except anthropic.APIConnectionError:
        raise LabGuardAPIError(
            "Could not connect to the API. "
            "Please check your internet connection."
        )
    except anthropic.APIStatusError as exc:
        logger.error("API status error: %s", exc.status_code)
        raise LabGuardAPIError(
            "Claude is temporarily unavailable. "
            "Please wait a moment and try again."
        )


def _extract_tool_input(response: anthropic.types.Message) -> dict:
    """Return the input dict from the forced tool_use content block."""
    if not response.content:
        raise LabGuardAPIError(
            "Claude returned an empty response. Please try again."
        )

    for block in response.content:
        # Support both SDK objects (attr access) and dict-like mocks
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        block_name = getattr(block, "name", None) or (
            block.get("name") if isinstance(block, dict) else None
        )
        if block_type == "tool_use" and block_name == ANALYSIS_TOOL_NAME:
            tool_input = getattr(block, "input", None)
            if tool_input is None and isinstance(block, dict):
                tool_input = block.get("input")
            if isinstance(tool_input, dict):
                return tool_input

    logger.error(
        "No %s tool_use block in response; content types: %s",
        ANALYSIS_TOOL_NAME,
        [getattr(b, "type", None) for b in response.content],
    )
    raise LabGuardAPIError(
        "Claude returned an unexpected response format. Please try again."
    )


def _parse_response(
    response: anthropic.types.Message,
    protocol_text: str,
) -> ProtocolAnalysis:
    """Parse Claude's raw response into a validated ProtocolAnalysis."""
    # Truncation: tool_use blocks can be cut off mid-JSON
    if response.stop_reason == "max_tokens":
        logger.warning("Response truncated (max_tokens reached)")
        raise LabGuardAPIError(
            "The analysis was too long and got cut off. "
            "Try submitting a shorter protocol."
        )

    tool_input = _extract_tool_input(response)

    try:
        payload = ClaudeResponsePayload.model_validate(tool_input)
    except Exception as exc:
        logger.error("Schema validation error: %s", exc)
        raise LabGuardAPIError(
            "The analysis was structured incorrectly. Please try again."
        )

    return ProtocolAnalysis.from_claude_response(payload, protocol_text, MODEL)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def analyze_protocol(protocol_text: str) -> ProtocolAnalysis:
    """Analyze a lab protocol for failure points. Returns structured results."""
    # Defensive length check — belt-and-suspenders with UI-layer Pydantic validation.
    # Prevents oversized input from reaching the API if validation is bypassed.
    if not isinstance(protocol_text, str) or len(protocol_text) > MAX_INPUT_LENGTH:
        raise LabGuardAPIError(
            "Invalid input. Protocol text exceeds maximum length."
        )
    try:
        response = _call_api(protocol_text)
        return _parse_response(response, protocol_text)
    except LabGuardAPIError:
        raise
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        raise LabGuardAPIError(
            "An unexpected error occurred. Please try again."
        )

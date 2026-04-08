"""Anthropic API integration for LabGuard protocol analysis.

Public interface: analyze_protocol() and LabGuardAPIError.
All other functions are private implementation details.
"""

from __future__ import annotations

import json
import logging
import os
import re

import anthropic
import streamlit as st

from llm.prompts import ANALYSIS_SYSTEM_PROMPT, build_user_message
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

def _extract_json(raw_text: str) -> str | None:
    """Extract a JSON object string from raw API response text."""
    text = raw_text.strip()

    # Try direct parse first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Brace matching — find the outermost { ... }, aware of JSON strings
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    return None
    return None


def _call_api(protocol_text: str) -> anthropic.types.Message:
    """Send protocol text to Claude and return the raw API response."""
    client = _get_client()
    user_message = build_user_message(protocol_text)

    try:
        return client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=ANALYSIS_SYSTEM_PROMPT,
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


def _parse_response(
    response: anthropic.types.Message,
    protocol_text: str,
) -> ProtocolAnalysis:
    """Parse Claude's raw response into a validated ProtocolAnalysis."""
    # Check for truncation
    if response.stop_reason == "max_tokens":
        logger.warning("Response truncated (max_tokens reached)")
        raise LabGuardAPIError(
            "The analysis was too long and got cut off. "
            "Try submitting a shorter protocol."
        )

    # Extract text content
    if not response.content:
        raise LabGuardAPIError(
            "Claude returned an empty response. Please try again."
        )
    raw_text = response.content[0].text

    # Extract JSON
    json_string = _extract_json(raw_text)
    if json_string is None:
        logger.error("No JSON found in response: %s", raw_text[:500])
        raise LabGuardAPIError(
            "Claude returned an unexpected response format. "
            "Please try again."
        )

    # Parse JSON
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as exc:
        logger.error("JSON decode error: %s", exc)
        raise LabGuardAPIError(
            "Claude returned malformed data. Please try again."
        )

    # Validate against Pydantic schema
    try:
        payload = ClaudeResponsePayload.model_validate(data)
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

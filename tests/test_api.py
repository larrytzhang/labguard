"""Tests for llm/api.py — JSON extraction, API calls, and response parsing."""

import json
from unittest.mock import MagicMock, patch

import pytest

from llm.api import (
    _extract_json,
    _get_client,
    _call_api,
    _parse_response,
    analyze_protocol,
    LabGuardAPIError,
    API_KEY_ENV_VAR,
    MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    API_TIMEOUT,
)
from models.schemas import ProtocolAnalysis, MAX_INPUT_LENGTH
from tests.conftest import make_claude_response_dict


# ---------------------------------------------------------------------------
# _extract_json() — 3 code paths + bug fix verification
# ---------------------------------------------------------------------------

class TestExtractJson:
    """Tests for JSON extraction from raw API response text."""

    # -- Path 1: Direct parse --

    def test_direct_valid_json(self):
        """Return valid JSON string unchanged."""
        raw = '{"key": "value"}'
        assert _extract_json(raw) == raw

    def test_direct_json_with_whitespace(self):
        """Strip outer whitespace and return valid JSON."""
        raw = '  {"key": "value"}  \n'
        assert _extract_json(raw) == '{"key": "value"}'

    def test_direct_nested_json(self):
        """Handle nested JSON objects."""
        raw = '{"outer": {"inner": "value"}, "list": [1, 2]}'
        assert _extract_json(raw) == raw

    # -- Path 2: Markdown fence stripping --

    def test_json_fence(self):
        """Extract JSON from ```json code fence."""
        raw = '```json\n{"key": "value"}\n```'
        assert _extract_json(raw) == '{"key": "value"}'

    def test_generic_fence(self):
        """Extract JSON from generic ``` code fence."""
        raw = '```\n{"key": "value"}\n```'
        assert _extract_json(raw) == '{"key": "value"}'

    def test_fence_with_surrounding_text(self):
        """Extract JSON from fence with text before and after."""
        raw = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        assert _extract_json(raw) == '{"key": "value"}'

    def test_invalid_json_in_fence_falls_through(self):
        """Invalid JSON inside fence falls through to brace matching."""
        raw = '```json\n{not valid json}\n```'
        assert _extract_json(raw) is None

    # -- Path 3: Brace matching --

    def test_brace_match_with_leading_text(self):
        """Extract JSON from text with leading non-JSON content."""
        raw = 'Here is the analysis: {"key": "value"}'
        assert _extract_json(raw) == '{"key": "value"}'

    def test_brace_match_nested_objects(self):
        """Handle nested braces correctly."""
        obj = {"outer": {"middle": {"inner": "deep"}}}
        raw = f"Some text {json.dumps(obj)} more text"
        result = _extract_json(raw)
        assert json.loads(result) == obj

    def test_brace_in_string_value(self):
        """Handle } inside a JSON string value without premature termination."""
        obj = {"description": "Use format {key} for templates"}
        raw = f"Analysis: {json.dumps(obj)}"
        result = _extract_json(raw)
        assert result is not None
        assert json.loads(result) == obj

    def test_escaped_quotes_in_string(self):
        """Handle escaped quotes inside JSON string values."""
        raw = r'Text {"key": "she said \"hello\""}'
        result = _extract_json(raw)
        assert result is not None
        parsed = json.loads(result)
        assert "hello" in parsed["key"]

    def test_complex_brace_in_string(self):
        """Handle multiple braces inside string values."""
        obj = {"text": "if (x > 0) { return y; } else { return z; }"}
        raw = f"Result: {json.dumps(obj)}"
        result = _extract_json(raw)
        assert result is not None
        assert json.loads(result) == obj

    # -- Negative cases --

    def test_no_json_returns_none(self):
        """Return None when no JSON is present."""
        assert _extract_json("Just plain text, no JSON here.") is None

    def test_no_opening_brace_returns_none(self):
        """Return None when there is no opening brace."""
        assert _extract_json("No braces at all") is None

    def test_unmatched_braces_returns_none(self):
        """Return None when braces are unmatched."""
        assert _extract_json('{"key": "value"') is None

    def test_empty_string_returns_none(self):
        """Return None for empty input."""
        assert _extract_json("") is None


# ---------------------------------------------------------------------------
# _get_client()
# ---------------------------------------------------------------------------

class TestGetClient:
    """Tests for lazy Anthropic client initialization."""

    @patch.dict("os.environ", {API_KEY_ENV_VAR: "sk-test-key"})
    def test_creates_client_with_env_var(self):
        """Create client when env var is set."""
        client = _get_client()
        assert client is not None

    @patch.dict("os.environ", {}, clear=True)
    @patch("llm.api.st")
    def test_falls_back_to_st_secrets(self, mock_st):
        """Fall back to st.secrets when env var is missing."""
        mock_st.secrets = {API_KEY_ENV_VAR: "sk-secret-key"}
        client = _get_client()
        assert client is not None

    @patch.dict("os.environ", {}, clear=True)
    @patch("llm.api.st")
    def test_raises_when_no_key_available(self, mock_st):
        """Raise LabGuardAPIError when neither env var nor secrets has key."""
        mock_st.secrets = MagicMock()
        mock_st.secrets.__getitem__ = MagicMock(side_effect=KeyError(API_KEY_ENV_VAR))
        with pytest.raises(LabGuardAPIError, match="API key is not configured"):
            _get_client()

    @patch.dict("os.environ", {}, clear=True)
    @patch("llm.api.st")
    def test_handles_missing_secrets_file(self, mock_st):
        """Handle FileNotFoundError when secrets.toml doesn't exist."""
        mock_st.secrets.__getitem__ = MagicMock(side_effect=FileNotFoundError)
        with pytest.raises(LabGuardAPIError, match="API key is not configured"):
            _get_client()

    @patch.dict("os.environ", {API_KEY_ENV_VAR: "sk-test-key"})
    def test_client_is_cached(self):
        """Second call returns the same client instance."""
        client1 = _get_client()
        client2 = _get_client()
        assert client1 is client2


# ---------------------------------------------------------------------------
# _call_api() error handling
# ---------------------------------------------------------------------------

class TestCallApi:
    """Tests for API call wrapper and exception mapping."""

    def _setup_mock_client(self, side_effect=None, return_value=None):
        """Patch _get_client to return a mock with configurable behavior."""
        mock_client = MagicMock()
        if side_effect:
            mock_client.messages.create.side_effect = side_effect
        elif return_value:
            mock_client.messages.create.return_value = return_value
        return mock_client

    @patch("llm.api._get_client")
    def test_authentication_error(self, mock_get_client):
        """Map AuthenticationError to user-friendly message."""
        import anthropic
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_client = self._setup_mock_client(
            side_effect=anthropic.AuthenticationError(
                message="Invalid API key",
                response=mock_response,
                body={"error": {"message": "Invalid API key"}},
            )
        )
        mock_get_client.return_value = mock_client
        with pytest.raises(LabGuardAPIError, match="API key is invalid"):
            _call_api("test protocol")

    @patch("llm.api._get_client")
    def test_rate_limit_error(self, mock_get_client):
        """Map RateLimitError to user-friendly message."""
        import anthropic
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_client = self._setup_mock_client(
            side_effect=anthropic.RateLimitError(
                message="Rate limited",
                response=mock_response,
                body={"error": {"message": "Rate limited"}},
            )
        )
        mock_get_client.return_value = mock_client
        with pytest.raises(LabGuardAPIError, match="temporarily overloaded"):
            _call_api("test protocol")

    @patch("llm.api._get_client")
    def test_timeout_error(self, mock_get_client):
        """Map APITimeoutError to user-friendly message."""
        import anthropic
        mock_client = self._setup_mock_client(
            side_effect=anthropic.APITimeoutError(request=MagicMock())
        )
        mock_get_client.return_value = mock_client
        with pytest.raises(LabGuardAPIError, match="timed out"):
            _call_api("test protocol")

    @patch("llm.api._get_client")
    def test_connection_error(self, mock_get_client):
        """Map APIConnectionError to user-friendly message."""
        import anthropic
        mock_client = self._setup_mock_client(
            side_effect=anthropic.APIConnectionError(request=MagicMock())
        )
        mock_get_client.return_value = mock_client
        with pytest.raises(LabGuardAPIError, match="Could not connect"):
            _call_api("test protocol")

    @patch("llm.api._get_client")
    def test_api_status_error(self, mock_get_client):
        """Map APIStatusError to user-friendly message."""
        import anthropic
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_client = self._setup_mock_client(
            side_effect=anthropic.APIStatusError(
                message="Internal server error",
                response=mock_response,
                body={"error": {"message": "Internal server error"}},
            )
        )
        mock_get_client.return_value = mock_client
        with pytest.raises(LabGuardAPIError, match="temporarily unavailable"):
            _call_api("test protocol")

    @patch("llm.api._get_client")
    def test_successful_call_returns_message(self, mock_get_client):
        """Successful API call returns the message object."""
        mock_msg = MagicMock()
        mock_client = self._setup_mock_client(return_value=mock_msg)
        mock_get_client.return_value = mock_client
        result = _call_api("test protocol")
        assert result is mock_msg


# ---------------------------------------------------------------------------
# _parse_response()
# ---------------------------------------------------------------------------

class TestParseResponse:
    """Tests for response parsing and validation."""

    def test_truncation_raises_error(self):
        """Raise error when response was truncated."""
        msg = MagicMock()
        msg.stop_reason = "max_tokens"
        with pytest.raises(LabGuardAPIError, match="too long"):
            _parse_response(msg, "protocol")

    def test_empty_content_raises_error(self):
        """Raise error when response content is empty."""
        msg = MagicMock()
        msg.stop_reason = "end_turn"
        msg.content = []
        with pytest.raises(LabGuardAPIError, match="empty response"):
            _parse_response(msg, "protocol")

    def test_no_json_in_response_raises_error(self):
        """Raise error when response contains no valid JSON."""
        msg = MagicMock()
        msg.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.text = "No JSON here, just text."
        msg.content = [text_block]
        with pytest.raises(LabGuardAPIError, match="unexpected response format"):
            _parse_response(msg, "protocol")

    def test_invalid_schema_raises_error(self):
        """Raise error when JSON doesn't match expected schema."""
        msg = MagicMock()
        msg.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.text = '{"wrong": "schema"}'
        msg.content = [text_block]
        with pytest.raises(LabGuardAPIError, match="structured incorrectly"):
            _parse_response(msg, "protocol")

    def test_valid_response_returns_analysis(self, mock_anthropic_message):
        """Valid response produces a ProtocolAnalysis object."""
        result = _parse_response(mock_anthropic_message, "test protocol")
        assert isinstance(result, ProtocolAnalysis)
        assert result.summary.total_steps == 2
        assert result.model_version == MODEL


# ---------------------------------------------------------------------------
# analyze_protocol()
# ---------------------------------------------------------------------------

class TestAnalyzeProtocol:
    """Tests for the public analyze_protocol() entry point."""

    def test_rejects_oversized_input(self):
        """Raise error for input exceeding MAX_INPUT_LENGTH."""
        long_text = "A" * (MAX_INPUT_LENGTH + 1)
        with pytest.raises(LabGuardAPIError, match="exceeds maximum length"):
            analyze_protocol.__wrapped__(long_text)

    def test_rejects_non_string_input(self):
        """Raise error for non-string input."""
        with pytest.raises(LabGuardAPIError, match="exceeds maximum length"):
            analyze_protocol.__wrapped__(12345)

    @patch("llm.api._call_api")
    def test_full_pipeline(self, mock_call, mock_anthropic_message):
        """Full pipeline with mocked API returns ProtocolAnalysis."""
        mock_call.return_value = mock_anthropic_message
        result = analyze_protocol.__wrapped__("test protocol text " * 5)
        assert isinstance(result, ProtocolAnalysis)

    @patch("llm.api._call_api")
    def test_labguard_error_passthrough(self, mock_call):
        """LabGuardAPIError from _call_api is re-raised unchanged."""
        mock_call.side_effect = LabGuardAPIError("Test error message")
        with pytest.raises(LabGuardAPIError, match="Test error message"):
            analyze_protocol.__wrapped__("test protocol text " * 5)

    @patch("llm.api._call_api")
    def test_unexpected_error_wrapped(self, mock_call):
        """Unexpected exceptions are caught and wrapped in LabGuardAPIError."""
        mock_call.side_effect = RuntimeError("something broke")
        with pytest.raises(LabGuardAPIError, match="unexpected error"):
            analyze_protocol.__wrapped__("test protocol text " * 5)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify API configuration constants."""

    def test_temperature_is_deterministic(self):
        """Temperature should be 0.0 for deterministic structured output."""
        assert TEMPERATURE == 0.0

    def test_timeout_is_reasonable(self):
        """API timeout should be set."""
        assert API_TIMEOUT > 0

    def test_max_tokens_set(self):
        """Max tokens should be configured."""
        assert MAX_TOKENS > 0

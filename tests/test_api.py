"""Tests for llm/api.py — tool-use plumbing, API calls, and response parsing."""

from unittest.mock import MagicMock, patch

import pytest

from llm.api import (
    _get_client,
    _call_api,
    _extract_tool_input,
    _parse_response,
    analyze_protocol,
    LabGuardAPIError,
    API_KEY_ENV_VAR,
    MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    API_TIMEOUT,
)
from llm.tools import ANALYSIS_TOOL, ANALYSIS_TOOL_NAME, ANALYSIS_TOOL_CHOICE
from models.schemas import ProtocolAnalysis, MAX_INPUT_LENGTH
from tests.conftest import make_claude_response_dict, make_tool_use_block


# ---------------------------------------------------------------------------
# _extract_tool_input() — pulls the forced tool_use block's input dict
# ---------------------------------------------------------------------------

class TestExtractToolInput:
    """Tests for extracting the tool_use input dict from a response."""

    def test_returns_input_from_tool_use_block(self, sample_claude_response_dict):
        """Return the input dict when a matching tool_use block is present."""
        msg = MagicMock()
        msg.content = [make_tool_use_block(sample_claude_response_dict)]
        result = _extract_tool_input(msg)
        assert result == sample_claude_response_dict

    def test_skips_text_blocks_before_tool_use(self, sample_claude_response_dict):
        """Tolerate a leading text block and still find the tool_use."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.name = None
        msg = MagicMock()
        msg.content = [text_block, make_tool_use_block(sample_claude_response_dict)]
        assert _extract_tool_input(msg) == sample_claude_response_dict

    def test_ignores_other_tool_names(self, sample_claude_response_dict):
        """Ignore tool_use blocks that don't match the analysis tool name."""
        wrong = make_tool_use_block(sample_claude_response_dict, name="other_tool")
        msg = MagicMock()
        msg.content = [wrong]
        with pytest.raises(LabGuardAPIError, match="unexpected response format"):
            _extract_tool_input(msg)

    def test_raises_when_only_text_block(self):
        """Raise when Claude replied with text instead of a tool call."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.name = None
        msg = MagicMock()
        msg.content = [text_block]
        with pytest.raises(LabGuardAPIError, match="unexpected response format"):
            _extract_tool_input(msg)

    def test_raises_on_empty_content(self):
        """Raise when the response has no content blocks at all."""
        msg = MagicMock()
        msg.content = []
        with pytest.raises(LabGuardAPIError, match="empty response"):
            _extract_tool_input(msg)


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

    @patch("llm.api._get_client")
    def test_forces_analysis_tool_use(self, mock_get_client):
        """Request forces the analysis tool via tool_choice."""
        mock_client = self._setup_mock_client(return_value=MagicMock())
        mock_get_client.return_value = mock_client
        _call_api("test protocol")
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["tools"] == [ANALYSIS_TOOL]
        assert kwargs["tool_choice"] == ANALYSIS_TOOL_CHOICE
        assert kwargs["tool_choice"]["name"] == ANALYSIS_TOOL_NAME


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
        """Raise error when the tool_use input doesn't match expected schema."""
        msg = MagicMock()
        msg.stop_reason = "tool_use"
        msg.content = [make_tool_use_block({"wrong": "schema"})]
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

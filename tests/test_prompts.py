"""Tests for llm/prompts.py — system prompt integrity and message building."""

from llm.prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_user_message,
)
from models.schemas import VALID_CATEGORIES, VALID_SEVERITIES


class TestSystemPrompt:
    """Verify system prompt contains required category and severity references."""

    def test_all_categories_present(self):
        """System prompt must reference every failure category."""
        for category in VALID_CATEGORIES:
            assert category in ANALYSIS_SYSTEM_PROMPT, (
                f"Category '{category}' missing from system prompt"
            )

    def test_all_severities_present(self):
        """System prompt must reference every severity level (case-insensitive)."""
        prompt_lower = ANALYSIS_SYSTEM_PROMPT.lower()
        for severity in VALID_SEVERITIES:
            assert severity.lower() in prompt_lower, (
                f"Severity '{severity}' missing from system prompt"
            )

    def test_prompt_version_exists(self):
        """PROMPT_VERSION is a non-empty string."""
        assert isinstance(PROMPT_VERSION, str)
        assert len(PROMPT_VERSION) > 0

    def test_prompt_forbids_generic_advice(self):
        """System prompt contains anti-generic instruction."""
        assert "generic" in ANALYSIS_SYSTEM_PROMPT.lower() or \
               "wear gloves" in ANALYSIS_SYSTEM_PROMPT.lower()

    def test_prompt_directs_tool_call(self):
        """System prompt instructs Claude to call the analysis tool."""
        assert "record_protocol_analysis" in ANALYSIS_SYSTEM_PROMPT


class TestBuildUserMessage:
    """Tests for user message construction."""

    def test_includes_protocol_text(self):
        """User message contains the protocol text verbatim."""
        text = "1. Add 200 uL of buffer to the sample."
        result = build_user_message(text)
        assert text in result

    def test_starts_with_analyze(self):
        """User message starts with analysis instruction."""
        result = build_user_message("test protocol")
        assert result.startswith("Analyze")

    def test_empty_protocol_text(self):
        """Handle empty protocol text without error."""
        result = build_user_message("")
        assert "Analyze" in result

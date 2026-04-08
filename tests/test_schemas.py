"""Tests for models/schemas.py — Pydantic models, enums, and validation."""

import pytest
from pydantic import ValidationError

from models.schemas import (
    ProtocolInput,
    Flag,
    QCCheckpoint,
    ProtocolStep,
    ProtocolAnalysis,
    ClaudeResponsePayload,
    SeverityLevel,
    FailureCategory,
    VALID_CATEGORIES,
    VALID_SEVERITIES,
    MAX_INPUT_LENGTH,
    MIN_INPUT_LENGTH,
    MIN_UNIQUE_CHARS,
)
from tests.conftest import make_claude_response_dict


# ---------------------------------------------------------------------------
# ProtocolInput validation
# ---------------------------------------------------------------------------

class TestProtocolInput:
    """Tests for ProtocolInput validation and sanitization."""

    def test_valid_input(self, sample_protocol_text):
        """Accept a well-formed protocol string."""
        result = ProtocolInput(text=sample_protocol_text)
        assert result.text == sample_protocol_text

    def test_strips_whitespace(self, sample_protocol_text):
        """Strip leading/trailing whitespace before validation."""
        padded = f"   {sample_protocol_text}   "
        result = ProtocolInput(text=padded)
        assert result.text == sample_protocol_text

    def test_min_length_boundary_reject(self):
        """Reject input below MIN_INPUT_LENGTH after stripping."""
        short = "A" * (MIN_INPUT_LENGTH - 1)
        with pytest.raises(ValidationError):
            ProtocolInput(text=short)

    def test_min_length_boundary_accept(self):
        """Accept input at exactly MIN_INPUT_LENGTH with enough unique chars."""
        # 50 chars with 10+ unique characters
        text = "ABCDEFGHIJ" * (MIN_INPUT_LENGTH // 10)
        result = ProtocolInput(text=text)
        assert len(result.text) == MIN_INPUT_LENGTH

    def test_max_length_boundary_reject(self):
        """Reject input exceeding MAX_INPUT_LENGTH."""
        # Create a long string with enough unique chars
        base = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        text = (base * ((MAX_INPUT_LENGTH + 1) // len(base) + 1))[:MAX_INPUT_LENGTH + 1]
        with pytest.raises(ValidationError):
            ProtocolInput(text=text)

    def test_max_length_boundary_accept(self):
        """Accept input at exactly MAX_INPUT_LENGTH."""
        base = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        text = (base * (MAX_INPUT_LENGTH // len(base) + 1))[:MAX_INPUT_LENGTH]
        result = ProtocolInput(text=text)
        assert len(result.text) == MAX_INPUT_LENGTH

    def test_rejects_null_byte(self):
        """Reject input containing null bytes."""
        text = "Valid protocol text\x00 with null byte" + "x" * 50
        with pytest.raises(ValidationError):
            ProtocolInput(text=text)

    def test_rejects_control_characters(self):
        """Reject input with control characters (not \\n, \\t, \\r)."""
        for char in ["\x01", "\x08", "\x0b", "\x0c", "\x0e", "\x1f", "\x7f"]:
            text = f"Valid protocol text {char} here" + "x" * 50
            with pytest.raises(ValidationError, match="invalid characters"):
                ProtocolInput(text=text)

    def test_allows_newline_tab_cr(self):
        """Preserve newline, tab, and carriage return in valid input."""
        text = "Step 1: Do something\n\tDetails here\r\n" + "Step 2: " + "x" * 40
        result = ProtocolInput(text=text)
        assert "\n" in result.text
        assert "\t" in result.text

    def test_rejects_spam_low_unique_chars(self):
        """Reject input with fewer than MIN_UNIQUE_CHARS distinct non-whitespace chars."""
        # 100 chars but only 1 unique non-whitespace character
        text = "A" * 100
        with pytest.raises(ValidationError, match="valid lab protocol"):
            ProtocolInput(text=text)

    def test_rejects_spam_boundary(self):
        """Reject input with exactly MIN_UNIQUE_CHARS - 1 distinct chars."""
        # 9 unique chars repeated to 100 total
        text = ("ABCDEFGHI" * 12)[:100]
        with pytest.raises(ValidationError, match="valid lab protocol"):
            ProtocolInput(text=text)

    def test_accepts_min_unique_chars(self):
        """Accept input meeting the MIN_UNIQUE_CHARS threshold."""
        text = ("ABCDEFGHIJ" * 10)[:100]
        result = ProtocolInput(text=text)
        assert len(set(result.text.replace(" ", ""))) >= MIN_UNIQUE_CHARS

    def test_empty_string_rejected(self):
        """Reject empty string input."""
        with pytest.raises(ValidationError):
            ProtocolInput(text="")

    def test_whitespace_only_rejected(self):
        """Reject whitespace-only input (stripped to empty)."""
        with pytest.raises(ValidationError):
            ProtocolInput(text="     ")


# ---------------------------------------------------------------------------
# Flag validation
# ---------------------------------------------------------------------------

class TestFlag:
    """Tests for Flag model validation."""

    def _valid_flag_kwargs(self):
        """Return kwargs for a valid Flag."""
        return {
            "category": FailureCategory.CONTAMINATION_RISK,
            "severity": SeverityLevel.CRITICAL,
            "title": "Test flag title",
            "description": "Detailed description of the issue.",
            "affected_text": "Some affected protocol text",
            "suggested_fix": "Do this instead.",
        }

    def test_valid_flag(self):
        """Accept a well-formed flag."""
        flag = Flag(**self._valid_flag_kwargs())
        assert flag.category == FailureCategory.CONTAMINATION_RISK
        assert flag.severity == SeverityLevel.CRITICAL

    def test_empty_title_rejected(self):
        """Reject flag with empty title."""
        kwargs = self._valid_flag_kwargs()
        kwargs["title"] = ""
        with pytest.raises(ValidationError):
            Flag(**kwargs)

    def test_whitespace_only_title_rejected(self):
        """Reject flag with whitespace-only title."""
        kwargs = self._valid_flag_kwargs()
        kwargs["title"] = "   "
        with pytest.raises(ValidationError):
            Flag(**kwargs)

    def test_empty_description_rejected(self):
        """Reject flag with empty description."""
        kwargs = self._valid_flag_kwargs()
        kwargs["description"] = ""
        with pytest.raises(ValidationError):
            Flag(**kwargs)

    def test_empty_affected_text_rejected(self):
        """Reject flag with empty affected_text."""
        kwargs = self._valid_flag_kwargs()
        kwargs["affected_text"] = ""
        with pytest.raises(ValidationError):
            Flag(**kwargs)

    def test_empty_suggested_fix_rejected(self):
        """Reject flag with empty suggested_fix."""
        kwargs = self._valid_flag_kwargs()
        kwargs["suggested_fix"] = ""
        with pytest.raises(ValidationError):
            Flag(**kwargs)

    def test_title_max_length(self):
        """Accept title at exactly 120 chars, reject at 121."""
        kwargs = self._valid_flag_kwargs()
        kwargs["title"] = "A" * 120
        flag = Flag(**kwargs)
        assert len(flag.title) == 120

        kwargs["title"] = "A" * 121
        with pytest.raises(ValidationError):
            Flag(**kwargs)

    def test_invalid_category_rejected(self):
        """Reject flag with invalid category string."""
        kwargs = self._valid_flag_kwargs()
        kwargs["category"] = "fake_category"
        with pytest.raises(ValidationError):
            Flag(**kwargs)

    def test_invalid_severity_rejected(self):
        """Reject flag with invalid severity string."""
        kwargs = self._valid_flag_kwargs()
        kwargs["severity"] = "extreme"
        with pytest.raises(ValidationError):
            Flag(**kwargs)

    def test_flag_id_defaults_empty(self):
        """Flag ID defaults to empty string (assigned client-side)."""
        flag = Flag(**self._valid_flag_kwargs())
        assert flag.flag_id == ""


# ---------------------------------------------------------------------------
# QCCheckpoint validation
# ---------------------------------------------------------------------------

class TestQCCheckpoint:
    """Tests for QCCheckpoint model validation."""

    def _valid_qc_kwargs(self):
        """Return kwargs for a valid QCCheckpoint."""
        return {
            "after_step": 1,
            "action": "Measure absorbance at 260nm",
            "expected_result": "A260/A280 ratio between 1.8 and 2.0",
            "failure_action": "Re-extract DNA from a new sample",
        }

    def test_valid_checkpoint(self):
        """Accept a well-formed QC checkpoint."""
        qc = QCCheckpoint(**self._valid_qc_kwargs())
        assert qc.after_step == 1

    def test_after_step_zero_rejected(self):
        """Reject checkpoint with after_step < 1."""
        kwargs = self._valid_qc_kwargs()
        kwargs["after_step"] = 0
        with pytest.raises(ValidationError):
            QCCheckpoint(**kwargs)

    def test_after_step_large_value(self):
        """Accept checkpoint with large step number."""
        kwargs = self._valid_qc_kwargs()
        kwargs["after_step"] = 999
        qc = QCCheckpoint(**kwargs)
        assert qc.after_step == 999

    def test_empty_action_rejected(self):
        """Reject checkpoint with empty action."""
        kwargs = self._valid_qc_kwargs()
        kwargs["action"] = "  "
        with pytest.raises(ValidationError):
            QCCheckpoint(**kwargs)

    def test_empty_expected_result_rejected(self):
        """Reject checkpoint with empty expected_result."""
        kwargs = self._valid_qc_kwargs()
        kwargs["expected_result"] = ""
        with pytest.raises(ValidationError):
            QCCheckpoint(**kwargs)

    def test_empty_failure_action_rejected(self):
        """Reject checkpoint with empty failure_action."""
        kwargs = self._valid_qc_kwargs()
        kwargs["failure_action"] = ""
        with pytest.raises(ValidationError):
            QCCheckpoint(**kwargs)

    def test_checkpoint_id_defaults_empty(self):
        """Checkpoint ID defaults to empty string (assigned client-side)."""
        qc = QCCheckpoint(**self._valid_qc_kwargs())
        assert qc.checkpoint_id == ""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
    """Tests for SeverityLevel and FailureCategory enums."""

    def test_severity_values(self):
        """SeverityLevel has exactly 3 valid values."""
        assert set(VALID_SEVERITIES) == {"critical", "warning", "info"}

    def test_failure_category_count(self):
        """FailureCategory has exactly 12 values."""
        assert len(VALID_CATEGORIES) == 12
        assert len(FailureCategory) == 12

    def test_severity_string_access(self):
        """Enum values match their string representation."""
        assert SeverityLevel.CRITICAL.value == "critical"
        assert SeverityLevel.WARNING.value == "warning"
        assert SeverityLevel.INFO.value == "info"

    def test_all_categories_in_enum(self):
        """Every VALID_CATEGORIES entry maps to a FailureCategory member."""
        for cat in VALID_CATEGORIES:
            assert FailureCategory(cat) is not None


# ---------------------------------------------------------------------------
# ProtocolAnalysis.from_claude_response()
# ---------------------------------------------------------------------------

class TestProtocolAnalysisFactory:
    """Tests for ProtocolAnalysis.from_claude_response() factory method."""

    def _build_analysis(self, **kwargs):
        """Build a ProtocolAnalysis from a claude response dict."""
        response_dict = make_claude_response_dict(**kwargs)
        payload = ClaudeResponsePayload.model_validate(response_dict)
        return ProtocolAnalysis.from_claude_response(
            payload, "test protocol text", "test-model"
        )

    def test_flag_id_format(self):
        """Flag IDs follow F-{step}-{seq} format."""
        analysis = self._build_analysis(num_steps=2, flags_per_step=3)
        step1_flags = analysis.steps[0].flags
        assert step1_flags[0].flag_id == "F-1-1"
        assert step1_flags[1].flag_id == "F-1-2"
        assert step1_flags[2].flag_id == "F-1-3"
        step2_flags = analysis.steps[1].flags
        assert step2_flags[0].flag_id == "F-2-1"

    def test_checkpoint_id_format(self):
        """Checkpoint IDs follow QC-{step}-{seq} format."""
        analysis = self._build_analysis(num_steps=2, qc_per_step=2)
        qcs = analysis.steps[0].qc_checkpoints
        assert qcs[0].checkpoint_id == "QC-1-1"
        assert qcs[1].checkpoint_id == "QC-1-2"

    def test_summary_total_steps(self):
        """Summary total_steps matches actual step count."""
        analysis = self._build_analysis(num_steps=5, flags_per_step=0)
        assert analysis.summary.total_steps == 5

    def test_summary_total_flags(self):
        """Summary total_flags counts all flags across steps."""
        analysis = self._build_analysis(num_steps=3, flags_per_step=2)
        assert analysis.summary.total_flags == 6

    def test_summary_critical_count(self):
        """Summary correctly counts critical flags."""
        # make_claude_response_dict: step 1 gets "critical", others get "warning"
        analysis = self._build_analysis(num_steps=3, flags_per_step=1)
        assert analysis.summary.critical_count == 1
        assert analysis.summary.warning_count == 2

    def test_summary_qc_count(self):
        """Summary total_qc_checkpoints counts all checkpoints."""
        analysis = self._build_analysis(num_steps=2, qc_per_step=3)
        assert analysis.summary.total_qc_checkpoints == 6

    def test_categories_detected_deduplication(self):
        """categories_detected contains unique categories preserving order."""
        # All flags in make_claude_response_dict use contamination_risk
        analysis = self._build_analysis(num_steps=3, flags_per_step=2)
        assert analysis.summary.categories_detected == [
            FailureCategory.CONTAMINATION_RISK
        ]

    def test_analysis_id_is_uuid(self):
        """analysis_id is a valid UUID string."""
        import uuid
        analysis = self._build_analysis()
        uuid.UUID(analysis.analysis_id)  # raises ValueError if invalid

    def test_timestamp_is_iso_format(self):
        """timestamp is a parseable ISO 8601 string."""
        from datetime import datetime
        analysis = self._build_analysis()
        datetime.fromisoformat(analysis.timestamp)

    def test_input_text_preserved(self):
        """input_text matches the protocol text passed to the factory."""
        response_dict = make_claude_response_dict()
        payload = ClaudeResponsePayload.model_validate(response_dict)
        analysis = ProtocolAnalysis.from_claude_response(
            payload, "my exact protocol", "model-v1"
        )
        assert analysis.input_text == "my exact protocol"

    def test_model_version_preserved(self):
        """model_version matches what was passed to the factory."""
        response_dict = make_claude_response_dict()
        payload = ClaudeResponsePayload.model_validate(response_dict)
        analysis = ProtocolAnalysis.from_claude_response(
            payload, "text", "claude-sonnet-4-20250514"
        )
        assert analysis.model_version == "claude-sonnet-4-20250514"

    def test_empty_flags_and_checkpoints(self):
        """Steps with no flags or checkpoints produce zero counts."""
        analysis = self._build_analysis(num_steps=2, flags_per_step=0, qc_per_step=0)
        assert analysis.summary.total_flags == 0
        assert analysis.summary.total_qc_checkpoints == 0
        assert analysis.summary.critical_count == 0
        assert analysis.summary.categories_detected == []

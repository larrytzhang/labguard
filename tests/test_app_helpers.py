"""Tests for app.py helper functions — severity_icon and _check_rate_limit.

app.py calls st.set_page_config() and creates widgets on import, so we
build a comprehensive Streamlit mock before importing any app functions.
"""

import importlib
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from models.schemas import Flag, SeverityLevel, FailureCategory


# ---------------------------------------------------------------------------
# Streamlit mock — supports attribute-access session state and widget returns
# ---------------------------------------------------------------------------

class _SessionStateMock(dict):
    """Dict subclass that also supports attribute access (like st.session_state)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


def _build_streamlit_mock(session_state=None):
    """Return a MagicMock that satisfies app.py's top-level Streamlit calls."""
    mock_st = MagicMock()
    mock_st.session_state = session_state or _SessionStateMock()
    # st.columns([3, 1]) must return an iterable of 2 context managers
    mock_st.columns.return_value = [MagicMock(), MagicMock()]
    # st.sidebar returns itself for chained calls
    mock_st.sidebar = MagicMock()
    mock_st.sidebar.button.return_value = False
    # st.text_area returns empty string by default
    mock_st.text_area.return_value = ""
    # st.button returns False by default
    mock_st.button.return_value = False
    # st.secrets — raise FileNotFoundError to simulate missing secrets file
    mock_st.secrets.__getitem__ = MagicMock(side_effect=FileNotFoundError)
    return mock_st


def _import_app_with_mock(mock_st):
    """Import (or reload) app.py with the given Streamlit mock injected."""
    with patch.dict(sys.modules, {"streamlit": mock_st}):
        if "app" in sys.modules:
            return importlib.reload(sys.modules["app"])
        import app as app_module
        return app_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flag(severity: SeverityLevel) -> Flag:
    """Create a minimal valid Flag with the given severity."""
    return Flag(
        category=FailureCategory.CONTAMINATION_RISK,
        severity=severity,
        title="Test flag",
        description="Test description.",
        affected_text="Affected text",
        suggested_fix="Fix suggestion.",
    )


# ---------------------------------------------------------------------------
# severity_icon() tests
# ---------------------------------------------------------------------------

class TestSeverityIcon:
    """Tests for severity_icon() from app.py."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        """Import severity_icon with Streamlit mocked."""
        mock_st = _build_streamlit_mock()
        app_mod = _import_app_with_mock(mock_st)
        self.severity_icon = app_mod.severity_icon

    def test_empty_list_returns_green(self):
        """Empty flag list returns green circle emoji."""
        assert self.severity_icon([]) == "\U0001f7e2"

    def test_info_only_returns_blue(self):
        """List with only INFO flags returns blue circle."""
        flags = [_make_flag(SeverityLevel.INFO)]
        assert self.severity_icon(flags) == "\U0001f535"

    def test_warning_returns_yellow(self):
        """List with WARNING (no CRITICAL) returns yellow circle."""
        flags = [
            _make_flag(SeverityLevel.WARNING),
            _make_flag(SeverityLevel.INFO),
        ]
        assert self.severity_icon(flags) == "\U0001f7e1"

    def test_critical_returns_red(self):
        """List with any CRITICAL flag returns red circle."""
        flags = [_make_flag(SeverityLevel.CRITICAL)]
        assert self.severity_icon(flags) == "\U0001f534"

    def test_critical_overrides_warning(self):
        """CRITICAL takes precedence over WARNING."""
        flags = [
            _make_flag(SeverityLevel.WARNING),
            _make_flag(SeverityLevel.CRITICAL),
            _make_flag(SeverityLevel.INFO),
        ]
        assert self.severity_icon(flags) == "\U0001f534"

    def test_single_warning_no_critical(self):
        """Single WARNING without CRITICAL returns yellow, not red."""
        flags = [_make_flag(SeverityLevel.WARNING)]
        assert self.severity_icon(flags) == "\U0001f7e1"


# ---------------------------------------------------------------------------
# _check_rate_limit() tests
# ---------------------------------------------------------------------------

class TestCheckRateLimit:
    """Tests for session-based rate limiting."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        """Import _check_rate_limit with Streamlit mocked."""
        self.session_state = _SessionStateMock()
        mock_st = _build_streamlit_mock(session_state=self.session_state)
        app_mod = _import_app_with_mock(mock_st)
        self._check_rate_limit = app_mod._check_rate_limit
        self.RATE_LIMIT_MAX = app_mod.RATE_LIMIT_MAX_REQUESTS
        self.RATE_LIMIT_WINDOW = app_mod.RATE_LIMIT_WINDOW_SECONDS
        # Ensure app module's st.session_state points to our mock
        app_mod.st.session_state = self.session_state

    def test_first_call_allowed(self):
        """First call with no prior timestamps is allowed."""
        allowed, msg = self._check_rate_limit()
        assert allowed is True
        assert msg == ""

    def test_within_limit_allowed(self):
        """Calls within the rate limit are allowed."""
        now = datetime.now()
        self.session_state["request_timestamps"] = [
            now - timedelta(seconds=i) for i in range(5)
        ]
        allowed, msg = self._check_rate_limit()
        assert allowed is True

    def test_at_limit_rejected(self):
        """Call at exactly the rate limit is rejected."""
        now = datetime.now()
        self.session_state["request_timestamps"] = [
            now - timedelta(seconds=i)
            for i in range(self.RATE_LIMIT_MAX)
        ]
        allowed, msg = self._check_rate_limit()
        assert allowed is False
        assert "Rate limit" in msg

    def test_retry_after_is_positive(self):
        """Rejected response includes a positive retry_after value."""
        now = datetime.now()
        self.session_state["request_timestamps"] = [
            now - timedelta(seconds=i)
            for i in range(self.RATE_LIMIT_MAX)
        ]
        allowed, msg = self._check_rate_limit()
        assert allowed is False
        assert "seconds" in msg

    def test_old_timestamps_purged(self):
        """Timestamps outside the rolling window are removed."""
        now = datetime.now()
        self.session_state["request_timestamps"] = [
            now - timedelta(seconds=self.RATE_LIMIT_WINDOW + 10 + i)
            for i in range(self.RATE_LIMIT_MAX)
        ]
        allowed, msg = self._check_rate_limit()
        assert allowed is True


# ---------------------------------------------------------------------------
# Example protocol validation
# ---------------------------------------------------------------------------

class TestExampleProtocols:
    """Verify example protocols pass input validation."""

    def test_all_examples_are_valid_input(self):
        """Each example protocol is valid ProtocolInput."""
        from data.examples import (
            EXAMPLE_CELL_CULTURE,
            EXAMPLE_DNA_EXTRACTION,
            EXAMPLE_TRANSFECTION,
            EXAMPLE_WESTERN_BLOT,
        )
        from models.schemas import ProtocolInput

        for name, text in [
            ("DNA Extraction", EXAMPLE_DNA_EXTRACTION),
            ("Transfection", EXAMPLE_TRANSFECTION),
            ("Western Blot", EXAMPLE_WESTERN_BLOT),
            ("Cell Culture", EXAMPLE_CELL_CULTURE),
        ]:
            result = ProtocolInput(text=text)
            assert len(result.text) >= 50, f"{name} is too short"

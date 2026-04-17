"""LabGuard — Streamlit UI for protocol failure prediction.

Layout, session state, rendering, and session-based rate limiting.
No API calls or prompts.
"""

import html
import logging
import os
from datetime import datetime, timedelta

import streamlit as st
from pydantic import ValidationError

from data.examples import (
    EXAMPLE_CELL_CULTURE,
    EXAMPLE_DNA_EXTRACTION,
    EXAMPLE_TRANSFECTION,
    EXAMPLE_WESTERN_BLOT,
)
from llm.api import analyze_protocol, LabGuardAPIError
from models.schemas import (
    ProtocolAnalysis,
    ProtocolInput,
    Flag,
    QCCheckpoint,
    SeverityLevel,
    FailureCategory,
    MAX_INPUT_LENGTH,
    MIN_INPUT_LENGTH,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting — session-based throttle prevents API cost abuse.
# Each browser tab gets its own Streamlit session. Resets on page refresh.
# IP-based limiting requires persistent storage, deferred for MVP.
# ---------------------------------------------------------------------------

RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LabGuard - Protocol Failure Prediction",
    page_icon="\U0001f52c",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .badge-critical {
        background: #DC2626; color: white;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.8em; font-weight: 600;
    }
    .badge-warning {
        background: #D97706; color: white;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.8em; font-weight: 600;
    }
    .badge-info {
        background: #2563EB; color: white;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.8em; font-weight: 600;
    }
    .qc-card {
        border-left: 4px solid #059669;
        background: rgba(5, 150, 105, 0.08);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
    .footer {
        text-align: center; opacity: 0.5; color: inherit;
        font-size: 0.8em; padding: 24px 0 12px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "protocol_text" not in st.session_state:
    st.session_state.protocol_text = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None

# ---------------------------------------------------------------------------
# API key check — warn early instead of failing on first analysis
# ---------------------------------------------------------------------------


def _api_key_configured() -> bool:
    """Check if the Anthropic API key is available from env or Streamlit secrets."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    try:
        return bool(st.secrets["ANTHROPIC_API_KEY"])
    except (KeyError, FileNotFoundError):
        return False


if not _api_key_configured():
    st.warning(
        "**API key not found.** Set `ANTHROPIC_API_KEY` as an environment "
        "variable or in `.streamlit/secrets.toml` to enable analysis.",
        icon="\U0001f511",
    )

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def severity_icon(flags: list[Flag]) -> str:
    """Return an emoji representing the worst severity in the flag list."""
    if any(f.severity == SeverityLevel.CRITICAL for f in flags):
        return "\U0001f534"
    if any(f.severity == SeverityLevel.WARNING for f in flags):
        return "\U0001f7e1"
    if flags:
        return "\U0001f535"
    return "\U0001f7e2"


def render_flag(flag: Flag) -> None:
    """Render a single failure flag with severity badge and details."""
    badge_class = f"badge-{flag.severity.value}"
    category_label = flag.category.value.replace("_", " ").title()

    safe_title = html.escape(flag.title)
    safe_description = html.escape(flag.description)
    safe_affected = html.escape(flag.affected_text)
    safe_fix = html.escape(flag.suggested_fix)
    st.markdown(
        f'<span class="{badge_class}">{flag.severity.value.upper()}</span>'
        f" &nbsp; **{category_label}** — {safe_title}",
        unsafe_allow_html=True,
    )
    st.markdown(safe_description)
    st.markdown(
        f"**Affected text:** *\"{safe_affected}\"*"
    )
    st.markdown(f"> **Suggested fix:** {safe_fix}")
    st.markdown("---")


def _check_rate_limit() -> tuple[bool, str]:
    """Check session-based rate limit. Returns (allowed, error_message)."""
    now = datetime.now()
    if "request_timestamps" not in st.session_state:
        st.session_state.request_timestamps = []

    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)
    # Purge timestamps outside the rolling window
    st.session_state.request_timestamps = [
        ts for ts in st.session_state.request_timestamps
        if ts > window_start
    ]

    if len(st.session_state.request_timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        # Calculate retry-after for a user-friendly 429-style message
        oldest = st.session_state.request_timestamps[0]
        retry_after = int(
            (oldest + timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS) - now
             ).total_seconds()
        ) + 1
        return False, (
            f"Rate limit reached ({RATE_LIMIT_MAX_REQUESTS} requests per "
            f"{RATE_LIMIT_WINDOW_SECONDS}s). "
            f"Please wait {retry_after} seconds before trying again."
        )

    st.session_state.request_timestamps.append(now)
    return True, ""


def render_qc_checkpoint(qc: QCCheckpoint) -> None:
    """Render a QC checkpoint card between protocol steps."""
    safe_action = html.escape(qc.action)
    safe_expected = html.escape(qc.expected_result)
    safe_failure = html.escape(qc.failure_action)
    st.markdown(
        f"""<div class="qc-card">
        <strong>\u2705 QC Checkpoint</strong> (after Step {qc.after_step})<br/>
        <strong>Action:</strong> {safe_action}<br/>
        <strong>Expected:</strong> {safe_expected}<br/>
        <strong>If failed:</strong> {safe_failure}
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("LabGuard")
st.sidebar.caption("Protocol Failure Prediction Agent")

st.sidebar.divider()
st.sidebar.subheader("Example Protocols")
st.sidebar.caption("Click to load an example and see LabGuard in action.")

def _load_example(protocol_text: str) -> None:
    """Load an example protocol and clear previous analysis state."""
    st.session_state.protocol_text = protocol_text
    st.session_state.analysis_result = None
    st.session_state.analysis_error = None


if st.sidebar.button(
    "DNA Extraction (Silica Column)", use_container_width=True
):
    _load_example(EXAMPLE_DNA_EXTRACTION)

if st.sidebar.button("HeLa Cell Transfection", use_container_width=True):
    _load_example(EXAMPLE_TRANSFECTION)

if st.sidebar.button(
    "Western Blot (Semi-Dry Transfer)", use_container_width=True
):
    _load_example(EXAMPLE_WESTERN_BLOT)

if st.sidebar.button(
    "Cell Culture (Routine Passage)", use_container_width=True
):
    _load_example(EXAMPLE_CELL_CULTURE)

# Filters — only visible when results exist
if st.session_state.get("analysis_result"):
    st.sidebar.divider()
    st.sidebar.subheader("Filter Results")

    severity_options = [s.value for s in SeverityLevel]
    severity_filter = st.sidebar.multiselect(
        "Severity",
        options=severity_options,
        default=severity_options,
        key="severity_filter",
    )

    category_options = [c.value for c in FailureCategory]
    category_filter = st.sidebar.multiselect(
        "Category",
        options=category_options,
        default=category_options,
        format_func=lambda x: x.replace("_", " ").title(),
        key="category_filter",
    )
else:
    severity_filter = [s.value for s in SeverityLevel]
    category_filter = [c.value for c in FailureCategory]

# About section
st.sidebar.divider()
st.sidebar.markdown(
    "**About LabGuard**\n\n"
    "AI-powered protocol failure prediction for bench scientists.\n\n"
    "Built on Claude's protocol analysis capabilities "
    "([0.83 Protocol QA benchmark]"
    "(https://www.anthropic.com/research/claude-for-life-sciences), "
    "exceeding human baseline of 0.79)."
)
st.sidebar.caption("Built by Larry Zhang | Powered by Claude API")

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.title("LabGuard")
st.markdown(
    "Paste a lab protocol below. LabGuard will parse it into steps, "
    "flag potential failure points, and suggest quality-control checkpoints."
)

# ---------------------------------------------------------------------------
# Main input
# ---------------------------------------------------------------------------

protocol_input = st.text_area(
    "Protocol / Methods Section",
    value=st.session_state.get("protocol_text", ""),
    height=250,
    placeholder=(
        "Paste your protocol here. Example:\n\n"
        "1. Lyse cells in 200 uL Buffer AL at 56C for 10 min..."
    ),
    max_chars=MAX_INPUT_LENGTH,
)

st.session_state.protocol_text = protocol_input

col_count, col_button = st.columns([3, 1])

char_count = len(protocol_input)

with col_count:
    if 0 < char_count < MIN_INPUT_LENGTH:
        st.caption(
            f"{char_count:,} / {MAX_INPUT_LENGTH:,} characters — "
            f"minimum {MIN_INPUT_LENGTH} required"
        )
    elif char_count > 0:
        st.caption(f"{char_count:,} / {MAX_INPUT_LENGTH:,} characters")

with col_button:
    analyze_disabled = char_count < MIN_INPUT_LENGTH
    analyze_clicked = st.button(
        "Analyze Protocol",
        type="primary",
        use_container_width=True,
        disabled=analyze_disabled,
    )

# ---------------------------------------------------------------------------
# Analysis trigger
# ---------------------------------------------------------------------------

if analyze_clicked and not analyze_disabled:
    # Rate limit: prevent API abuse (session-based, graceful 429-style rejection)
    allowed, rate_limit_msg = _check_rate_limit()
    if not allowed:
        st.session_state.analysis_result = None
        st.session_state.analysis_error = rate_limit_msg
    else:
        # Validate and sanitize input (schema-based: type, length, content checks)
        try:
            validated = ProtocolInput(text=protocol_input)
        except ValidationError:
            st.session_state.analysis_result = None
            st.session_state.analysis_error = (
                "Invalid input. Please paste a real lab protocol."
            )
            validated = None

        if validated is not None:
            st.session_state.analysis_result = None
            st.session_state.analysis_error = None
            with st.spinner(
                "Analyzing protocol... Claude is reading your methods section."
            ):
                try:
                    result = analyze_protocol(validated.text)
                    st.session_state.analysis_result = result
                except LabGuardAPIError as exc:
                    st.session_state.analysis_error = exc.user_message
                except Exception:
                    logger.exception("Unexpected error during analysis")
                    st.session_state.analysis_error = (
                        "An unexpected error occurred. Please try again."
                    )
    st.rerun()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if st.session_state.get("analysis_error"):
    error_msg = st.session_state.analysis_error
    # Rate limit and validation errors get warning style (user action needed),
    # API errors get error style (transient/system issue)
    if "Rate limit" in error_msg:
        st.warning(error_msg, icon="\u23f3")
    elif "Invalid input" in error_msg:
        st.warning(error_msg, icon="\u26a0\ufe0f")
    else:
        st.error(error_msg)

elif st.session_state.get("analysis_result"):
    result: ProtocolAnalysis = st.session_state.analysis_result

    # --- Protocol metadata ---
    st.divider()
    meta = result.metadata
    meta_parts = [f"**{html.escape(meta.protocol_title)}**"]
    if meta.protocol_type and meta.protocol_type != "general":
        meta_parts.append(html.escape(meta.protocol_type))
    if meta.organism:
        meta_parts.append(html.escape(meta.organism))
    if meta.estimated_duration:
        meta_parts.append(html.escape(meta.estimated_duration))
    st.markdown(" | ".join(meta_parts))
    if meta.technique_tags:
        st.caption(", ".join(html.escape(t) for t in meta.technique_tags))

    # --- Summary bar ---
    st.subheader("Analysis Summary")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Steps Parsed", result.summary.total_steps)
    m2.metric("Critical Flags", result.summary.critical_count)
    m3.metric("Warnings", result.summary.warning_count)
    m4.metric("QC Checkpoints", result.summary.total_qc_checkpoints)

    # --- Overall assessment ---
    if result.overall_assessment:
        st.info(html.escape(result.overall_assessment), icon="\U0001f4cb")

    # --- Step-by-step walkthrough ---
    st.subheader("Step-by-Step Analysis")

    # Build QC checkpoint lookup
    qc_by_step: dict[int, list[QCCheckpoint]] = {}
    for step in result.steps:
        if step.qc_checkpoints:
            qc_by_step[step.step_number] = step.qc_checkpoints

    for step in result.steps:
        # Apply sidebar filters
        visible_flags = [
            f
            for f in step.flags
            if f.severity.value in severity_filter
            and f.category.value in category_filter
        ]

        icon = severity_icon(visible_flags)
        step_label = step.original_text[:80]
        if len(step.original_text) > 80:
            step_label += "..."

        has_visible_critical = any(
            f.severity == SeverityLevel.CRITICAL for f in visible_flags
        )
        with st.expander(
            f"{icon}  Step {step.step_number}: {step_label}",
            expanded=has_visible_critical,
        ):
            st.markdown(
                f"**Protocol text:** {html.escape(step.original_text)}"
            )

            if not visible_flags:
                st.success(
                    "No issues found for current filter settings.",
                    icon="\u2705",
                )
            else:
                for flag in visible_flags:
                    render_flag(flag)

        # QC checkpoints after this step
        if step.step_number in qc_by_step:
            for qc in qc_by_step[step.step_number]:
                render_qc_checkpoint(qc)

    # Zero flags state
    if result.summary.total_flags == 0:
        st.success(
            "No potential failure points detected. "
            "This protocol appears well-specified. "
            "Consider having a colleague review it as an additional check.",
            icon="\u2705",
        )

    # --- Download results ---
    st.divider()
    st.download_button(
        label="Download Results (JSON)",
        data=result.model_dump_json(indent=2),
        file_name=f"labguard-analysis-{result.analysis_id[:8]}.json",
        mime="application/json",
    )

else:
    # Empty state — first visit
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #6B7280; padding: 40px 0;">'
        "Paste a protocol above and click "
        "<strong>Analyze Protocol</strong> to get started. "
        "Or try an example from the sidebar."
        "</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    '<div class="footer">Powered by Claude API &nbsp;|&nbsp; Built by Larry Zhang</div>',
    unsafe_allow_html=True,
)

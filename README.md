# LabGuard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=larrytzhang/labguard&branch=main&mainModule=app.py)

Protocol failure prediction for life sciences researchers. Paste a lab protocol, get expert-level failure analysis in seconds.

## The Problem

Written protocols almost never capture the implicit knowledge that experienced bench scientists accumulate over years of failed experiments. Junior researchers waste weeks, expensive reagents, and irreplaceable samples because this "bench intuition" lives in people's heads, not in methods sections.

## What LabGuard Does

LabGuard takes a methods section or lab protocol and flags real failure points that textbooks don't mention.

- **Parses** protocols into discrete steps (numbered, narrative, bulleted, mixed formats)
- **Flags** issues across 12 failure categories: contamination risk, reagent incompatibility, missing controls, temperature sensitivity, ambiguous quantities, timing dependencies, equipment assumptions, safety hazards, regulatory compliance, sample integrity, reproducibility, and validation gaps
- **Rates** each flag as Critical / Warning / Info with specific chemical and biological reasoning
- **Suggests** QC checkpoints at critical junctures with measurable pass/fail criteria
- **Exports** results as JSON for documentation

Every flag meets the "bench intuition" standard -- no generic advice like "wear gloves." Flags cite specific mechanisms, thresholds, and consequences. Example: *"EDTA chelates Mg2+ needed by Taq -- expect failed PCR amplification."*

## Why Not Just Ask Claude Directly?

You could paste a protocol into Claude and ask "what could go wrong." LabGuard adds:

- **Structured 12-category taxonomy** with severity calibration, not freeform text
- **Prompt engineering** tuned for bench-level specificity ([`llm/prompts.py`](llm/prompts.py)) -- the system prompt enforces domain expertise and rejects generic advice
- **QC checkpoint placement** at irreversible steps, before expensive downstream steps, with measurable criteria
- **Filtering** by severity and failure category
- **Validated output** through Pydantic schemas -- malformed LLM responses are caught and retried, never shown to the user

Built on Claude's protocol analysis capabilities ([0.83 Protocol QA benchmark](https://www.anthropic.com/research/claude-for-life-sciences), exceeding human baseline of 0.79).

## Architecture

```
app.py              Streamlit UI -- layout, rendering, session state
llm/
  prompts.py        System prompt -- single-shot parse + analyze
  tools.py          Anthropic tool schema (generated from Pydantic models)
  api.py            Anthropic API calls with forced tool use, error handling
models/
  schemas.py        Pydantic models, enums, validation -- single source of truth
data/
  examples.py       Example protocols (DNA extraction, transfection, western blot, cell culture)
tests/              97 tests, all runnable without an API key
```

Clean separation: UI touches no prompts or API calls. API layer touches no rendering. Prompts layer owns all prompt text. Schemas layer owns all types.

## Tech Stack

- **Claude API** (claude-sonnet-4-20250514) -- forced tool use for guaranteed schema-conformant output, 12-category failure taxonomy
- **Streamlit** -- UI framework
- **Pydantic v2** -- response validation and type safety

## Run Locally

```bash
git clone https://github.com/larrytzhang/labguard.git
cd labguard
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key
streamlit run app.py
```

## Run Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Built By Larry Zhang
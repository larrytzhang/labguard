# LabGuard

Protocol failure prediction for life sciences researchers. Paste a lab protocol, get back expert-level failure analysis in seconds.

## What It Does

LabGuard takes a methods section or lab protocol and runs it through Claude to identify real failure points that experienced bench scientists catch but written protocols rarely mention.

- **Parses** protocols into discrete steps (handles numbered, narrative, bulleted, and mixed formats)
- **Flags** 12 categories of failure: contamination risk, reagent incompatibility, missing controls, temperature sensitivity, ambiguous quantities, and more
- **Rates** each flag as Critical / Warning / Info with specific chemical and biological reasoning
- **Suggests** QC checkpoints at critical junctures with pass/fail criteria

Every flag meets the "bench intuition" standard — no generic advice like "wear gloves." Flags cite specific mechanisms, thresholds, and consequences.

## Architecture

```
app.py              Streamlit UI — layout, rendering, session state
llm/
  prompts.py        System prompt (single-shot parse + analyze)
  api.py            Anthropic API calls, JSON extraction, error handling
models/
  schemas.py        Pydantic models, enums, validation — single source of truth
```

Clean separation: UI touches no prompts or API calls. API layer touches no rendering. Prompts layer owns all prompt text. Schemas layer owns all types.

## Tech Stack

- **Claude API** (claude-sonnet-4-20250514) — structured JSON output, 12-category failure taxonomy
- **Streamlit** — UI framework
- **Pydantic v2** — response validation and type safety

## Run Locally

```bash
git clone <your-repo-url>
cd labguard
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key
python3 -m streamlit run app.py
```

## Built By

Larry — life sciences researcher building AI tools for the bench.

Powered by Claude API.

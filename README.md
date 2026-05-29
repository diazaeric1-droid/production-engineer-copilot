# Production Engineer Copilot

> An open-source AI agent that performs a complete well review — decline analysis, artificial lift diagnostics, intervention recommendations, and economics — in 60 seconds.

Built by a Staff Production Engineer (ex-OXY, ex-Shell) who spent 9 years doing this work by hand.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://pe-copilot.streamlit.app)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![Eval](https://img.shields.io/badge/eval-0.90%20agreement-blue)](evals/sample_review.md)

**Try it now → [pe-copilot.streamlit.app](https://pe-copilot.streamlit.app)**

---

## What it does

Drop in a well file (decline curve data, dyno cards or ESP readings, last 30 days of SCADA, completion details). The agent returns:

1. **Current state diagnosis** — is the well performing to type curve? Where is it deviating and why?
2. **Ranked intervention candidates** — acid stim, ESP swap, ESP-to-beam conversion, P&A, etc.
3. **Quick economics** — incremental BOE, treatment cost, NPV, payout time
4. **Interactive chat** — "why acid over a workover?" "what's the second-best option?"

Output is a one-page review that mirrors the format you'd hand to a VP Production on a Monday morning.

**See real samples:**
- [`evals/sample_review.md`](evals/sample_review.md) — flagship demo: Delaware Basin ESP with compound scale + gas + below-POR signature; agent surfaces a sequenced 3-step workover
- [`evals/samples/case_005_gas_interference.md`](evals/samples/case_005_gas_interference.md) — gas-interference well: agent defers ESP swap, recommends low-cost VSD + separator first
- [`evals/samples/case_014_healthy_beam_pump.md`](evals/samples/case_014_healthy_beam_pump.md) — healthy beam pump: agent correctly recommends "continue surveillance" rather than inventing interventions
- [`evals/samples/case_020_p_and_a.md`](evals/samples/case_020_p_and_a.md) — sub-economic stripper: agent recommends P&A with explicit negative-NPV evidence
- [Index of all samples](evals/samples/README.md)

## Why this exists

Production engineers spend 30–50% of their time on routine well reviews. The work is pattern-heavy, data-driven, and follows a repeatable framework — which makes it a textbook case for agentic AI. This repo is a reference implementation of that pattern, designed to be forked and adapted to your operator's data sources.

## Quick start

```bash
git clone https://github.com/diazaeric1-droid/production-engineer-copilot
cd production-engineer-copilot
pip install -e .
cp .env.example .env  # add your ANTHROPIC_API_KEY
python -m src.agent --well data/synthetic/well_001.json
```

For the interactive demo:
```bash
streamlit run demo/app.py
```

## Architecture

```
┌──────────────┐      ┌─────────────────┐      ┌──────────────────┐
│  Well File   │ ───> │  Agent Loop     │ ───> │  Review (MD)     │
│  (JSON/CSV)  │      │  (Claude + tools)│      │  + interactive   │
└──────────────┘      └─────────────────┘      └──────────────────┘
                              │
                ┌─────────────┼──────────────┐
                ▼             ▼              ▼
         decline_curve   esp_diagnostics   economics
         (Arps fit)      (POR check)       (NPV/IRR)
```

The agent has access to deterministic analyzers (no hallucinated math) and uses Claude as the reasoning layer to decide which to call and how to synthesize results.

## Evaluation

20 synthetic well cases spanning 8 intervention types (acid stim, scale treatment, ESP swap, ESP-to-beam conversion, gas separator install, gas lift optimization, paraffin treatment, P&A, plus healthy-well "continue surveillance" controls). Each case has an expert-baseline expected primary recommendation and diagnosis keywords. Runner saves every report to `evals/results/` and writes a `summary.json`.

```bash
python -m evals.run_evals               # full set
python -m evals.run_evals --limit 3     # quick check
python -m evals.run_evals --case case_005   # single case
```

**Current (v0.1):**
- Primary recommendation agreement: **0.90** (18 / 20)
- Diagnosis keyword hit rate: **0.90**
- Two outstanding misses both involve ambiguous interventions where the agent's diagnosis was correct but the recommendation phrasing diverged from the eval's expected term — v0.2 will add dedicated tools for dyno-card interpretation and ESP-economic-life calculation to close the gap.

## Roadmap

- [x] v0.1 — Decline curve + ESP diagnostics + economics + intervention selection heuristics + 20-case eval set @ 0.90 agreement
- [x] v0.1 — Streamlit interactive demo
- [ ] v0.2 — Dedicated dyno-card interpretation tool (closes case_014 pump-off gap) + ESP-economic-life evaluator (closes case_012 ESP-to-beam gap)
- [ ] v0.3 — Multi-well portfolio mode (rank a field of wells by intervention NPV)
- [ ] v0.4 — Connect to common SCADA/historian APIs (PI, Ignition)
- [ ] v0.5 — Chain into AFE Copilot — well review → intervention selection → draft AFE in one workflow

## License

MIT. Built for the community; use it however helps you.

## Contact

Eric Diaz II — [LinkedIn](https://www.linkedin.com/in/eric-a-diaz2) — diaz.a.eric1@gmail.com

Available for senior AI engineering roles and selective consulting engagements with E&P operators.

# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-06-03

Eval-credibility + domain-depth overhaul. The headline number is now earned by
reasoning, not by keyword-matching a leaked answer.

### Closed the two known eval misses at the root
- New **dyno-card interpretation tool** (`interpret_dyno_card`) classifies the latest
  dynamometer card into fluid-pound/pump-off, parted rods (flat card), gas interference,
  or healthy full fillage — the primary downhole diagnostic for beam pumps, which the
  decline curve is blind to. Closes the pump-off miss (the signal existed in the data but
  no tool read it, and `well.summary()` now advertises which data is available).
- New **ESP economic-life evaluator** (`evaluate_esp_economic_life`) decides ESP-swap vs
  ESP-to-beam conversion on *lifecycle* economics (remaining EUR net of the ESP re-fail
  cadence vs a beam unit's longer run life). The ESP-to-beam wells are now genuinely old,
  depleted, and below POR, so the conversion is reachable by reasoning instead of by a
  contradicted age heuristic. Closes the esp-to-beam miss.

### Made a larger sample actually mean something
- **De-leaked the generator:** well-file `notes` now carry only raw field observations +
  distractors — never the diagnosis or recommendation. Expert labels live only in
  `cases.yaml` (which the agent never sees). Enforced by a unit-test invariant.
- **Parameterized every archetype** over bounded reservoir/lift distributions (qi, Di, b,
  intake pressure, amps, water cut, GOR, fillage, age) so each case is a distinct realistic
  point, not a hand-tuned singleton.
- **Boundary / ambiguous cases:** scale-with-gas-distractor, sequenced acid-then-swap, and
  insufficient-data (correct answer = "get more data", not a fabricated call).
- Scaled the dev set to **41 cases**; added a **blind 18-case holdout** (`--holdout`,
  separate seed + id range) — tune on dev, report the holdout number as the headline.

### Upgraded scoring beyond keyword-match
- **LLM-as-judge** rubric (`evals/judge.py`, `run_evals --judge`) scores each report 1-5 on
  diagnosis / recommendation / economics / restraint.
- **Per-class agreement table + confusion matrix** (expected → predicted) so systematic
  confusions are visible instead of hidden in a single blended number.
- **Blind human-grading sheet** generator (`evals/make_human_grading_sheet.py`) for an
  inter-rater PE panel — the most defensible metric for the flagship narrative.

### Domain depth
- `analyze_water_gas_trends`: water-cut and GOR levels + least-squares trends (drives
  interventions the oil-rate curve hides). Healthy-well water-cut flag tuned to 8%/yr so
  maturing wells stay clean.
- VP-grade risked economics in `evaluate_intervention`: chance-of-success, deferred
  production during the job, and SWD/water-disposal drag on net margin (all default to a
  no-op so the point estimate stays comparable to the Monte-Carlo path).
- ESP physics: optional `frequency_hz` / `discharge_pressure_psi` readings and a thrust
  (down/up/neutral) call; a low-Hz-at-below-POR flag confirms the pump is already turned
  down (strengthens a swap/conversion over "just slow it down").

### Correctness fixes surfaced during the live re-run
- **Type-curve analyzer was noise-unstable** — the fixed early-window Arps fit made HEALTHY
  wells read anywhere from −38% to +68% "off type curve," which drove the agent to recommend
  acid stim on a perfectly healthy well. Replaced with **leave-one-out degraded-tail trimming**:
  fit the full history and peel back only points that have genuinely departed from the
  established decline (tested against a fit that EXCLUDES them, so the fit can't bend to absorb a
  break). Healthy wells now read ~0%; real roll-overs still expose deferred production. Regression
  test added. This is a production-credibility fix, not just an eval fix.
- **Empty `ANTHROPIC_API_KEY` shadowing** — `load_dotenv()` would not override a blank env var
  exported by the shell, yielding a cryptic SDK auth error. `agent.py` / `judge.py` now fall back
  to `.env` and raise a clear message if the key is truly missing.
- **Insufficient-data escalation** — the agent defaulted a data-poor well to "continue
  monitoring" (which implies a confirmed-healthy well). Prompt hardened so it must state
  "insufficient data to make a recommendation" and list the data needed, rather than a false
  all-clear.

### Results (v0.4)
- Dev (41 cases): **41/41 (1.00)** recommendation agreement, 0.87 diagnosis-keyword.
- Blind holdout (18 cases): **18/18 (1.00)** recommendation agreement, 0.92 diagnosis-keyword.
- Per-class agreement 100% across all twelve recommendation classes on both sets. Up from the
  prior 0.90 on 20 hand-tuned wells whose notes leaked the answer. CI gate (≥0.85) passes.
- Honest caveat: synthetic wells with clean, separable signatures; real-data + the inter-rater
  PE panel are the next credibility step, not a higher synthetic number.

## [0.3.3] — 2026-06-02

- Self-heal stale Streamlit bytecode cache at startup: purge `src/` `__pycache__`
  and evict cached `src` modules so newly-added functions reload from current source
  after a redeploy. Fixes the startup ImportError cascade seen after adding new
  symbols to existing modules (the app no longer needs a manual Reboot to pick them up).

## [0.3.2] — 2026-06-02

- Resilience: `analyze_type_curve` (added to the pre-existing decline module) is now
  imported defensively, so a stale Streamlit bytecode cache serving an old
  `decline_curve.pyc` degrades gracefully to the plain fit instead of taking the app
  down. Clear the cache fully with a Reboot (Manage app → Reboot).

## [0.3.1] — 2026-06-02

- Hotfix: decline-curve module crashed on import under numpy ≥ 2.3 (where `np.trapz`
  was removed), taking the whole app down. Resolve `np.trapezoid` without eagerly
  touching the removed `np.trapz` alias.

## [0.3.0] — 2026-06-02

- True type-curve benchmark (early-window fit + cumulative deferred bbl/$)
- Monte-Carlo intervention economics (P10/P50/P90 + tornado sensitivity)
- Eval dashboard (20-case agreement, confusion breakdown, $/review) + CI regression gate
- Structured, validated diagnosis export for AFE-Copilot chaining
- Fixed payout off-by-one; replaced mislabeled "rate of return" with discounted profitability index

## [0.2.0]

- Initial public demo
</content>
</invoke>

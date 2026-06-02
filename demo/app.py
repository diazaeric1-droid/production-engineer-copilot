"""Streamlit demo for Production Engineer Copilot.

Layout favours visual storytelling over text — KPI cards, decline plot, ESP
diagnostic multi-panel, and intervention economics chart all render immediately
from deterministic analyzers. The AI agent's narrative review is in its own tab.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src.*` imports work on Streamlit Cloud
# (where the package isn't pip-installed, just the deps from requirements.txt).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import __version__ as APP_VERSION
from src.agent import run_review
from src.analyzers.decline_curve import fit_decline, analyze_type_curve
from src.analyzers.economics import evaluate_intervention, simulate_intervention
from src.analyzers.esp_diagnostics import evaluate_esp
from src.data_loader import WellFile
from src.tools import AFE_INTERVENTIONS, export_afe_diagnosis


st.set_page_config(
    page_title="Production Engineer Copilot",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- styling ---------------------------------------------------------

st.markdown("""
<style>
    /* Don't touch Streamlit's sticky header — that breaks scroll behavior.
       Just trim the gap between Streamlit's toolbar and our content. */
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
    [data-testid="stMetricValue"] {font-size: 1.3rem; line-height: 1.2;}
    [data-testid="stMetricLabel"] {font-size: 0.75rem; font-weight: 600; opacity: 0.8;}
    [data-testid="stMetricDelta"] {font-size: 0.75rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {padding: 0.4rem 1.1rem; font-weight: 600;}
    hr {margin: 0.4rem 0 !important;}
    div.flag-high {background: #4a1010; color: #ffb3b3; padding: 0.3rem 0.7rem;
                   border-radius: 6px; display: inline-block; margin: 0.15rem;
                   font-size: 0.8rem; font-weight: 600;}
    div.flag-ok {background: #103b1a; color: #b3ffc7; padding: 0.3rem 0.7rem;
                 border-radius: 6px; display: inline-block; margin: 0.15rem;
                 font-size: 0.8rem; font-weight: 600;}
    div.app-header {
        display: flex; align-items: center; gap: 1rem;
        padding: 0.2rem 0 0.5rem 0;
    }
    .app-title {font-size: 1.4rem; font-weight: 700; line-height: 1.1;}
    .app-subtitle {font-size: 0.82rem; color: #999;}
    .eval-chip {background:#103b1a; color:#b3ffc7; padding:0.2rem 0.65rem;
                border-radius:10px; font-size:0.75rem; font-weight:600;
                margin-left: auto; white-space: nowrap;}
    .ver-chip {background:#10233b; color:#b3d4ff; padding:0.2rem 0.65rem;
               border-radius:10px; font-size:0.75rem; font-weight:600;
               margin-left: 0.5rem; white-space: nowrap;}
</style>
""", unsafe_allow_html=True)

# ---------- sidebar ----------------------------------------------------------

DATA_DIR = REPO_ROOT / "data" / "synthetic"

with st.sidebar:
    st.subheader("Well selection")
    well_files = sorted(DATA_DIR.glob("well_*.json"))
    if not well_files:
        st.error("No well files found in data/synthetic/")
        st.stop()
    chosen = st.selectbox(
        "Pick a well",
        well_files,
        format_func=lambda p: p.stem.replace("_", " ").title(),
    )
    show_tools = st.checkbox("Show agent tool calls in review", value=True)
    run = st.button("Run AI well review", type="primary", use_container_width=True)
    st.caption("Review takes ~30 sec and costs ~$0.05 in API.")

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "Claude reasons and writes; **deterministic Python tools** "
        "do the engineering math (Arps decline fit, ESP POR check, NPV/IRR). "
        "Engineering numbers stay trusted; LLM stays in its lane."
    )

# ---------- compute deterministic analytics once -----------------------------

well = WellFile.from_json(chosen)
hist = pd.DataFrame(well.production_history)
fit = fit_decline(hist["day"].values, hist["oil_bopd"].values, model="hyperbolic")
# True type curve: fit early/established decline and extrapolate (not dragged down
# by the degraded tail like the full-history fit is).
try:
    tc = analyze_type_curve(hist["day"].values, hist["oil_bopd"].values, model="hyperbolic")
except ValueError:
    tc = None

latest_oil = float(hist["oil_bopd"].iloc[-1])
latest_water = float(hist["water_bwpd"].iloc[-1])
latest_gas = float(hist["gas_mcfd"].iloc[-1])
days_on = int(hist["day"].iloc[-1])

esp_diag = None
if well.artificial_lift.get("type") == "ESP" and well.esp_readings:
    try:
        esp_diag = evaluate_esp(well.esp_readings, well.artificial_lift["pump_spec"])
    except Exception:
        esp_diag = None

# ---------- compact header (title + well meta + eval chip on one row) -------

st.markdown(
    f"<div class='app-header'>"
    f"<div>"
    f"<div class='app-title'>⛽ Production Engineer Copilot</div>"
    f"<div class='app-subtitle'>"
    f"{well.well_id} · {well.api_number} · {well.field} · "
    f"{well.completion.get('formation', '—')} · {well.artificial_lift.get('type', '—')} lift"
    f" — <a href='https://github.com/diazaeric1-droid/production-engineer-copilot' "
    f"style='color:#5a9fd4;'>GitHub</a>"
    f"</div>"
    f"</div>"
    f"<div class='eval-chip'>● 0.90 eval agreement</div>"
    f"<div class='ver-chip'>v{APP_VERSION}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

with st.expander(f"🆕 What's new in v{APP_VERSION}"):
    st.markdown(
        "- **True type-curve benchmark** — early-window fit + cumulative deferred bbl/$\n"
        "- **Monte-Carlo intervention economics** — P10/P50/P90 + tornado sensitivity\n"
        "- **Eval dashboard** — 20-case agreement, confusion breakdown, $/review + CI regression gate\n"
        "- **Structured, validated diagnosis export** for AFE-Copilot chaining\n"
        "- **Fixed payout off-by-one**; replaced mislabeled \"rate of return\" with discounted profitability index"
    )

# ---------- KPI metrics row --------------------------------------------------

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    tc_ref = tc.type_curve_at_last if tc else fit.last_predicted
    delta = latest_oil - tc_ref
    st.metric(
        "Oil rate (BOPD)",
        f"{latest_oil:,.0f}",
        delta=f"{delta:+,.0f} vs type curve",
        delta_color="normal",
    )
with k2:
    st.metric("Days on production", f"{days_on:,}")
with k3:
    wc = latest_water / (latest_water + latest_oil) * 100 if (latest_water + latest_oil) > 0 else 0
    st.metric("Water cut", f"{wc:.0f}%")
with k4:
    glr = latest_gas * 1000 / latest_oil if latest_oil > 0 else 0
    st.metric("GLR (scf/bbl)", f"{glr:,.0f}")
with k5:
    if esp_diag:
        st.metric(
            "ESP intake (psi)",
            f"{esp_diag.intake_pressure_psi:.0f}",
            delta="IN POR" if esp_diag.in_por else "OUT OF POR",
            delta_color="off" if esp_diag.in_por else "inverse",
        )
    else:
        st.metric("Lift type", well.artificial_lift.get("type", "—"))

# ---------- tabs -------------------------------------------------------------

tab_trends, tab_econ, tab_review, tab_evals, tab_raw = st.tabs([
    "📈 Production Trends",
    "💰 Economics (Monte-Carlo)",
    "🤖 AI Review",
    "🧪 Evals",
    "📋 Raw Data",
])

# ---- Tab 1: Production trends ---

with tab_trends:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("Production decline vs. hyperbolic type curve")

        # Type-curve line: use the early-window type curve when available (it reflects
        # what the well *should* be doing); fall back to the full fit otherwise.
        days_dense = np.linspace(hist["day"].min(), hist["day"].max(), 100)
        curve_qi, curve_di, curve_b = (
            (tc.qi, tc.di, tc.b) if tc else (fit.qi, fit.di, fit.b)
        )
        fit_curve = curve_qi / np.power(1 + curve_b * curve_di * days_dense, 1 / max(curve_b, 1e-6))
        tc_label = (
            f"Type curve (b={curve_b:.2f}, fit on first {tc.established_days} pts)"
            if tc else f"Fit (b={fit.b:.2f}, R²={fit.r_squared:.3f})"
        )
        today_below = (tc.deviation_pct < -10) if tc else (fit.fit_residual_pct < -10)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["day"], y=hist["oil_bopd"],
            mode="markers+lines", name="Actual oil rate",
            marker=dict(size=10, color="#1f77b4"),
            line=dict(color="#1f77b4", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=days_dense, y=fit_curve,
            mode="lines", name=tc_label,
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        ))
        # Highlight the last actual point
        fig.add_trace(go.Scatter(
            x=[hist["day"].iloc[-1]], y=[latest_oil],
            mode="markers", name="Today",
            marker=dict(size=18, color="red" if today_below else "green",
                       symbol="circle-open", line=dict(width=3)),
            showlegend=False,
        ))
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Days on production",
            yaxis_title="Oil rate (BOPD)",
            legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="left", x=0),
            template="plotly_dark",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Fit summary")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Initial rate (qᵢ)", f"{fit.qi:,.0f} BOPD")
            st.metric("Decline (Dᵢ)", f"{fit.di:.4f}/day")
        with c2:
            st.metric("Hyperbolic b", f"{fit.b:.2f}")
            st.metric("R²", f"{fit.r_squared:.3f}")

        st.markdown("##### Performance vs. type curve")
        if tc is None:
            st.markdown(
                "<div style='color:#aaa; font-size:0.85rem;'>Not enough history for a "
                "type-curve benchmark.</div>", unsafe_allow_html=True,
            )
        else:
            deviation = tc.deviation_pct
            deferred_note = (
                f"<div style='margin-top:0.4rem; color:#aaa; font-size:0.85rem;'>"
                f"Actual {tc.last_actual:.0f} BOPD vs type curve {tc.type_curve_at_last:.0f} BOPD · "
                f"deferred ≈ {tc.deferred_bbl/1000:,.1f} MBO (${tc.deferred_value_usd/1e6:,.1f}MM) "
                f"vs the early-time type curve</div>"
            )
            if deviation < -10:
                st.markdown(
                    f"<div class='flag-high'>⚠ Underperforming type curve by {abs(deviation):.0f}%</div>"
                    f"{deferred_note}", unsafe_allow_html=True,
                )
            elif deviation > 10:
                st.markdown(
                    f"<div class='flag-ok'>✓ Outperforming type curve by {deviation:.0f}%</div>"
                    f"{deferred_note}", unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='flag-ok'>✓ On type curve ({deviation:+.1f}%)</div>"
                    f"{deferred_note}", unsafe_allow_html=True,
                )

    # ESP diagnostic multi-panel
    if esp_diag and well.esp_readings:
        st.divider()
        st.subheader("ESP diagnostic signals (last 5 days)")

        readings = pd.DataFrame(well.esp_readings)
        readings["date"] = pd.to_datetime(readings["date"])

        fig_esp = make_subplots(
            rows=2, cols=2, subplot_titles=(
                "BFPD vs. POR window",
                "Intake pressure (psi)",
                "Motor temp (°F)",
                "Motor amps (A)",
            ),
            vertical_spacing=0.18, horizontal_spacing=0.10,
        )
        # BFPD with POR shaded band
        fig_esp.add_trace(go.Scatter(
            x=readings["date"], y=readings["bfpd"], mode="lines+markers",
            line=dict(color="#1f77b4", width=2), marker=dict(size=8),
            showlegend=False,
        ), row=1, col=1)
        fig_esp.add_hrect(
            y0=esp_diag.por_min_bfpd, y1=esp_diag.por_max_bfpd,
            fillcolor="green", opacity=0.15, line_width=0, row=1, col=1,
        )
        # Intake
        intake_color = "red" if readings["intake_pressure_psi"].iloc[-1] < 50 else "#1f77b4"
        fig_esp.add_trace(go.Scatter(
            x=readings["date"], y=readings["intake_pressure_psi"],
            mode="lines+markers", line=dict(color=intake_color, width=2),
            marker=dict(size=8), showlegend=False,
        ), row=1, col=2)
        fig_esp.add_hline(y=50, line_dash="dash", line_color="orange", row=1, col=2)
        # Motor temp
        temp_color = "red" if readings["motor_temp_f"].iloc[-1] > 320 else "#1f77b4"
        fig_esp.add_trace(go.Scatter(
            x=readings["date"], y=readings["motor_temp_f"],
            mode="lines+markers", line=dict(color=temp_color, width=2),
            marker=dict(size=8), showlegend=False,
        ), row=2, col=1)
        # Motor amps with nameplate reference
        nameplate = well.artificial_lift["pump_spec"].get("motor_amps_nameplate", 0)
        amp_color = "red" if readings["motor_amps"].iloc[-1] > nameplate * 1.15 else "#1f77b4"
        fig_esp.add_trace(go.Scatter(
            x=readings["date"], y=readings["motor_amps"],
            mode="lines+markers", line=dict(color=amp_color, width=2),
            marker=dict(size=8), showlegend=False,
        ), row=2, col=2)
        if nameplate:
            fig_esp.add_hline(y=nameplate, line_dash="dash", line_color="orange",
                              annotation_text="Nameplate", row=2, col=2)
        fig_esp.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                              template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_esp, use_container_width=True)

        # Flag badges
        if esp_diag.flags:
            flag_html = " ".join(f"<div class='flag-high'>⚠ {f}</div>" for f in esp_diag.flags)
            st.markdown(f"**Active ESP flags:** {flag_html}", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='flag-ok'>✓ ESP operating within all thresholds</div>",
                unsafe_allow_html=True,
            )

# ---- Economics: Monte-Carlo intervention economics ---

with tab_econ:
    st.subheader("Monte-Carlo intervention economics")
    st.caption(
        "Runs ~10,000 trials over uncertain inputs — incremental rate (lognormal ±30%), "
        "uplift decline (±0.15 abs), realized price (sd ~$12) — through the same NPV math "
        "the agent's deterministic tool uses. P10 = optimistic, P90 = conservative."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        mc_name = st.text_input("Intervention", value="Acid stimulation")
        mc_cost = st.number_input("Treatment cost ($)", value=150_000, step=10_000, min_value=1_000)
    with c2:
        mc_rate = st.number_input("Incremental rate (BOPD)", value=120.0, step=10.0, min_value=0.0)
        mc_decline = st.number_input("Uplift decline (/yr)", value=0.6, step=0.05, min_value=0.0)
    with c3:
        mc_price = st.number_input("Realized price ($/bbl)", value=65.0, step=1.0, min_value=1.0)
        mc_trials = st.select_slider("Trials", options=[1_000, 5_000, 10_000, 20_000], value=10_000)

    sim = simulate_intervention(
        name=mc_name,
        treatment_cost_usd=float(mc_cost),
        incremental_rate_bopd=float(mc_rate),
        uplift_decline_per_yr=float(mc_decline),
        realized_price_per_bbl=float(mc_price),
        n_trials=int(mc_trials),
        seed=42,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P90 NPV (conservative)", f"${sim['npv_p90_usd']/1e6:,.2f}MM")
    m2.metric("P50 NPV (median)", f"${sim['npv_p50_usd']/1e6:,.2f}MM")
    m3.metric("P10 NPV (optimistic)", f"${sim['npv_p10_usd']/1e6:,.2f}MM")
    m4.metric(
        "P(payout)",
        f"{sim['probability_of_payout']*100:.0f}%",
        help=f"Fraction of trials with NPV>0 AND payout < {sim['payout_cutoff_months']:.0f} months",
    )

    # Tornado chart (one-at-a-time low/high NPV swing per variable, sorted by swing).
    tdata = sim["tornado"]
    base_npv = sim["npv_p50_usd"]
    order = sorted(tdata, key=lambda k: tdata[k]["swing"])
    labels = {
        "incremental_rate_bopd": "Incremental rate",
        "uplift_decline_per_yr": "Uplift decline",
        "realized_price_per_bbl": "Realized price",
    }
    fig_t = go.Figure()
    for var in order:
        d = tdata[var]
        lo, hi = d["low_npv"], d["high_npv"]
        left, right = min(lo, hi), max(lo, hi)
        fig_t.add_trace(go.Bar(
            y=[labels.get(var, var)],
            x=[right - left],
            base=[left],
            orientation="h",
            marker=dict(color="#5a9fd4"),
            showlegend=False,
            hovertemplate=(
                f"{labels.get(var, var)}<br>"
                f"low NPV: ${left/1e6:,.2f}MM<br>high NPV: ${right/1e6:,.2f}MM<br>"
                f"swing: ${d['swing']/1e6:,.2f}MM<extra></extra>"
            ),
        ))
    fig_t.add_vline(x=base_npv, line_dash="dash", line_color="orange",
                    annotation_text="P50", annotation_position="top")
    fig_t.update_layout(
        title="Tornado — NPV sensitivity (one-at-a-time)",
        height=300, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="NPV ($)", template="plotly_dark", bargap=0.4,
    )
    st.plotly_chart(fig_t, use_container_width=True)

    verdict = (
        "ROBUST" if sim["npv_p90_usd"] > 0 and sim["probability_of_payout"] > 0.8
        else "MARGINAL" if sim["npv_p50_usd"] > 0
        else "HIGH RISK"
    )
    chip = "flag-ok" if verdict == "ROBUST" else "flag-high"
    st.markdown(
        f"<div class='{chip}'>Risk verdict: {verdict}</div> "
        f"<span style='color:#aaa; font-size:0.85rem;'>"
        f"mean NPV ${sim['npv_mean_usd']/1e6:,.2f}MM over {sim['n_trials']:,} trials</span>",
        unsafe_allow_html=True,
    )

    # ---- AFE-Copilot chaining export -----------------------------------------
    st.divider()
    st.markdown("##### ⬇ Export AFE diagnosis (for AFE-Copilot)")
    st.caption(
        "Emits a validated JSON object matching AFE-Copilot's AFEDiagnosis schema — "
        "the pe→afe chain. Pick the canonical intervention; identity fields come from the well."
    )
    af1, af2 = st.columns(2)
    with af1:
        afe_interv = st.selectbox("Intervention (AFE key)", AFE_INTERVENTIONS,
                                  index=AFE_INTERVENTIONS.index("acid_stimulation"))
    with af2:
        afe_diag = st.text_input(
            "Primary diagnosis",
            value="Below type curve; mechanical degradation indicated",
        )
    try:
        afe_obj = export_afe_diagnosis(well, {
            "intervention": afe_interv,
            "primary_diagnosis": afe_diag,
            "incremental_rate_bopd": float(mc_rate),
            "expected_uplift_decline_per_yr": float(mc_decline),
        })
        st.download_button(
            "⬇ Export AFE diagnosis (for AFE-Copilot)",
            data=json.dumps(afe_obj, indent=2),
            file_name=f"{well.well_id}-afe-diagnosis.json",
            mime="application/json",
        )
        with st.expander("Preview AFE diagnosis JSON"):
            st.json(afe_obj)
    except ValueError as e:
        st.warning(f"Cannot build AFE diagnosis: {e}")


# ---- Tab 2: AI review ---

with tab_review:
    if run:
        with st.spinner("Agent reasoning + tool calls…"):
            report = run_review(str(chosen), verbose=show_tools)
        st.markdown(report)
        st.download_button(
            "⬇ Download review (Markdown)",
            report,
            file_name=f"{well.well_id}-review.md",
        )
    else:
        st.info("👈 Click **Run AI well review** in the sidebar to generate the agent's full diagnosis "
                "and ranked intervention recommendations. The charts to the left already show what the "
                "agent's deterministic tools have computed.")

# ---- Evals dashboard (static files, no API) ---

with tab_evals:
    st.subheader("Eval dashboard — 20-case benchmark")
    st.caption(
        "Reads the committed eval artifacts (evals/results/summary.json + case_*.md). "
        "No API calls — this is the checked-in baseline the CI regression gate guards."
    )

    EVAL_RESULTS = REPO_ROOT / "evals" / "results"
    summary_path = EVAL_RESULTS / "summary.json"

    if not summary_path.exists():
        st.info(
            "No eval summary found at `evals/results/summary.json`. "
            "Run `python -m evals.run_evals` to generate it."
        )
    else:
        try:
            rows = json.loads(summary_path.read_text())
        except Exception as e:
            rows = None
            st.warning(f"Could not parse summary.json: {e}")

        if rows:
            scored = [r for r in rows if "recommendation_match" in r]
            n = len(scored) if scored else len(rows)
            rec_hits = sum(1 for r in scored if r.get("recommendation_match"))
            agreement = rec_hits / n if n else 0.0
            kw_vals = [r["keyword_hit_rate"] for r in rows if "keyword_hit_rate" in r]
            kw_rate = sum(kw_vals) / len(kw_vals) if kw_vals else 0.0
            errors = [r for r in rows if "error" in r]

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Recommendation agreement", f"{agreement*100:.0f}%", f"{rec_hits}/{n}")
            e2.metric("Keyword hit rate", f"{kw_rate*100:.0f}%")
            e3.metric("Cases", f"{len(rows)}")
            # Estimated API cost per review (~$0.05) x number of cases — cost to run the set.
            e4.metric("Est. $/eval run", f"${0.05*len(rows):,.2f}", help="~$0.05/review × cases")

            if errors:
                st.warning(f"{len(errors)} case(s) errored during the last run.")

            # Per-case pass/fail table
            st.markdown("##### Per-case results")
            table_rows = []
            for r in rows:
                table_rows.append({
                    "case": r.get("id", "—"),
                    "notes": r.get("notes", ""),
                    "expected": r.get("expected", "—"),
                    "keyword_hit": (
                        f"{r['keyword_hit_rate']*100:.0f}%" if "keyword_hit_rate" in r else "—"
                    ),
                    "recommendation": (
                        "✅ pass" if r.get("recommendation_match")
                        else "❌ miss" if "recommendation_match" in r
                        else ("⚠ error" if "error" in r else "—")
                    ),
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            # Confusion-style breakdown: expected recommendation -> pass / miss counts.
            st.markdown("##### Recommendation breakdown (expected → pass / miss)")
            conf = {}
            for r in scored:
                exp = r.get("expected", "—")
                bucket = conf.setdefault(exp, {"pass": 0, "miss": 0})
                bucket["pass" if r.get("recommendation_match") else "miss"] += 1
            if conf:
                conf_df = pd.DataFrame([
                    {"expected": k, "pass": v["pass"], "miss": v["miss"],
                     "n": v["pass"] + v["miss"]}
                    for k, v in sorted(conf.items())
                ])
                st.dataframe(conf_df, use_container_width=True, hide_index=True)
                misses = conf_df[conf_df["miss"] > 0]
                if not misses.empty:
                    st.caption(
                        "Outstanding misses concentrate in: "
                        + ", ".join(misses["expected"].tolist())
                    )
            else:
                st.caption("No recommendation_match field in summary rows — breakdown unavailable.")

            # Drill into a single case report (case_*.md)
            st.markdown("##### Case report")
            case_ids = [r.get("id") for r in rows if r.get("id")]
            if case_ids:
                pick = st.selectbox("View a case report", case_ids)
                md_path = EVAL_RESULTS / f"{pick}.md"
                if md_path.exists():
                    st.markdown(md_path.read_text())
                else:
                    st.info(f"No saved report for {pick} (expected `{md_path.name}`).")


# ---- Tab 3: Raw data ---

with tab_raw:
    st.subheader("Raw well file (JSON)")
    with open(chosen) as f:
        st.json(json.load(f))

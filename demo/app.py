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

from src.agent import run_review
from src.analyzers.decline_curve import fit_decline
from src.analyzers.esp_diagnostics import evaluate_esp
from src.data_loader import WellFile


st.set_page_config(
    page_title="Production Engineer Copilot",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- styling ---------------------------------------------------------

st.markdown("""
<style>
    .block-container {padding-top: 0.6rem; padding-bottom: 2rem; max-width: 1400px;}
    header[data-testid="stHeader"] {height: 2rem;}
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
        padding: 0.2rem 0 0.6rem 0;
    }
    .app-title {font-size: 1.5rem; font-weight: 700; line-height: 1;}
    .app-subtitle {font-size: 0.85rem; color: #999;}
    .eval-chip {background:#103b1a; color:#b3ffc7; padding:0.2rem 0.65rem;
                border-radius:10px; font-size:0.75rem; font-weight:600;
                margin-left: auto; white-space: nowrap;}
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
    f"</div>",
    unsafe_allow_html=True,
)

# ---------- KPI metrics row --------------------------------------------------

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    delta = latest_oil - fit.last_predicted
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

tab_trends, tab_review, tab_raw = st.tabs([
    "📈 Production Trends",
    "🤖 AI Review",
    "📋 Raw Data",
])

# ---- Tab 1: Production trends ---

with tab_trends:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("Production decline vs. hyperbolic type curve")

        # Build the fit line over the actual day range
        days_dense = np.linspace(hist["day"].min(), hist["day"].max(), 100)
        fit_curve = fit.qi / np.power(1 + fit.b * fit.di * days_dense, 1 / max(fit.b, 1e-6))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["day"], y=hist["oil_bopd"],
            mode="markers+lines", name="Actual oil rate",
            marker=dict(size=10, color="#1f77b4"),
            line=dict(color="#1f77b4", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=days_dense, y=fit_curve,
            mode="lines", name=f"Type curve (b={fit.b:.2f}, R²={fit.r_squared:.3f})",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        ))
        # Highlight the last actual point
        fig.add_trace(go.Scatter(
            x=[hist["day"].iloc[-1]], y=[latest_oil],
            mode="markers", name="Today",
            marker=dict(size=18, color="red" if fit.deviation_pct < -10 else "green",
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
        deviation = fit.deviation_pct
        if deviation < -10:
            st.markdown(
                f"<div class='flag-high'>⚠ Underperforming by {abs(deviation):.0f}%</div>"
                f"<div style='margin-top:0.4rem; color:#aaa; font-size:0.85rem;'>"
                f"Last actual {fit.last_actual:.0f} BOPD vs predicted {fit.last_predicted:.0f} BOPD</div>",
                unsafe_allow_html=True,
            )
        elif deviation > 10:
            st.markdown(
                f"<div class='flag-ok'>✓ Outperforming by {deviation:.0f}%</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='flag-ok'>✓ On type curve ({deviation:+.1f}%)</div>",
                unsafe_allow_html=True,
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

# ---- Tab 3: Raw data ---

with tab_raw:
    st.subheader("Raw well file (JSON)")
    with open(chosen) as f:
        st.json(json.load(f))

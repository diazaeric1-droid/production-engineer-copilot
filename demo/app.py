"""Streamlit demo for Production Engineer Copilot."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.agent import run_review
from src.data_loader import WellFile


st.set_page_config(page_title="Production Engineer Copilot", page_icon="⛽", layout="wide")

st.title("Production Engineer Copilot")
st.caption("AI-driven well reviews for production engineers. Built by an ex-OXY / ex-Shell Staff Production Engineer.")

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

with st.sidebar:
    st.header("Well selection")
    well_files = sorted(DATA_DIR.glob("*.json"))
    if not well_files:
        st.error("No well files found.")
        st.stop()
    chosen = st.selectbox("Pick a well", well_files, format_func=lambda p: p.stem)
    verbose = st.checkbox("Show tool calls", value=True)
    run = st.button("Run review", type="primary")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Well data")
    with chosen.open() as f:
        data = json.load(f)
    well = WellFile.from_json(chosen)
    st.write(well.summary())
    with st.expander("Raw JSON"):
        st.json(data)

with col2:
    st.subheader("AI Review")
    if run:
        with st.spinner("Agent running well review..."):
            report = run_review(str(chosen), verbose=verbose)
        st.markdown(report)
        st.download_button("Download review (Markdown)", report, file_name=f"{well.well_id}-review.md")
    else:
        st.info("Click **Run review** in the sidebar to start.")

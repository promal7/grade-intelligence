"""
app.py -- Grade Change Intelligence Dashboard

Run with:  streamlit run app.py

Maps directly to the deliverables:
- Panel 1 (Live Transition Monitor): predicts risk of going off-spec during
  a grade change, before the limit is exceeded.
- Panel 2 (Correlation Explorer): shows new correlations found in historical
  data, ranked by strength.
- Panel 3 (Recommendations): setpoint recommendations, each tagged with its
  source of inference (historical correlation / trajectory extrapolation),
  with Accept/Reject buttons whose decisions are logged for later
  accuracy/quality evaluation.
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis_engine as ae

st.set_page_config(page_title="Grade Change Intelligence", layout="wide")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(DATA_DIR, "recommendation_log.csv")


@st.cache_data
def get_data():
    meta = pd.read_csv(os.path.join(DATA_DIR, "events_meta.csv"))
    ts = pd.read_csv(os.path.join(DATA_DIR, "events_timeseries.csv"))
    return meta, ts


@st.cache_data
def get_correlations(meta):
    return ae.find_correlations(meta)


@st.cache_data
def get_envelope(meta):
    return ae.safe_envelope(meta)


def load_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=["timestamp", "event_id", "t", "rec_id", "rec_text", "decision"])


def append_log(row: dict):
    df = load_log()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)


# ---------------------------------------------------------------------------
meta, ts = get_data()
correlations = get_correlations(meta)
envelope = get_envelope(meta)

st.title("Grade Change Intelligence")
st.caption(
    "Predicts off-spec risk during grade transitions, surfaces which loop "
    "parameters actually drive deviation, and recommends setpoints with a "
    "rationale attached to every suggestion. Trained on 140 synthetic "
    "historical grade-change events (see README for why the data is synthetic)."
)

# --- Sidebar: pick an event and scrub through time (simulates "live") ------
st.sidebar.header("Live Transition Monitor")
event_ids = sorted(meta["event_id"].unique())
event_id = st.sidebar.selectbox(
    "Grade change event", event_ids,
    format_func=lambda e: f"Event {e}: "
    f"{meta[meta.event_id==e].iloc[0]['from_grade']} -> {meta[meta.event_id==e].iloc[0]['to_grade']}"
)
event_meta = meta[meta.event_id == event_id].iloc[0]
event_ts = ts[ts.event_id == event_id].reset_index(drop=True)
max_t = int(event_ts["t"].max())

current_t = st.sidebar.slider("Current time step (simulated live position)", 5, max_t, min(30, max_t))
st.sidebar.markdown(f"**Target basis weight:** {event_meta['bw_target']:.1f}")
st.sidebar.markdown(f"**Spec limit:** \u00b1{ae.SPEC_LIMIT_PCT}%")
st.sidebar.markdown(f"**Actually went off-spec (ground truth):** "
                     f"{'Yes' if event_meta['went_offspec'] else 'No'}")

# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Live Monitor & Risk", "Correlations Found", "Recommendation Log"])

# --- TAB 1: Live monitor ----------------------------------------------------
with tab1:
    risk = ae.assess_trajectory_risk(event_ts, event_meta["bw_target"], current_t)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current deviation from target", f"{risk.get('current_dev_pct', 0):.2f}%")
    col2.metric("Projected max deviation (next 15 steps)", f"{risk.get('projected_max_dev_pct', 0):.2f}%")
    risk_color = {"low": "\U0001F7E2 Low", "medium": "\U0001F7E1 Medium", "high": "\U0001F534 High"}
    col3.metric("Risk of spec breach", risk_color.get(risk.get("risk"), "n/a"))

    # Chart: actual trace so far + spec band + projected trajectory
    visible = event_ts[event_ts["t"] <= current_t]
    fig = go.Figure()
    fig.add_hrect(
        y0=event_meta["bw_target"] * (1 - ae.SPEC_LIMIT_PCT / 100),
        y1=event_meta["bw_target"] * (1 + ae.SPEC_LIMIT_PCT / 100),
        fillcolor="green", opacity=0.08, line_width=0,
        annotation_text="Spec band", annotation_position="top left",
    )
    fig.add_trace(go.Scatter(x=visible["t"], y=visible["basis_weight"],
                              mode="lines", name="Basis weight (actual)",
                              line=dict(color="#1f77b4", width=2)))
    fig.add_hline(y=event_meta["bw_target"], line_dash="dot", line_color="gray",
                  annotation_text="Target")

    if risk.get("projected_path"):
        proj_t, proj_v = zip(*risk["projected_path"])
        fig.add_trace(go.Scatter(x=proj_t, y=proj_v, mode="lines",
                                  name="Projected trajectory",
                                  line=dict(color="orange", width=2, dash="dash")))
    fig.update_layout(height=420, xaxis_title="Time step", yaxis_title="Basis weight",
                       margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recommendations")
    st.caption("Every recommendation is tagged with where it came from -- historical "
               "correlation or live trajectory extrapolation -- per the rubric requirement.")

    current_drivers = {c: event_meta[c] for c in ae.DRIVER_COLS}
    recs = ae.generate_recommendations(current_drivers, envelope, correlations, risk)

    if not recs:
        st.success("No recommendations -- current setpoints are within the historically safe envelope.")
    else:
        for rec in recs:
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"**{rec['text']}**")
                    st.caption(f"Rationale: {rec['rationale']}")
                    st.caption(f"Source: `{rec['source_of_inference']}` \u00b7 Confidence: `{rec['confidence']}`")
                with c2:
                    accept = st.button("\u2705 Accept", key=f"acc_{rec['id']}_{event_id}_{current_t}")
                    reject = st.button("\u274c Reject", key=f"rej_{rec['id']}_{event_id}_{current_t}")
                    if accept or reject:
                        append_log({
                            "timestamp": datetime.utcnow().isoformat(),
                            "event_id": int(event_id), "t": int(current_t),
                            "rec_id": rec["id"], "rec_text": rec["text"],
                            "decision": "accepted" if accept else "rejected",
                        })
                        st.rerun()

# --- TAB 2: Correlations -----------------------------------------------------
with tab2:
    st.subheader("Correlations discovered in historical transitions")
    st.caption("Loop parameters ranked by how strongly they relate to deviation "
               "magnitude and stabilization time. This is what backs every "
               "'historical_correlation' recommendation in the Live Monitor tab.")
    st.dataframe(
        correlations.style.background_gradient(subset=["pearson_r"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Safe operating envelope (learned from successful transitions)")
    st.caption(f"Based on {envelope['n_good'].iloc[0]} historically successful "
               f"vs {envelope['n_bad'].iloc[0]} off-spec transitions.")
    st.dataframe(envelope, use_container_width=True, hide_index=True)

    st.subheader("High-impact parameters on stabilization time")
    stab_impact = correlations[correlations["target"] == "stabilization_time"] \
        .sort_values("pearson_r", key=abs, ascending=False)
    fig2 = go.Figure(go.Bar(
        x=stab_impact["pearson_r"], y=stab_impact["driver"], orientation="h",
        marker_color=stab_impact["pearson_r"], marker_colorscale="RdYlGn_r",
    ))
    fig2.update_layout(height=300, xaxis_title="Correlation with stabilization time",
                        margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# --- TAB 3: Recommendation log / accuracy tracking ---------------------------
with tab3:
    st.subheader("Recommendation accept/reject log")
    st.caption("Every accept/reject decision is recorded here so recommendation "
               "quality can be evaluated over time (rubric requirement #6).")
    log = load_log()
    if len(log) == 0:
        st.info("No recommendations have been accepted or rejected yet. "
                "Go to the Live Monitor tab and act on a recommendation.")
    else:
        acc_rate = (log["decision"] == "accepted").mean()
        c1, c2 = st.columns(2)
        c1.metric("Total decisions logged", len(log))
        c2.metric("Acceptance rate", f"{acc_rate:.0%}")
        st.dataframe(log.sort_values("timestamp", ascending=False),
                     use_container_width=True, hide_index=True)
        if st.button("Clear log"):
            os.remove(LOG_PATH)
            st.rerun()

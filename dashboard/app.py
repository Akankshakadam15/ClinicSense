"""
ClinicSense – Streamlit Demo Dashboard
Provides an interactive view of:
  - Day's appointment schedule with risk scores
  - Overbooking recommendation
  - Individual patient risk explanation

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD,
    SLOTS_PER_DAY, COST_IDLE_SLOT,
    COST_WAITING_MINUTE, COST_OVERTIME_MINUTE, SLOT_DURATION_MIN,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinicSense",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.risk-high   { background-color:#FFCCCC; padding:4px 10px; border-radius:6px;
               color:#C62828; font-weight:bold; }
.risk-medium { background-color:#FFF3CC; padding:4px 10px; border-radius:6px;
               color:#E65100; font-weight:bold; }
.risk-low    { background-color:#CCFFCC; padding:4px 10px; border-radius:6px;
               color:#2E7D32; font-weight:bold; }
.metric-box  { background:#F5F5F5; border-radius:8px; padding:12px;
               text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/200x60?text=ClinicSense", width=200)
st.sidebar.title("ClinicSense Controls")

n_slots_input = st.sidebar.slider(
    "Base Slots per Day", min_value=10, max_value=30, value=SLOTS_PER_DAY)
n_patients    = st.sidebar.slider(
    "Appointments to Generate", min_value=5, max_value=40, value=25)
random_seed   = st.sidebar.number_input("Random Seed", value=42, step=1)
show_details  = st.sidebar.checkbox("Show patient details table", value=True)
run_sim       = st.sidebar.button("🔄  Generate New Day", type="primary")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏥 ClinicSense – Patient No-Show Risk Dashboard")
st.markdown(
    "**B.Sc. Data Science Honours Project** | Student: Kadam Akanksha Dinkar  \n"
    "Real-time no-show risk scores and dynamic overbooking recommendation for clinic managers."
)
st.divider()

# ── Generate synthetic day ────────────────────────────────────────────────────
@st.cache_data(ttl=0)
def generate_day(n_patients: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "PatientID":     [f"P{1000+i}" for i in range(n_patients)],
        "Name":          [f"Patient {chr(65+i%26)}{i+1}" for i in range(n_patients)],
        "Age":           rng.integers(5, 90, n_patients),
        "Gender":        rng.choice(["M","F"], n_patients),
        "LeadTime":      rng.integers(0, 60, n_patients),
        "PriorNoShows":  rng.integers(0, 5, n_patients),
        "SMSReceived":   rng.integers(0, 2, n_patients),
        "Hypertension":  rng.integers(0, 2, n_patients),
        "Diabetes":      rng.integers(0, 2, n_patients),
        "ApptTime":      [f"{8+i//3:02d}:{(i%3)*20:02d}" for i in range(n_patients)],
    }
    df = pd.DataFrame(data)
    # Compute simplified no-show probability using a rule-based proxy
    base = 0.20
    df["NoShowProb"] = (
        base
        + 0.003 * df["LeadTime"]
        + 0.06  * df["PriorNoShows"]
        - 0.04  * df["SMSReceived"]
        - 0.001 * df["Age"]
        + rng.uniform(-0.05, 0.05, n_patients)
    ).clip(0.02, 0.95)
    df["AttendProb"] = 1 - df["NoShowProb"]
    df["Risk"] = df["NoShowProb"].apply(
        lambda p: "High"   if p >= RISK_HIGH_THRESHOLD
             else "Medium" if p >= RISK_LOW_THRESHOLD
             else "Low"
    )
    return df


if run_sim or "day_df" not in st.session_state:
    st.session_state["day_df"]  = generate_day(n_patients, random_seed)
    st.session_state["n_slots"] = n_slots_input

df = st.session_state["day_df"]
n_slots = st.session_state["n_slots"]

# ── Top metrics ───────────────────────────────────────────────────────────────
expected_shows   = df["AttendProb"].sum()
expected_noshows = df["NoShowProb"].sum()
rec_extra        = max(0, round(expected_noshows))
total_booked     = n_patients
high_risk_count  = (df["Risk"] == "High").sum()
utilisation_est  = min(expected_shows, n_slots) / n_slots * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Appointments", f"{total_booked}")
col2.metric("Expected Shows", f"{expected_shows:.1f}")
col3.metric("Expected No-Shows", f"{expected_noshows:.1f}")
col4.metric("High-Risk Patients", f"{high_risk_count}",
            delta=f"+{high_risk_count} need attention",
            delta_color="inverse")
col5.metric("Est. Utilisation", f"{utilisation_est:.1f}%",
            delta=f"{'✓ Good' if utilisation_est >= 85 else '⚠ Low'}")

st.divider()

# ── Overbooking Recommendation ────────────────────────────────────────────────
st.subheader("📋 Overbooking Recommendation")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.markdown(f"""
    **Base capacity:** {n_slots} slots  
    **Booked patients:** {total_booked}  
    **Expected no-shows:** {expected_noshows:.1f}  
    """)
with c2:
    if rec_extra == 0:
        color = "green"
        msg   = "No extra bookings needed."
    elif rec_extra <= 2:
        color = "orange"
        msg   = f"Accept **{rec_extra}** extra patient(s)."
    else:
        color = "red"
        msg   = f"Accept **{rec_extra}** extra patient(s) ⚠ monitor wait times."
    st.markdown(f"""
    **Recommended extra bookings:** :{color}[**{rec_extra}**]  
    _{msg}_
    """)
with c3:
    # Visual capacity bar
    fig_cap, ax_cap = plt.subplots(figsize=(5, 0.8))
    ax_cap.barh(0, n_slots + rec_extra, height=0.5, color="#E3F2FD", edgecolor="grey")
    ax_cap.barh(0, min(expected_shows, n_slots + rec_extra), height=0.5,
                color="#1976D2", label="Expected shows")
    ax_cap.barh(0, n_slots, height=0.5, left=0, color="none",
                edgecolor="red", linewidth=2, label="Capacity")
    ax_cap.set_xlim(0, n_slots + rec_extra + 2)
    ax_cap.set_yticks([])
    ax_cap.legend(loc="lower right", fontsize=8)
    ax_cap.set_title("Capacity vs Expected Shows", fontsize=9)
    st.pyplot(fig_cap, use_container_width=True)
    plt.close(fig_cap)

st.divider()

# ── Risk distribution chart ────────────────────────────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("📊 Risk Distribution")
    risk_counts = df["Risk"].value_counts().reindex(["Low","Medium","High"], fill_value=0)
    fig_risk, ax_risk = plt.subplots(figsize=(5, 3.5))
    colors = {"Low":"#4CAF50","Medium":"#FFC107","High":"#F44336"}
    ax_risk.bar(risk_counts.index, risk_counts.values,
                color=[colors[r] for r in risk_counts.index], edgecolor="black")
    ax_risk.set_title("Patients by Risk Level")
    ax_risk.set_ylabel("Count")
    for i, (r, v) in enumerate(risk_counts.items()):
        ax_risk.text(i, v+0.1, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_risk, use_container_width=True)
    plt.close(fig_risk)

with col_b:
    st.subheader("⏰ No-Show Probability Distribution")
    fig_prob, ax_prob = plt.subplots(figsize=(5, 3.5))
    ax_prob.hist(df["NoShowProb"], bins=15, color="steelblue", edgecolor="white", alpha=0.8)
    ax_prob.axvline(RISK_LOW_THRESHOLD,  color="orange", linestyle="--",
                    linewidth=1.5, label=f"Medium ({RISK_LOW_THRESHOLD})")
    ax_prob.axvline(RISK_HIGH_THRESHOLD, color="red",    linestyle="--",
                    linewidth=1.5, label=f"High ({RISK_HIGH_THRESHOLD})")
    ax_prob.set_xlabel("Predicted No-Show Probability")
    ax_prob.set_ylabel("Count")
    ax_prob.set_title("Probability Distribution")
    ax_prob.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_prob, use_container_width=True)
    plt.close(fig_prob)

st.divider()

# ── Patient schedule table ────────────────────────────────────────────────────
if show_details:
    st.subheader("🗓️ Today's Appointment Schedule")

    def risk_badge(r):
        if r == "High":   return "🔴 High"
        elif r == "Medium": return "🟡 Medium"
        else:             return "🟢 Low"

    def suggest_action(row):
        if row["Risk"] == "High":   return "📞 Phone + SMS reminder"
        elif row["Risk"] == "Medium": return "📱 SMS reminder"
        else:                        return "✓ No action"

    display_df = df[["PatientID","Name","Age","Gender","ApptTime",
                      "LeadTime","SMSReceived","NoShowProb","Risk"]].copy()
    display_df["NoShowProb"]  = display_df["NoShowProb"].round(3)
    display_df["Risk"]        = display_df["Risk"].apply(risk_badge)
    display_df["Action"]      = df.apply(suggest_action, axis=1)
    display_df = display_df.rename(columns={
        "PatientID": "ID", "LeadTime": "Lead(d)",
        "SMSReceived": "SMS", "NoShowProb": "P(No-Show)"
    })

    st.dataframe(display_df.set_index("ID"), use_container_width=True, height=400)

    st.download_button(
        label="⬇️ Download Schedule CSV",
        data=display_df.to_csv(index=True),
        file_name="todays_schedule.csv",
        mime="text/csv",
    )

st.divider()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
---
**ClinicSense** | B.Sc. Data Science Honours | Kadam Akanksha Dinkar  
*Note: This dashboard uses a simplified probability model for demonstration. 
In production, probabilities come from the tuned, calibrated LightGBM model 
trained in Notebooks 03–04.*
""")

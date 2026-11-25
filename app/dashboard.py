import json
import sys
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.generate_data import main as generate_all_data
from src.train_model import train_and_save
from src.copilot import SLACopilot

DATA_PATH = ROOT_DIR / "data" / "raw_tickets.csv"
MODEL_PATH = ROOT_DIR / "models" / "sla_model.pkl"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"

st.set_page_config(
    page_title="SLA-Guard Copilot | IT Ops Decision Suite",
    page_icon="\U0001F6E1",
    layout="wide",
)


@st.cache_resource
def load_assets():
    if not DATA_PATH.exists():
        generate_all_data()
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        train_and_save()

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    copilot = SLACopilot()
    return df, model, metrics, copilot


df, model, metrics, copilot = load_assets()

SLA_DEFAULTS = {
    "P1 - Critical": 4,
    "P2 - High": 8,
    "P3 - Medium": 24,
    "P4 - Low": 48,
}

st.title("\U0001F6E1 SLA-Guard Copilot")
st.caption(
    f"Enterprise IT Ops Predictive Analytics & Azure OpenAI Decision Suite  |  Active AI Engine: **{copilot.mode_label}**"
)

tab1, tab2, tab3 = st.tabs(
    [
        "\U0001F4CA Ops Command Center",
        "\U0001F6A8 Live Breach Predictor & Copilot",
        "\U0001F4DA SOP Runbook Q&A",
    ]
)

# ---------------- TAB 1: OPS COMMAND CENTER ----------------
with tab1:
    total_tickets = len(df)
    breach_rate = df["sla_breached"].mean() * 100
    compliance_rate = 100 - breach_rate
    avg_res = df["actual_resolution_hours"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Historical Tickets", f"{total_tickets:,}")
    c2.metric(
        "SLA Compliance Rate",
        f"{compliance_rate:.1f}%",
        delta=f"-{breach_rate:.1f}% Breach Rate",
        delta_color="inverse",
    )
    c3.metric("Avg Resolution Time", f"{avg_res:.1f} hrs")
    c4.metric("XGBoost ROC-AUC / F1", f"{metrics['roc_auc']:.2f} / {metrics['f1_score']:.2f}")

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        cat_df = (
            df.groupby("category", as_index=False)["sla_breached"]
            .mean()
            .assign(breach_pct=lambda d: (d["sla_breached"] * 100).round(1))
            .sort_values("breach_pct", ascending=True)
        )
        fig_cat = px.bar(
            cat_df,
            x="breach_pct",
            y="category",
            orientation="h",
            title="SLA Breach Rate (%) by Incident Category",
            labels={"breach_pct": "Breach Rate (%)", "category": "Category"},
            color="breach_pct",
            color_continuous_scale="Reds",
        )
        fig_cat.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_right:
        heat_df = (
            df.groupby(["assigned_group", "priority"], as_index=False)["sla_breached"]
            .mean()
            .pivot(index="assigned_group", columns="priority", values="sla_breached")
            * 100
        ).round(1)
        fig_heat = px.imshow(
            heat_df,
            text_auto=True,
            color_continuous_scale="YlOrRd",
            title="Team vs Priority Breach Risk Heatmap (%)",
            labels={"color": "Breach %"},
        )
        fig_heat.update_layout(height=360)
        st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("View Model Performance & Top Risk Drivers"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        m2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
        m3.metric("Recall (Breach Catch)", f"{metrics['recall'] * 100:.1f}%")
        m4.metric("F1-Score", f"{metrics['f1_score'] * 100:.1f}%")

        feat_df = pd.DataFrame(
            list(metrics["top_features"].items()), columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)
        fig_feat = px.bar(
            feat_df, x="Importance", y="Feature", orientation="h", title="Top XGBoost Risk Drivers"
        )
        fig_feat.update_layout(height=300)
        st.plotly_chart(fig_feat, use_container_width=True)

# ---------------- TAB 2: LIVE PREDICTOR & COPILOT ----------------
with tab2:
    st.subheader("Simulate Incoming IT Incident")
    left_form, right_out = st.columns([1, 1.35], gap="large")

    with left_form:
        priority = st.selectbox("Ticket Priority", list(SLA_DEFAULTS.keys()), index=0)
        category = st.selectbox(
            "Incident Category",
            [
                "Database & SQL",
                "Network & VPN",
                "Cloud Infrastructure",
                "Identity & Access (IAM)",
                "Application Error",
            ],
        )
        assigned_group = st.selectbox(
            "Current Assigned Team",
            ["L1-ServiceDesk", "L2-InfraOps", "L3-DBA", "L2-NetOps", "SecOps"],
        )
        open_queue = st.slider("Open Tickets in Team Queue", min_value=1, max_value=70, value=42)
        reassignments = st.slider("Ticket Reassignment Count", min_value=0, max_value=6, value=2)
        created_hour = st.slider("Ticket Created Hour (24h)", min_value=0, max_value=23, value=22)
        is_weekend = st.checkbox("Created on Weekend?", value=False)
        description = st.text_area(
            "Incident Description",
            value="Production PostgreSQL reporting database experiencing severe lock contention, query timeouts >60s, and 92% connection pool saturation.",
            height=100,
        )

        run_btn = st.button(
            "\U0001F680 Predict SLA Risk & Run AI Copilot",
            type="primary",
            use_container_width=True,
        )

    with right_out:
        if run_btn:
            ticket_payload = {
                "priority": priority,
                "category": category,
                "assigned_group": assigned_group,
                "open_tickets_in_queue": int(open_queue),
                "reassignment_count": int(reassignments),
                "created_hour": int(created_hour),
                "is_weekend": int(is_weekend),
                "sla_target_hours": SLA_DEFAULTS[priority],
                "description": description,
            }

            input_df = pd.DataFrame([{k: v for k, v in ticket_payload.items() if k != "description"}])
            breach_prob = float(model.predict_proba(input_df)[0][1])
            breach_pct = breach_prob * 100

            gauge_color = "#dc2626" if breach_pct >= 70 else ("#f59e0b" if breach_pct >= 40 else "#16a34a")
            risk_label = (
                "CRITICAL BREACH RISK"
                if breach_pct >= 70
                else ("MODERATE WARNING" if breach_pct >= 40 else "LOW RISK (SAFE)")
            )

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=round(breach_pct, 1),
                    number={"suffix": "%"},
                    title={"text": f"{risk_label} (SLA Target: {SLA_DEFAULTS[priority]}h)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, 40], "color": "#dcfce7"},
                            {"range": [40, 70], "color": "#fef3c7"},
                            {"range": [70, 100], "color": "#fee2e2"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(height=250, margin=dict(t=50, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            with st.spinner("Retrieving enterprise SOP runbooks & generating decision plan..."):
                advice = copilot.generate_advice(ticket_payload, breach_prob)

            st.markdown(f"**Root Cause Hypothesis** *(Source: `{advice.cited_runbook}`)*")
            st.info(advice.root_cause_hypothesis)

            st.markdown("**Immediate Technical Action Steps**")
            for idx, step in enumerate(advice.immediate_action_steps, start=1):
                st.markdown(f"{idx}. {step}")

            st.markdown(f"**Recommended Escalation Team:** `{advice.recommended_escalation_team}`")

            with st.expander("Ready-to-Send Manager Escalation Alert", expanded=True):
                st.code(advice.manager_alert_draft, language="text")
        else:
            st.info(
                "Left side par incident details choose karo aur **Predict SLA Risk & Run AI Copilot** button click karo."
            )

# ---------------- TAB 3: RUNBOOK Q&A ----------------
with tab3:
    st.subheader("Enterprise IT Runbook Knowledge Base")
    st.write("Directly query indexed IT SOPs (Database, VPN/Network, Cloud Infra, IAM, App Errors).")

    sample_q = st.selectbox(
        "Quick Sample Questions",
        [
            "Custom question...",
            "What is the escalation policy and first steps for a P1 Database deadlock?",
            "How do we handle a corporate VPN outage with high packet loss?",
            "What should we do if Kubernetes pods are in CrashLoopBackOff?",
        ],
    )
    user_q = st.text_input(
        "Ask SLA-Guard Runbook Copilot:",
        value="" if sample_q == "Custom question..." else sample_q,
        placeholder="e.g., What are the steps for an expired Azure AD client secret?",
    )

    if st.button("Search Runbooks", type="primary") and user_q.strip():
        with st.spinner("Searching runbooks..."):
            res = copilot.ask_runbooks(user_q.strip())
        st.markdown(res["answer"])
        st.caption(f"Cited SOPs: {', '.join(res['sources'])}")

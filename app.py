import json
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from src.alert_loader import load_validated_alerts
from src.config import AUDIT_LOG_PATH, GEMINI_API_KEY
from src.data_validator import AlertRecord, Baseline12M, BehavioralSignals
from src.llm_synthesizer import synthesize_triage_dossier
from src.mock_enricher import EnrichedAlert, enrich_alert

st.set_page_config(
    page_title="AML/KYC Triage & SAR Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_enriched_queue() -> list[EnrichedAlert]:
    alerts = load_validated_alerts()
    return [enrich_alert(alert) for alert in alerts]


def log_audit_decision(
    enriched: EnrichedAlert,
    disposition: str,
    final_narrative: str,
    ai_recommendation: str,
    generation_mode: str,
) -> dict:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "alert_id": enriched.alert.alert_id,
        "subject_id": enriched.alert.subject_id,
        "counterparty_id": enriched.alert.counterparty_id,
        "transaction_amount_inr": enriched.alert.transaction_amount,
        "spike_multiplier": enriched.alert.baseline_12m.spike_multiplier,
        "rule_risk_score": enriched.risk_score,
        "rule_risk_tier": enriched.rule_risk_tier,
        "ai_recommended_action": ai_recommendation,
        "investigator_disposition": disposition,
        "generation_mode": generation_mode,
        "final_narrative": final_narrative,
    }
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


if "decisions" not in st.session_state:
    st.session_state["decisions"] = {}

enriched_queue = get_enriched_queue()

# --- Sidebar Controls ---
st.sidebar.title("Triage Queue Controls")

has_api_key = bool(GEMINI_API_KEY) and GEMINI_API_KEY != "your_gemini_api_key_here"
use_live_llm = st.sidebar.checkbox(
    "Use Live Gemini Synthesis",
    value=has_api_key,
    disabled=not has_api_key,
    help="Automatically uses deterministic fallback when GEMINI_API_KEY is not configured.",
)

risk_filter = st.sidebar.selectbox(
    "Filter by Risk Tier",
    options=["ALL", "HIGH", "MEDIUM", "LOW"],
    index=0,
)

status_filter = st.sidebar.radio(
    "Review Status",
    options=["All Alerts", "Pending Review", "Completed"],
    index=0,
)

filtered_items = []
for item in enriched_queue:
    aid = item.alert.alert_id
    is_done = aid in st.session_state["decisions"]
    if risk_filter != "ALL" and item.rule_risk_tier != risk_filter:
        continue
    if status_filter == "Pending Review" and is_done:
        continue
    if status_filter == "Completed" and not is_done:
        continue
    filtered_items.append(item)

st.sidebar.divider()
completed_count = len(st.session_state["decisions"])
st.sidebar.metric("Queue Progress", f"{completed_count} / {len(enriched_queue)} Reviewed")

# --- Header ---
st.title("AML/KYC Alert Triage & SAR Synthesis Engine")
st.caption("Zero-PII Surrogate Pipeline • Deterministic Risk Scoring + Gemini Explanation Layer")

tab_queue, tab_single = st.tabs(["1. Alert Triage Queue", "2. Single Transaction Simulator"])

with tab_queue:
    if not filtered_items:
        st.info("No alerts match the selected filter criteria.")
    else:
        alert_labels = {
            item.alert.alert_id: (
                f"{item.alert.alert_id} | {item.alert.subject_id} | "
                f"INR {item.alert.transaction_amount:,.0f} ({item.alert.baseline_12m.spike_multiplier:.1f}x) | "
                f"Tier: {item.rule_risk_tier}"
                + (" [REVIEWED]" if item.alert.alert_id in st.session_state["decisions"] else "")
            )
            for item in filtered_items
        }

        selected_id = st.selectbox(
            "Select Flagged Alert for Investigation",
            options=list(alert_labels.keys()),
            format_func=lambda x: alert_labels[x],
        )

        selected_enriched = next(i for i in filtered_items if i.alert.alert_id == selected_id)
        alert = selected_enriched.alert
        dossier = synthesize_triage_dossier(selected_enriched, use_live_llm=use_live_llm)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Flagged Transaction",
            f"INR {alert.transaction_amount:,.2f}",
            delta=f"+{selected_enriched.baseline_deviation_pct:,.0f}% vs daily avg",
            delta_color="inverse",
        )
        k2.metric(
            "12M Daily Baseline",
            f"INR {alert.baseline_12m.avg_daily_tx_amount:,.2f}",
            delta=f"Prior max: INR {alert.baseline_12m.max_historical_tx_amount:,.0f}",
            delta_color="off",
        )
        k3.metric(
            "Spike Multiplier",
            f"{alert.baseline_12m.spike_multiplier:.1f}x",
            delta=f"Rail: {alert.payment_channel}",
            delta_color="off",
        )
        k4.metric(
            "Composite Risk Score",
            f"{selected_enriched.risk_score} / 100",
            delta=f"{selected_enriched.rule_risk_tier} RISK",
            delta_color="inverse" if selected_enriched.rule_risk_tier == "HIGH" else "normal",
        )

        st.divider()

        left_col, right_col = st.columns([1, 1.25], gap="large")

        with left_col:
            st.subheader("1. Entity & Behavioral Context")
            meta_df = pd.DataFrame(
                [
                    ("Alert ID", alert.alert_id),
                    ("Timestamp", alert.timestamp),
                    ("Subject (Zero-PII)", alert.subject_id),
                    ("Counterparty (Zero-PII)", alert.counterparty_id),
                    ("Account Profile", f"{alert.account_type} ({alert.account_tenure_months} mos)"),
                    ("Counterparty Rel.", alert.behavioral_signals.counterparty_relationship),
                    ("24h Burst Velocity", f"{alert.behavioral_signals.velocity_24h_tx_count} tx(s)"),
                    ("Funds Retention", f"{alert.behavioral_signals.funds_retention_hours:.1f} hours"),
                ],
                columns=["Attribute", "Value"],
            )
            st.dataframe(meta_df, hide_index=True, width="stretch")

            st.markdown("**Detected Red Flags**")
            if selected_enriched.red_flags:
                for flag in selected_enriched.red_flags:
                    st.error(flag)
            else:
                st.caption("No behavioral red flags triggered.")

            st.markdown("**Mitigating Factors**")
            if selected_enriched.mitigating_factors:
                for factor in selected_enriched.mitigating_factors:
                    st.success(factor)
            else:
                st.caption("No mitigating factors identified.")

        with right_col:
            st.subheader("2. AI Triage Synthesis & Disposition")

            action_label = (
                "ESCALATE — DRAFT SAR"
                if dossier.recommended_action == "ESCALATE_SAR"
                else "CLEAR — FALSE POSITIVE"
            )
            st.info(
                f"**AI Recommendation:** {action_label} "
                f"(Confidence: {dossier.confidence_score:.0%} | Mode: `{dossier.generation_mode}`)"
            )

            st.markdown("**Analytical Reasoning**")
            st.write(dossier.reasoning_summary)

            edited_draft = st.text_area(
                "Compliance Narrative (Editable SAR Draft / Clearance Memo)",
                value=dossier.compliance_draft,
                height=230,
                key=f"draft_{alert.alert_id}",
            )

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("Escalate & File SAR", type="primary", width="stretch"):
                    log_audit_decision(
                        selected_enriched,
                        disposition="ESCALATED_SAR",
                        final_narrative=edited_draft,
                        ai_recommendation=dossier.recommended_action,
                        generation_mode=dossier.generation_mode,
                    )
                    st.session_state["decisions"][alert.alert_id] = "ESCALATED_SAR"
                    st.success(f"{alert.alert_id} escalated and logged to audit trail.")

            with btn_col2:
                if st.button("Clear as False Positive", width="stretch"):
                    log_audit_decision(
                        selected_enriched,
                        disposition="CLEARED_FALSE_POSITIVE",
                        final_narrative=edited_draft,
                        ai_recommendation=dossier.recommended_action,
                        generation_mode=dossier.generation_mode,
                    )
                    st.session_state["decisions"][alert.alert_id] = "CLEARED_FALSE_POSITIVE"
                    st.success(f"{alert.alert_id} cleared and logged to audit trail.")

    if AUDIT_LOG_PATH.exists():
        st.divider()
        st.subheader("3. Investigator Audit Log (`data/audit_logs/triage_decisions.jsonl`)")
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            log_rows = [json.loads(line) for line in f if line.strip()]
        if log_rows:
            audit_df = pd.DataFrame(log_rows)[
                [
                    "logged_at_utc",
                    "alert_id",
                    "subject_id",
                    "transaction_amount_inr",
                    "rule_risk_tier",
                    "investigator_disposition",
                ]
            ]
            st.dataframe(audit_df.tail(10), hide_index=True, width="stretch")

with tab_single:
    st.subheader("Simulate & Score a Single Transaction")
    st.caption(
        "Test custom scenarios live through Pydantic Zero-PII validation, deterministic risk scoring, and Gemini explanation."
    )

    with st.form("single_tx_form"):
        c1, c2, c3 = st.columns(3)
        subj_id = c1.text_input("Subject ID (must start with SUBJ_)", value="SUBJ_DEMO_901")
        cpty_id = c2.text_input("Counterparty ID (must start with CPTY_)", value="CPTY_DEMO_404")
        channel = c3.selectbox("Payment Channel", options=["SWIFT", "NEFT", "RTGS", "IMPS"], index=0)

        c4, c5, c6 = st.columns(3)
        tx_amount = c4.number_input("Transaction Amount (INR)", min_value=100.0, value=285000.0, step=5000.0)
        avg_daily = c5.number_input("12M Daily Avg Amount (INR)", min_value=100.0, value=7200.0, step=500.0)
        tenure_mos = c6.number_input("Account Tenure (Months)", min_value=1, value=8, step=1)

        c7, c8, c9 = st.columns(3)
        acct_type = c7.selectbox("Account Type", options=["RETAIL_SAVINGS", "SME_CURRENT", "FREELANCE_CURRENT"])
        cpty_rel = c8.selectbox(
            "Counterparty Relationship",
            options=[
                "NEW_FIRST_TIME_COUNTERPARTY",
                "UNLINKED_HIGH_VELOCITY_NODE",
                "CROSS_BORDER_NEW_ENTITY",
                "KNOWN_RECURRING_VENDOR_12M",
                "VERIFIED_INSTITUTIONAL_SOURCE",
            ],
        )
        velocity_24h = c9.number_input("24h Transaction Velocity", min_value=1, value=5, step=1)
        retention_hrs = st.slider("Funds Retention Window (Hours)", min_value=0.1, max_value=120.0, value=0.8, step=0.1)

        submitted = st.form_submit_button("Evaluate Transaction", type="primary")

    if submitted:
        try:
            custom_alert = AlertRecord(
                alert_id="ALT-DEMO-LIVE",
                timestamp=datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M"),
                subject_id=subj_id.strip(),
                counterparty_id=cpty_id.strip(),
                ibm_source_flag=1 if tx_amount / avg_daily >= 20 else 0,
                account_type=acct_type,
                account_tenure_months=int(tenure_mos),
                payment_channel=channel,
                currency="INR",
                transaction_amount=float(tx_amount),
                baseline_12m=Baseline12M(
                    avg_daily_tx_amount=float(avg_daily),
                    max_historical_tx_amount=round(float(avg_daily) * 2.2, 2),
                    avg_monthly_tx_count=22,
                    spike_multiplier=round(float(tx_amount) / float(avg_daily), 2),
                ),
                behavioral_signals=BehavioralSignals(
                    scenario_archetype="LIVE_SIMULATION",
                    counterparty_relationship=cpty_rel,
                    velocity_24h_tx_count=int(velocity_24h),
                    funds_retention_hours=float(retention_hrs),
                    triggered_rule="RULE_HIGH_VALUE_BASELINE_DEVIATION",
                ),
                ground_truth_label="ANOMALY_ESCALATE",
            )
            custom_enriched = enrich_alert(custom_alert)
            custom_dossier = synthesize_triage_dossier(custom_enriched, use_live_llm=use_live_llm)

            st.success(
                f"Risk Score: **{custom_enriched.risk_score}/100 ({custom_enriched.rule_risk_tier})** | "
                f"Recommendation: **{custom_dossier.recommended_action}** (`{custom_dossier.generation_mode}`)"
            )
            st.write("**Analytical Reasoning:**", custom_dossier.reasoning_summary)
            st.text_area("Generated Compliance Narrative", value=custom_dossier.compliance_draft, height=180)
        except Exception as exc:
            st.error(f"Validation error: {exc}")

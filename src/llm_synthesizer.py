import json
import os
from typing import Literal
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from src.config import GEMINI_API_KEY, LLM_MODEL
from src.mock_enricher import EnrichedAlert


class TriageDossier(BaseModel):
    alert_id: str
    recommended_action: Literal["ESCALATE_SAR", "CLEAR_ALERT"]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning_summary: str
    compliance_draft: str
    generation_mode: Literal["GEMINI_LIVE", "DETERMINISTIC_FALLBACK"]


SYSTEM_PROMPT = """You are a senior AML/KYC financial crime investigator assistant.
Evaluate the enriched transaction alert (strictly Zero-PII surrogate IDs) and return a JSON object with:
- "recommended_action": either "ESCALATE_SAR" (for suspicious activity requiring regulatory filing) or "CLEAR_ALERT" (for explainable false positives).
- "confidence_score": float between 0.0 and 1.0.
- "reasoning_summary": concise 2-3 sentence analytical summary comparing the transaction spike against 12-month baseline and behavioral signals.
- "compliance_draft":
  * If "ESCALATE_SAR": write a structured 5-W Suspicious Activity Report (SAR) narrative covering WHO (surrogate subject & counterparty), WHAT (INR amount, channel, spike multiplier), WHEN (timestamp & retention window), WHERE (payment rail & account type), and WHY (specific red flags).
  * If "CLEAR_ALERT": write a concise Alert Clearance Justification Memo documenting why the baseline deviation is benign based on mitigating factors.
"""


def _build_zero_pii_payload(enriched: EnrichedAlert) -> dict:
    a = enriched.alert
    return {
        "alert_id": a.alert_id,
        "timestamp": a.timestamp,
        "subject_id": a.subject_id,
        "counterparty_id": a.counterparty_id,
        "account_type": a.account_type,
        "account_tenure_months": a.account_tenure_months,
        "payment_channel": a.payment_channel,
        "currency": a.currency,
        "transaction_amount": a.transaction_amount,
        "baseline_12m": a.baseline_12m.model_dump(),
        "behavioral_signals": {
            "counterparty_relationship": a.behavioral_signals.counterparty_relationship,
            "velocity_24h_tx_count": a.behavioral_signals.velocity_24h_tx_count,
            "funds_retention_hours": a.behavioral_signals.funds_retention_hours,
            "triggered_rule": a.behavioral_signals.triggered_rule,
        },
        "enrichment_summary": {
            "baseline_deviation_pct": enriched.baseline_deviation_pct,
            "max_historical_ratio": enriched.max_historical_ratio,
            "risk_score": enriched.risk_score,
            "rule_risk_tier": enriched.rule_risk_tier,
            "red_flags": enriched.red_flags,
            "mitigating_factors": enriched.mitigating_factors,
        },
    }


def _generate_fallback_dossier(enriched: EnrichedAlert) -> TriageDossier:
    a = enriched.alert
    is_high_risk = enriched.risk_score >= 40

    if is_high_risk:
        flags_text = " ".join(enriched.red_flags)
        reasoning = (
            f"Subject {a.subject_id} executed an anomalous {a.payment_channel} transfer of "
            f"INR {a.transaction_amount:,.2f} ({a.baseline_12m.spike_multiplier:.1f}x above its "
            f"INR {a.baseline_12m.avg_daily_tx_amount:,.2f} daily baseline) to {a.counterparty_id}. "
            f"Elevated velocity ({a.behavioral_signals.velocity_24h_tx_count} tx/24h), short funds retention "
            f"({a.behavioral_signals.funds_retention_hours:.1f}h), and an unverified counterparty relationship "
            f"indicate potential layering or illicit fund movement requiring SAR escalation."
        )
        draft = (
            f"SUSPICIOUS ACTIVITY REPORT (SAR) NARRATIVE — {a.alert_id}\n"
            f"- WHO: Subject {a.subject_id} ({a.account_type}, tenure {a.account_tenure_months} months) "
            f"transacting with counterparty {a.counterparty_id} ({a.behavioral_signals.counterparty_relationship}).\n"
            f"- WHAT: High-value transfer of INR {a.transaction_amount:,.2f}, representing a "
            f"{a.baseline_12m.spike_multiplier:.1f}x spike over the 12-month daily average of "
            f"INR {a.baseline_12m.avg_daily_tx_amount:,.2f} and {enriched.max_historical_ratio:.2f}x the prior maximum.\n"
            f"- WHEN: Flagged at {a.timestamp} with a {a.behavioral_signals.velocity_24h_tx_count}-transaction "
            f"24-hour burst and {a.behavioral_signals.funds_retention_hours:.1f}-hour funds retention window.\n"
            f"- WHERE: Executed via {a.payment_channel} rail under rule {a.behavioral_signals.triggered_rule}.\n"
            f"- WHY: {flags_text}"
        )
        return TriageDossier(
            alert_id=a.alert_id,
            recommended_action="ESCALATE_SAR",
            confidence_score=round(min(0.97, max(0.65, 0.70 + (enriched.risk_score - 65) * 0.008)), 2),
            reasoning_summary=reasoning,
            compliance_draft=draft,
            generation_mode="DETERMINISTIC_FALLBACK",
        )

    mitigating_text = " ".join(enriched.mitigating_factors)
    reasoning = (
        f"Although the INR {a.transaction_amount:,.2f} {a.payment_channel} transfer exceeds subject "
        f"{a.subject_id}'s routine INR {a.baseline_12m.avg_daily_tx_amount:,.2f} daily average by "
        f"{a.baseline_12m.spike_multiplier:.1f}x, behavioral enrichment confirms a legitimate counterparty "
        f"profile ({a.behavioral_signals.counterparty_relationship}) and normal settlement retention "
        f"({a.behavioral_signals.funds_retention_hours:.1f}h). Recommend clearing as a benign false positive."
    )
    draft = (
        f"ALERT CLEARANCE JUSTIFICATION MEMO — {a.alert_id}\n"
        f"- SUBJECT / COUNTERPARTY: {a.subject_id} -> {a.counterparty_id} ({a.payment_channel}, INR {a.transaction_amount:,.2f})\n"
        f"- TRIGGER REVIEWED: {a.behavioral_signals.triggered_rule} ({a.baseline_12m.spike_multiplier:.1f}x daily baseline variance)\n"
        f"- MITIGATING EVIDENCE: {mitigating_text}\n"
        f"- DISPOSITION: Closed as False Positive. No rapid pass-through layering or unverified counterparty risk observed."
    )
    return TriageDossier(
        alert_id=a.alert_id,
        recommended_action="CLEAR_ALERT",
        confidence_score=round(min(0.96, 0.82 + (40 - enriched.risk_score) * 0.004), 2),
        reasoning_summary=reasoning,
        compliance_draft=draft,
        generation_mode="DETERMINISTIC_FALLBACK",
    )


def synthesize_triage_dossier(
    enriched: EnrichedAlert, use_live_llm: bool = True
) -> TriageDossier:
    """Generates a structured AI triage dossier via Gemini, falling back cleanly if offline."""
    active_key = os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
    has_valid_key = bool(active_key) and active_key != "your_gemini_api_key_here"

    if not use_live_llm or not has_valid_key:
        return _generate_fallback_dossier(enriched)

    try:
        client = genai.Client(api_key=active_key)
        payload = _build_zero_pii_payload(enriched)

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=json.dumps(payload, indent=2),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        raw_json = response.text or "{}"
        parsed = json.loads(raw_json)

        return TriageDossier(
            alert_id=enriched.alert.alert_id,
            recommended_action=parsed["recommended_action"],
            confidence_score=float(parsed["confidence_score"]),
            reasoning_summary=parsed["reasoning_summary"],
            compliance_draft=parsed["compliance_draft"],
            generation_mode="GEMINI_LIVE",
        )
    except Exception:
        return _generate_fallback_dossier(enriched)

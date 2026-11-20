from typing import Literal
from pydantic import BaseModel, Field
from src.config import (
    HIGH_SPIKE_MULTIPLIER_THRESHOLD,
    LOW_RETENTION_HOURS_THRESHOLD,
)
from src.data_validator import AlertRecord

SUSPICIOUS_COUNTERPARTY_RELS = {
    "NEW_FIRST_TIME_COUNTERPARTY",
    "UNLINKED_HIGH_VELOCITY_NODE",
    "CROSS_BORDER_NEW_ENTITY",
}

TRUSTED_COUNTERPARTY_RELS = {
    "KNOWN_RECURRING_VENDOR_12M",
    "VERIFIED_INSTITUTIONAL_SOURCE",
}


class EnrichedAlert(BaseModel):
    alert: AlertRecord
    baseline_deviation_pct: float
    max_historical_ratio: float
    red_flags: list[str] = Field(default_factory=list)
    mitigating_factors: list[str] = Field(default_factory=list)
    risk_score: int = Field(..., ge=0, le=100)
    rule_risk_tier: Literal["HIGH", "MEDIUM", "LOW"]


def enrich_alert(alert: AlertRecord) -> EnrichedAlert:
    """Computes historical baseline deviations and deterministic AML risk signals."""
    tx_amt = alert.transaction_amount
    avg_daily = alert.baseline_12m.avg_daily_tx_amount
    max_hist = alert.baseline_12m.max_historical_tx_amount
    spike_mult = alert.baseline_12m.spike_multiplier

    signals = alert.behavioral_signals
    retention_hrs = signals.funds_retention_hours
    velocity_24h = signals.velocity_24h_tx_count
    cpty_rel = signals.counterparty_relationship

    baseline_deviation_pct = round(((tx_amt - avg_daily) / avg_daily) * 100.0, 1)
    max_historical_ratio = round(tx_amt / max_hist, 2)

    red_flags: list[str] = []
    mitigating_factors: list[str] = []
    score = 30

    if spike_mult >= 35.0:
        red_flags.append(
            f"Extreme baseline deviation: INR {tx_amt:,.2f} is {spike_mult:.1f}x the 12-month daily average (INR {avg_daily:,.2f})."
        )
        score += 20
    elif spike_mult >= HIGH_SPIKE_MULTIPLIER_THRESHOLD:
        red_flags.append(
            f"High-value baseline spike: INR {tx_amt:,.2f} is {spike_mult:.1f}x the 12-month daily average (INR {avg_daily:,.2f})."
        )
        score += 15

    if cpty_rel in SUSPICIOUS_COUNTERPARTY_RELS:
        red_flags.append(
            f"Unverified counterparty profile ({cpty_rel}) with no prior 12-month settlement history."
        )
        score += 30
    elif cpty_rel in TRUSTED_COUNTERPARTY_RELS:
        mitigating_factors.append(
            f"Established counterparty relationship ({cpty_rel}) consistent with historical activity."
        )
        score -= 25

    if retention_hrs <= LOW_RETENTION_HOURS_THRESHOLD:
        red_flags.append(
            f"Rapid pass-through layering indicator: funds retained for only {retention_hrs:.1f} hours."
        )
        score += 12
    elif retention_hrs >= 24.0:
        mitigating_factors.append(
            f"Normal settlement retention: funds remained in account for {retention_hrs:.1f} hours with no rapid outflow."
        )
        score -= 15

    if velocity_24h >= 3:
        red_flags.append(
            f"Elevated 24-hour burst velocity ({velocity_24h} transactions in 24h vs. {alert.baseline_12m.avg_monthly_tx_count}/month norm)."
        )
        score += 10
    elif velocity_24h <= 2:
        mitigating_factors.append(
            f"Low daily transaction frequency ({velocity_24h} transaction(s) in 24h)."
        )
        score -= 10

    if alert.account_tenure_months < 12:
        red_flags.append(
            f"Recently onboarded account (tenure: {alert.account_tenure_months} months)."
        )
        score += 5
    elif alert.account_type == "SME_CURRENT":
        mitigating_factors.append(
            f"Seasoned commercial account ({alert.account_type}, tenure: {alert.account_tenure_months} months)."
        )
        score -= 5

    bounded_score = max(5, min(98, score))

    if bounded_score >= 65:
        tier: Literal["HIGH", "MEDIUM", "LOW"] = "HIGH"
    elif bounded_score >= 40:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    return EnrichedAlert(
        alert=alert,
        baseline_deviation_pct=baseline_deviation_pct,
        max_historical_ratio=max_historical_ratio,
        red_flags=red_flags,
        mitigating_factors=mitigating_factors,
        risk_score=bounded_score,
        rule_risk_tier=tier,
    )

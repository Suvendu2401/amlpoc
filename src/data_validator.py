from typing import Literal
from pydantic import BaseModel, Field, field_validator


class Baseline12M(BaseModel):
    avg_daily_tx_amount: float = Field(..., gt=0)
    max_historical_tx_amount: float = Field(..., gt=0)
    avg_monthly_tx_count: int = Field(..., ge=1)
    spike_multiplier: float = Field(..., gt=0)


class BehavioralSignals(BaseModel):
    scenario_archetype: str
    counterparty_relationship: str
    velocity_24h_tx_count: int = Field(..., ge=1)
    funds_retention_hours: float = Field(..., ge=0)
    triggered_rule: str


class AlertRecord(BaseModel):
    alert_id: str
    timestamp: str
    subject_id: str
    counterparty_id: str
    ibm_source_flag: Literal[0, 1] = 0
    account_type: str
    account_tenure_months: int = Field(..., ge=1)
    payment_channel: Literal["SWIFT", "NEFT", "RTGS", "IMPS"]
    currency: Literal["INR"]
    transaction_amount: float = Field(..., gt=0)
    baseline_12m: Baseline12M
    behavioral_signals: BehavioralSignals
    ground_truth_label: Literal["ANOMALY_ESCALATE", "BENIGN_FALSE_POSITIVE", "UNLABELED_LIVE"] = "UNLABELED_LIVE"

    @field_validator("subject_id")
    @classmethod
    def enforce_surrogate_subject(cls, value: str) -> str:
        if not value.startswith("SUBJ_"):
            raise ValueError(f"Zero-PII violation: subject_id '{value}' must start with 'SUBJ_'")
        return value

    @field_validator("counterparty_id")
    @classmethod
    def enforce_surrogate_counterparty(cls, value: str) -> str:
        if not value.startswith("CPTY_"):
            raise ValueError(f"Zero-PII violation: counterparty_id '{value}' must start with 'CPTY_'")
        return value

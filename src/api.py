from fastapi import FastAPI
from pydantic import BaseModel
from src.config import (
    FLAG_RISK_SCORE_THRESHOLD,
    HIGH_SPIKE_MULTIPLIER_THRESHOLD,
    LLM_MODEL,
    LLM_PROVIDER,
    LOW_RETENTION_HOURS_THRESHOLD,
)
from src.data_validator import AlertRecord
from src.llm_synthesizer import TriageDossier, synthesize_triage_dossier
from src.mock_enricher import EnrichedAlert, enrich_alert

app = FastAPI(
    title="AML/KYC Triage & SAR Engine API",
    version="1.0.0",
    description="Zero-PII AML risk enrichment and Gemini explanation service.",
)


class PredictionResponse(BaseModel):
    alert_id: str
    subject_id: str
    risk_score: int
    rule_risk_tier: str
    flagged: bool
    baseline_deviation_pct: float
    red_flags: list[str]
    mitigating_factors: list[str]


class ExplainResponse(BaseModel):
    prediction: PredictionResponse
    dossier: TriageDossier


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "aml-triage-sar-engine"}


@app.get("/model-info")
def model_info() -> dict:
    return {
        "scoring_engine": "deterministic_behavioral_scorer_v1",
        "flag_score_threshold": FLAG_RISK_SCORE_THRESHOLD,
        "high_spike_multiplier_threshold": HIGH_SPIKE_MULTIPLIER_THRESHOLD,
        "low_retention_hours_threshold": LOW_RETENTION_HOURS_THRESHOLD,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "fallback_enabled": True,
    }


def _to_prediction_response(enriched: EnrichedAlert) -> PredictionResponse:
    return PredictionResponse(
        alert_id=enriched.alert.alert_id,
        subject_id=enriched.alert.subject_id,
        risk_score=enriched.risk_score,
        rule_risk_tier=enriched.rule_risk_tier,
        flagged=enriched.risk_score >= FLAG_RISK_SCORE_THRESHOLD,
        baseline_deviation_pct=enriched.baseline_deviation_pct,
        red_flags=enriched.red_flags,
        mitigating_factors=enriched.mitigating_factors,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_alert(alert: AlertRecord) -> PredictionResponse:
    enriched = enrich_alert(alert)
    return _to_prediction_response(enriched)


@app.post("/explain", response_model=ExplainResponse)
def explain_alert(alert: AlertRecord, use_live_llm: bool = True) -> ExplainResponse:
    enriched = enrich_alert(alert)
    dossier = synthesize_triage_dossier(enriched, use_live_llm=use_live_llm)
    return ExplainResponse(
        prediction=_to_prediction_response(enriched),
        dossier=dossier,
    )

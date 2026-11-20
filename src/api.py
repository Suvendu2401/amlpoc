from fastapi import FastAPI, Query
from pydantic import BaseModel
from src.config import FLAG_RISK_SCORE_THRESHOLD, LLM_MODEL, LLM_PROVIDER
from src.data_validator import AlertRecord
from src.llm_synthesizer import TriageDossier, synthesize_triage_dossier
from src.mock_enricher import EnrichedAlert, enrich_alert

app = FastAPI(
    title="AML/KYC Triage & SAR Engine API",
    version="1.0.0",
    description="Zero-PII behavioral risk scoring and Gemini SAR/clearance synthesis service.",
)


class PredictionResponse(BaseModel):
    alert_id: str
    subject_id: str
    counterparty_id: str
    risk_score: int
    rule_risk_tier: str
    flagged: bool
    baseline_deviation_pct: float
    max_historical_ratio: float
    red_flags: list[str]
    mitigating_factors: list[str]


class ExplanationResponse(BaseModel):
    enrichment: EnrichedAlert
    dossier: TriageDossier


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": "aml-triage-sar-engine",
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
    }


@app.get("/model-info")
def model_info() -> dict:
    return {
        "pipeline_stage": "Behavioral Enrichment + Gemini Synthesis",
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "flag_risk_score_threshold": FLAG_RISK_SCORE_THRESHOLD,
        "zero_pii_enforced": True,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_alert(alert: AlertRecord) -> PredictionResponse:
    enriched = enrich_alert(alert)
    return PredictionResponse(
        alert_id=alert.alert_id,
        subject_id=alert.subject_id,
        counterparty_id=alert.counterparty_id,
        risk_score=enriched.risk_score,
        rule_risk_tier=enriched.rule_risk_tier,
        flagged=enriched.risk_score >= FLAG_RISK_SCORE_THRESHOLD,
        baseline_deviation_pct=enriched.baseline_deviation_pct,
        max_historical_ratio=enriched.max_historical_ratio,
        red_flags=enriched.red_flags,
        mitigating_factors=enriched.mitigating_factors,
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain_alert(
    alert: AlertRecord,
    use_live_llm: bool = Query(default=True, description="Use live Gemini API if configured"),
) -> ExplanationResponse:
    enriched = enrich_alert(alert)
    dossier = synthesize_triage_dossier(enriched, use_live_llm=use_live_llm)
    return ExplanationResponse(enrichment=enriched, dossier=dossier)

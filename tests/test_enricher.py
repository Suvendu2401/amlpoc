from src.alert_loader import load_validated_alerts
from src.llm_synthesizer import synthesize_triage_dossier
from src.mock_enricher import enrich_alert


def test_anomaly_enrichment_and_sar_draft():
    alerts = load_validated_alerts()
    anomaly = next(a for a in alerts if a.ground_truth_label == "ANOMALY_ESCALATE")

    enriched = enrich_alert(anomaly)
    assert enriched.rule_risk_tier == "HIGH"
    assert enriched.risk_score >= 65
    assert len(enriched.red_flags) >= 2

    dossier = synthesize_triage_dossier(enriched, use_live_llm=False)
    assert dossier.recommended_action == "ESCALATE_SAR"
    assert "SUSPICIOUS ACTIVITY REPORT" in dossier.compliance_draft
    assert anomaly.subject_id in dossier.compliance_draft


def test_benign_enrichment_and_clearance_memo():
    alerts = load_validated_alerts()
    benign = next(a for a in alerts if a.ground_truth_label == "BENIGN_FALSE_POSITIVE")

    enriched = enrich_alert(benign)
    assert enriched.rule_risk_tier == "LOW"
    assert enriched.risk_score < 40
    assert len(enriched.mitigating_factors) >= 2

    dossier = synthesize_triage_dossier(enriched, use_live_llm=False)
    assert dossier.recommended_action == "CLEAR_ALERT"
    assert "CLEARANCE JUSTIFICATION MEMO" in dossier.compliance_draft


def test_full_100_alert_pipeline_consistency():
    alerts = load_validated_alerts()
    results = [
        synthesize_triage_dossier(enrich_alert(a), use_live_llm=False)
        for a in alerts
    ]
    escalations = sum(1 for r in results if r.recommended_action == "ESCALATE_SAR")
    clearances = sum(1 for r in results if r.recommended_action == "CLEAR_ALERT")

    assert escalations == 60
    assert clearances == 40

from fastapi.testclient import TestClient
from src.alert_loader import load_validated_alerts
from src.api import app

client = TestClient(app)


def test_health_and_model_info():
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["fallback_enabled"] is True


def test_predict_and_explain_endpoints():
    alerts = load_validated_alerts()
    anomaly_payload = next(
        a for a in alerts if a.ground_truth_label == "ANOMALY_ESCALATE"
    ).model_dump()

    pred = client.post("/predict", json=anomaly_payload)
    assert pred.status_code == 200
    assert pred.json()["flagged"] is True
    assert pred.json()["rule_risk_tier"] == "HIGH"

    exp = client.post("/explain?use_live_llm=false", json=anomaly_payload)
    assert exp.status_code == 200
    assert exp.json()["dossier"]["recommended_action"] == "ESCALATE_SAR"


def test_predict_rejects_pii_payload():
    alerts = load_validated_alerts()
    bad_payload = alerts[0].model_dump()
    bad_payload["subject_id"] = "Ravi_Kumar_PII"

    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422

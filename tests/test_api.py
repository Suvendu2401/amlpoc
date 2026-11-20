from fastapi.testclient import TestClient
from src.alert_loader import load_validated_alerts
from src.api import app

client = TestClient(app)


def test_health_and_model_info():
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_info = client.get("/model-info")
    assert r_info.status_code == 200
    assert r_info.json()["zero_pii_enforced"] is True


def test_predict_and_explain_endpoints():
    alerts = load_validated_alerts()
    anomaly_payload = next(
        a for a in alerts if a.ground_truth_label == "ANOMALY_ESCALATE"
    ).model_dump()

    r_pred = client.post("/predict", json=anomaly_payload)
    assert r_pred.status_code == 200
    assert r_pred.json()["flagged"] is True

    r_exp = client.post("/explain?use_live_llm=false", json=anomaly_payload)
    assert r_exp.status_code == 200
    assert r_exp.json()["dossier"]["recommended_action"] == "ESCALATE_SAR"


def test_predict_rejects_pii_payload():
    alerts = load_validated_alerts()
    bad_payload = alerts[0].model_dump()
    bad_payload["counterparty_id"] = "Ravi_Kumar_HDFC"

    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422

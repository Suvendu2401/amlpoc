import pytest
from pydantic import ValidationError
from src.alert_loader import get_alert_by_id, load_validated_alerts
from src.data_validator import AlertRecord


def test_all_100_sample_alerts_validate():
    alerts = load_validated_alerts()
    assert len(alerts) == 100
    assert alerts[0].alert_id == "ALT-2026-0001"


def test_get_alert_by_id():
    alerts = load_validated_alerts()
    target = get_alert_by_id(alerts, "ALT-2026-0010")
    assert target.subject_id.startswith("SUBJ_")


def test_rejects_non_surrogate_pii_id():
    alerts = load_validated_alerts()
    bad_payload = alerts[0].model_dump()
    bad_payload["subject_id"] = "Rahul_Sharma_9876543210"

    with pytest.raises(ValidationError, match="Zero-PII violation"):
        AlertRecord.model_validate(bad_payload)

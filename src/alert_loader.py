import json
from pathlib import Path
from src.config import SAMPLE_ALERTS_PATH
from src.data_validator import AlertRecord


def load_validated_alerts(file_path: Path = SAMPLE_ALERTS_PATH) -> list[AlertRecord]:
    """Loads JSON alerts from disk and validates every record against the Zero-PII schema."""
    if not file_path.exists():
        raise FileNotFoundError(f"Alert dataset not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    return [AlertRecord.model_validate(record) for record in raw_records]


def get_alert_by_id(alerts: list[AlertRecord], alert_id: str) -> AlertRecord:
    """Fetches a single validated alert by its alert_id."""
    for alert in alerts:
        if alert.alert_id == alert_id:
            return alert
    raise KeyError(f"Alert ID '{alert_id}' not found in queue")

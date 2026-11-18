import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_ALERTS_PATH = DATA_DIR / "processed" / "sample_alerts.json"
AUDIT_LOG_PATH = DATA_DIR / "audit_logs" / "triage_decisions.jsonl"

# LLM Configuration (defaults to safe mock fallback if API key is unset)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Triage Thresholds
HIGH_SPIKE_MULTIPLIER_THRESHOLD = 20.0
LOW_RETENTION_HOURS_THRESHOLD = 4.0

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_ALERTS_PATH = DATA_DIR / "processed" / "sample_alerts.json"
AUDIT_LOG_PATH = DATA_DIR / "audit_logs" / "triage_decisions.jsonl"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

HIGH_SPIKE_MULTIPLIER_THRESHOLD = 20.0
LOW_RETENTION_HOURS_THRESHOLD = 4.0
FLAG_RISK_SCORE_THRESHOLD = 40

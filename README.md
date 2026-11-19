# AML/KYC Triage & SAR Synthesis Engine (PoC)

A Zero-PII, hybrid AML alert triage prototype combining deterministic behavioral risk scoring with structured Gemini explanation synthesis (`gemini-2.5-flash`) and an offline deterministic fallback.

## Architecture Overview
1. **Zero-PII Data Pipeline (`scripts/prepare_ibm_data.py`):** Transforms IBM AML benchmark transactions (`HI-Small_Trans.csv`) into non-attributive surrogate records (`SUBJ_...`, `CPTY_...`) modeling high-value baseline deviations (INR 2L–3.5L spikes vs. INR 5k–10k daily averages).
2. **Schema & Privacy Gate (`src/data_validator.py`):** Enforces Pydantic v2 contracts rejecting any non-surrogate identifier before scoring or LLM synthesis.
3. **Deterministic Risk Scorer (`src/mock_enricher.py`):** Evaluates baseline spike multipliers, counterparty relationship history, 24-hour burst velocity, and funds retention windows to produce a 0–100 risk score (`HIGH` / `MEDIUM` / `LOW`).
4. **LLM Explanation Layer (`src/llm_synthesizer.py`):** Generates structured 5-W SAR drafts or clearance justification memos via Google Gemini (`gemini-2.5-flash`) strictly for flagged alerts, with automatic fallback to deterministic templates when offline.
5. **FastAPI & Streamlit Interfaces (`src/api.py`, `app.py`):** Exposes REST endpoints (`/health`, `/predict`, `/explain`, `/model-info`) and an interactive investigator workspace with JSONL audit logging.

## Repository Structure
```text
amlpoc/
├── data/
│   ├── raw/                  # Gitignored raw IBM Kaggle CSVs
│   ├── processed/            # Committed 100-row Zero-PII sample_alerts.json
│   └── audit_logs/           # Gitignored local JSONL decision logs
├── docs/
│   └── design_notes.md       # Architecture and design notes
├── scripts/
│   └── prepare_ibm_data.py   # Offline IBM AML surrogate data generator
├── src/
│   ├── __init__.py
│   ├── config.py             # Environment & threshold configuration
│   ├── data_validator.py     # Pydantic v2 Zero-PII schema contracts
│   ├── alert_loader.py       # Validated JSON alert loader
│   ├── mock_enricher.py      # Deterministic behavioral enrichment & scoring
│   ├── llm_synthesizer.py    # Gemini 2.5 Flash synthesis + offline fallback
│   └── api.py                # FastAPI prediction & explanation service
├── tests/
│   ├── test_data_validator.py
│   ├── test_enricher.py
│   └── test_api.py
├── app.py                    # Streamlit investigator dashboard & simulator
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup & Installation

### 1. Clone & Create Virtual Environment
```bash
git clone [https://github.com/Suvendu2401/amlpoc.git](https://github.com/Suvendu2401/amlpoc.git)
cd amlpoc
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Core libraries: `streamlit>=1.38.0`, `fastapi>=0.115.0`, `uvicorn>=0.30.0`, `pydantic>=2.8.0`, `google-genai>=1.0.0`, `pandas>=2.2.0`, `numpy>=1.26.0`, `python-dotenv>=1.0.1`, `pytest>=8.3.0`.

### 3. Configure Environment Variables (Optional for Live Gemini Mode)
```bash
cp .env.example .env
```
Edit `.env` and set `GEMINI_API_KEY=<your_key>`. If left unset, the application automatically runs in `DETERMINISTIC_FALLBACK` mode with zero errors.

## Running Verification & Services

### Run Automated Test Suite
```bash
PYTHONPATH=. pytest -v
```

### Launch FastAPI Service (Swagger UI at `http://localhost:8000/docs`)
```bash
uvicorn src.api:app --reload --port 8000
```
* `GET /health` — Service health check
* `GET /model-info` — Active scoring thresholds and LLM configuration
* `POST /predict` — Deterministic risk score and flag status
* `POST /explain` — Risk score + structured SAR/clearance narrative

### Launch Streamlit Investigator Dashboard (`http://localhost:8501`)
```bash
streamlit run app.py
```

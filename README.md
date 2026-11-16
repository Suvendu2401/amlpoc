# AML/KYC Triage & SAR Engine (PoC)

A Zero-PII, AI-assisted AML alert triage prototype designed to reduce manual compliance investigation time from ~45 minutes to under 2 minutes.

## Core Use Case
Detects sudden high-value transaction spikes (**INR 2,00,000 – 3,50,000**) against a customer's routine 12-month daily baseline (**INR 5,000 – 10,000**), automatically enriching behavioral context so an investigator can rapidly escalate true anomalies or dismiss benign false positives.

## Data & Privacy Design
- **Zero PII:** Uses strictly non-attributive surrogate identifiers (`SUBJ_XXXXXXXXX`, `CPTY_XXXXXXXXX`) derived from the IBM AML benchmark schema.
- **Committed Sample Dataset:** `data/processed/sample_alerts.json` contains 100 varied alerts (60 true anomalies, 40 benign false positives) ready for immediate local execution.

## Repository Structure
```text
aml-triage-sar-engine/
├── docs/                  # Architecture diagrams and design notes
├── data/
│   ├── raw/               # Gitignored raw IBM Kaggle CSVs
│   ├── processed/         # Committed 100-row surrogate sample_alerts.json
│   └── audit_logs/        # Gitignored local decision logs
├── scripts/               # Offline dataset preparation (prepare_ibm_data.py)
├── src/                   # Core modular pipeline (loader, validator, enricher, LLM)
├── tests/                 # Unit tests for validation and enrichment
├── app.py                 # Streamlit investigator dashboard
└── requirements.txt       # Python dependencies

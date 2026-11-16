import json
import random
from pathlib import Path
import numpy as np
import pandas as pd

# Fixed seeds for reproducible demo output
random.seed(42)
np.random.seed(42)

RAW_CSV_PATH = Path("data/raw/HI-Small_Trans.csv")
OUTPUT_JSON_PATH = Path("data/processed/sample_alerts.json")

CHANNEL_MAP = {
    "Wire": "SWIFT",
    "ACH": "NEFT",
    "Cheque": "RTGS",
    "Credit Card": "IMPS",
    "Cash": "RTGS",
    "Reinvestment": "NEFT",
}

ANOMALY_ARCHETYPES = [
    {
        "archetype": "SUDDEN_SPIKE_NEW_COUNTERPARTY",
        "account_type": "RETAIL_SAVINGS",
        "counterparty_rel": "NEW_FIRST_TIME_COUNTERPARTY",
        "retention_hrs": (0.2, 2.5),
        "velocity_24h": (3, 7),
    },
    {
        "archetype": "RAPID_PASS_THROUGH_LAYERING",
        "account_type": "RETAIL_SAVINGS",
        "counterparty_rel": "UNLINKED_HIGH_VELOCITY_NODE",
        "retention_hrs": (0.1, 1.0),
        "velocity_24h": (5, 11),
    },
    {
        "archetype": "DORMANT_ACCOUNT_HIGH_VALUE_BURST",
        "account_type": "FREELANCE_CURRENT",
        "counterparty_rel": "CROSS_BORDER_NEW_ENTITY",
        "retention_hrs": (1.0, 6.0),
        "velocity_24h": (2, 4),
    },
]

BENIGN_ARCHETYPES = [
    {
        "archetype": "RECURRING_B2B_VENDOR_SETTLEMENT",
        "account_type": "SME_CURRENT",
        "counterparty_rel": "KNOWN_RECURRING_VENDOR_12M",
        "retention_hrs": (24.0, 96.0),
        "velocity_24h": (1, 2),
    },
    {
        "archetype": "PROPERTY_OR_TAX_REFUND_CREDIT",
        "account_type": "RETAIL_SAVINGS",
        "counterparty_rel": "VERIFIED_INSTITUTIONAL_SOURCE",
        "retention_hrs": (48.0, 168.0),
        "velocity_24h": (1, 1),
    },
]


def build_surrogate_alerts(
    csv_path: Path, n_anomalies: int = 60, n_benign: int = 40
) -> list[dict]:
    """Transforms IBM AML CSV rows into 100 varied, Zero-PII JSON alerts."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw IBM dataset not found at {csv_path}")

    cols = ["Timestamp", "Account", "Account.1", "Amount Paid", "Payment Format", "Is Laundering"]
    pos_chunks = []
    neg_chunks = []
    pos_collected = 0

    # Stream in chunks so we gather enough rare Is Laundering == 1 pairs fast
    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=500000):
        chunk = chunk.dropna(subset=cols)
        chunk = chunk[chunk["Account"] != chunk["Account.1"]]

        pos_sub = chunk[chunk["Is Laundering"] == 1].drop_duplicates(subset=["Account", "Account.1"])
        if not pos_sub.empty:
            pos_chunks.append(pos_sub)
            pos_collected += len(pos_sub)

        if not neg_chunks:
            neg_sub = chunk[chunk["Is Laundering"] == 0].drop_duplicates(subset=["Account", "Account.1"])
            neg_chunks.append(neg_sub.head(500))

        if pos_collected >= n_anomalies * 3:
            break

    pos_all = pd.concat(pos_chunks, ignore_index=True).drop_duplicates(subset=["Account", "Account.1"])
    neg_all = pd.concat(neg_chunks, ignore_index=True).drop_duplicates(subset=["Account", "Account.1"])

    pos_df = pos_all.sample(n=n_anomalies, replace=(len(pos_all) < n_anomalies), random_state=42)
    neg_df = neg_all.sample(n=n_benign, replace=(len(neg_all) < n_benign), random_state=42)

    combined = pd.concat([pos_df, neg_df], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    alerts = []
    for idx, row in combined.iterrows():
        is_anomaly = int(row["Is Laundering"]) == 1
        profile = random.choice(ANOMALY_ARCHETYPES if is_anomaly else BENIGN_ARCHETYPES)

        # Routine daily baseline: INR 5,000 to 10,000
        avg_daily = round(float(np.random.uniform(5100.0, 9950.0)), 2)
        max_hist = round(avg_daily * float(np.random.uniform(1.6, 2.8)), 2)
        monthly_count = int(np.random.randint(14, 38))

        # High-value flagged spike: INR 2,00,000 to 3,50,000
        flagged_amt = round(float(np.random.uniform(202000.0, 348000.0)), 2)
        spike_mult = round(flagged_amt / avg_daily, 2)

        raw_format = str(row["Payment Format"]).strip()
        channel = CHANNEL_MAP.get(raw_format, "NEFT")

        alert = {
            "alert_id": f"ALT-2026-{idx + 1:04d}",
            "timestamp": str(row["Timestamp"]),
            "subject_id": f"SUBJ_{row['Account']}",
            "counterparty_id": f"CPTY_{row['Account.1']}",
            "ibm_source_flag": int(row["Is Laundering"]),
            "account_type": profile["account_type"],
            "account_tenure_months": int(np.random.randint(4, 72)),
            "payment_channel": channel,
            "currency": "INR",
            "transaction_amount": flagged_amt,
            "baseline_12m": {
                "avg_daily_tx_amount": avg_daily,
                "max_historical_tx_amount": max_hist,
                "avg_monthly_tx_count": monthly_count,
                "spike_multiplier": spike_mult,
            },
            "behavioral_signals": {
                "scenario_archetype": profile["archetype"],
                "counterparty_relationship": profile["counterparty_rel"],
                "velocity_24h_tx_count": random.randint(*profile["velocity_24h"]),
                "funds_retention_hours": round(
                    random.uniform(*profile["retention_hrs"]), 1
                ),
                "triggered_rule": "RULE_HIGH_VALUE_BASELINE_DEVIATION",
            },
            "ground_truth_label": (
                "ANOMALY_ESCALATE" if is_anomaly else "BENIGN_FALSE_POSITIVE"
            ),
        }
        alerts.append(alert)

    return alerts


if __name__ == "__main__":
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = build_surrogate_alerts(RAW_CSV_PATH, n_anomalies=60, n_benign=40)

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} surrogate alerts to {OUTPUT_JSON_PATH}")

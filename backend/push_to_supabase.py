"""
Step 4 — Push Bearing 1's readings + computed anomalies into Supabase,
populating the Time-Series Store from the architecture document.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

BEARING_ID = "bearing_1"
BATCH_SIZE = 200

def main():
    df = pd.read_csv("bearing2_anomalies.csv", parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows from bearing2_anomalies.csv")

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    records = []
    for _, row in df.iterrows():
        records.append({
            "bearing_id": BEARING_ID,
            "reading_time": row["timestamp"].isoformat(),
            "rms": float(row["bearing_1_rms"]),
            "kurtosis": float(row["bearing_1_kurtosis"]),
            "bpfo_energy": float(row["bearing_1_bpfo_energy"]),
            "anomaly_type": str(row["anomaly_type"]),
            "severity": str(row["severity"]),
            "pct_above_baseline": float(row["pct_above_baseline"]),
        })

    print(f"Pushing {len(records)} records in batches of {BATCH_SIZE}...")
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        supabase.table("bearing_readings").insert(batch).execute()
        print(f"  Inserted rows {i} to {i + len(batch)}")

    print("\nDone. Check Supabase Table Editor -> bearing_readings to confirm.")

if __name__ == "__main__":
    main()
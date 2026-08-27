"""
Step 3 — Anomaly Detection: flag deviations from healthy baseline,
classify anomaly type, and visualize flagged points on the trend.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = ["bearing_1_rms", "bearing_1_kurtosis", "bearing_1_bpfo_energy"]
BASELINE_WINDOW = 200          # first ~33 hours, confirmed healthy from the plot
MIN_CONSECUTIVE = 3            # require 3 points in a row to avoid flagging single-point noise
Z_THRESHOLD = 4                # how many std-devs above baseline counts as anomalous

# Severity is graded separately, as % above the healthy baseline mean —
# more interpretable than z-score, and not distorted by how tight the
# baseline std happens to be.
MILD_PCT_ABOVE = 20            # 20% above healthy baseline -> Mild
SEVERE_PCT_ABOVE = 100         # 100% above healthy baseline (double) -> Severe

def compute_baseline(df: pd.DataFrame) -> dict:
    baseline = {}
    for col in FEATURES:
        window = df[col].iloc[:BASELINE_WINDOW]
        baseline[col] = {"mean": window.mean(), "std": window.std()}
    return baseline

def compute_zscores(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    for col in FEATURES:
        mean, std = baseline[col]["mean"], baseline[col]["std"]
        df[f"{col}_z"] = (df[col] - mean) / std
    return df

def flag_anomalies(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    rms_z = df["bearing_1_rms_z"]
    bpfo_z = df["bearing_1_bpfo_energy_z"]

    rms_flag = (rms_z > Z_THRESHOLD).rolling(MIN_CONSECUTIVE).sum() >= MIN_CONSECUTIVE
    bpfo_flag = (bpfo_z > Z_THRESHOLD).rolling(MIN_CONSECUTIVE).sum() >= MIN_CONSECUTIVE

    df["anomaly_type"] = "Normal"
    df.loc[rms_flag, "anomaly_type"] = "Gradual Degradation Trend"
    df.loc[bpfo_flag, "anomaly_type"] = "Outer Race Fault Onset"  # takes priority if both fire

    # Severity graded by % above healthy baseline mean (using RMS as the
    # primary health indicator)
    rms_baseline_mean = baseline["bearing_1_rms"]["mean"]
    pct_above = (df["bearing_1_rms"] - rms_baseline_mean) / rms_baseline_mean * 100

    df["severity"] = "Normal"
    df.loc[df["anomaly_type"] != "Normal", "severity"] = "Mild"
    df.loc[(df["anomaly_type"] != "Normal") & (pct_above > MILD_PCT_ABOVE), "severity"] = "Mild"
    df.loc[(df["anomaly_type"] != "Normal") & (pct_above > SEVERE_PCT_ABOVE), "severity"] = "Severe"
    df["pct_above_baseline"] = pct_above.round(1)

    return df

def main():
    df = pd.read_csv("bearing2_features.csv", parse_dates=["timestamp"])

    baseline = compute_baseline(df)
    print("Baseline (healthy) stats:")
    for col, stats in baseline.items():
        print(f"  {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    df = compute_zscores(df, baseline)
    df = flag_anomalies(df, baseline)

    anomalies = df[df["anomaly_type"] != "Normal"]
    print(f"\nFlagged {len(anomalies)} anomalous snapshots out of {len(df)} total.")
    print("\nAnomaly type counts:")
    print(df["anomaly_type"].value_counts())
    print("\nSeverity counts:")
    print(df["severity"].value_counts())

    df.to_csv("bearing2_anomalies.csv", index=False)
    print("\nSaved full results to bearing2_anomalies.csv")

    # Visualize: RMS trend with anomaly points marked, color-coded by severity
    mild = df[df["severity"] == "Mild"]
    severe = df[df["severity"] == "Severe"]

    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["bearing_1_rms"], label="RMS", color="tab:blue", linewidth=1)
    plt.scatter(mild["timestamp"], mild["bearing_1_rms"],
                color="orange", label="Mild — Monitor", zorder=5, s=15)
    plt.scatter(severe["timestamp"], severe["bearing_1_rms"],
                color="red", label="Severe — Act Now", zorder=6, s=20)
    plt.xlabel("Time")
    plt.ylabel("RMS")
    plt.title("Bearing 1 — RMS Trend with Severity-Tiered Anomaly Flags")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bearing1_anomalies.png")
    print("Saved plot to bearing1_anomalies.png")
    plt.show()

if __name__ == "__main__":
    main()
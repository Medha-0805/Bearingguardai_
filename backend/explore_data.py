"""
Step 2b — Visualize extracted features (RMS, Kurtosis, BPFO energy)
for Bearing 1, to confirm they all flag the same failure window.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("bearing2_features.csv", parse_dates=["timestamp"])

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(df["timestamp"], df["bearing_1_rms"], color="tab:blue")
axes[0].set_ylabel("RMS")
axes[0].set_title("Bearing 1 — RMS, Kurtosis, and BPFO Energy Over Time")

axes[1].plot(df["timestamp"], df["bearing_1_kurtosis"], color="tab:orange")
axes[1].set_ylabel("Kurtosis")

axes[2].plot(df["timestamp"], df["bearing_1_bpfo_energy"], color="tab:red")
axes[2].set_ylabel("BPFO Energy")
axes[2].set_xlabel("Time")

plt.tight_layout()
plt.savefig("bearing1_features_compare.png")
print("Saved comparison plot to bearing1_features_compare.png")
plt.show()
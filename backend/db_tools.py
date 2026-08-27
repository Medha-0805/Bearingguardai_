"""
Backend query functions ("tools") the AI agent will call.
Each one queries the bearing_readings table in Supabase directly —
no AI involved here, just real data lookups. The agent's only job
later is deciding which of these to call, and with what arguments.
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# This dataset is historical (Feb 2004), so "now" for relative queries
# ("last 3 days") is anchored to the last actual reading, not today's
# real-world date.
DATASET_END = datetime(2004, 2, 19, 6, 22, 39)

# Matches the "Severe" tier from the anomaly model: 100% above the
# healthy baseline RMS of ~0.0773
ALERT_THRESHOLD_RMS = 0.1546

# If the latest reading drops below this fraction of the recent peak,
# right after that peak was itself Severe, treat it as equipment
# stoppage/failure rather than a genuine return to health — this is
# what actually happens at the end of a run-to-failure recording.
DROPOUT_RATIO = 0.3
RECENT_WINDOW_FOR_PEAK = 10


def get_trend(bearing_id: str, start_date: str, end_date: str) -> dict:
    """Return RMS/kurtosis/bpfo_energy readings for a bearing between two dates."""
    response = (
        supabase.table("bearing_readings")
        .select("reading_time, rms, kurtosis, bpfo_energy, anomaly_type, severity, pct_above_baseline")
        .eq("bearing_id", bearing_id)
        .gte("reading_time", start_date)
        .lte("reading_time", end_date)
        .order("reading_time")
        .execute()
    )
    return {"bearing_id": bearing_id, "start_date": start_date, "end_date": end_date, "readings": response.data}


def get_anomalies(bearing_id: str, start_date: str, end_date: str) -> dict:
    """
    Return a SUMMARY of flagged anomalies for a bearing between two dates —
    not every raw row. A multi-day window can contain hundreds of anomalous
    readings (10-minute sampling), and handing all of that to the LLM as
    raw text blows past its token limit for no real benefit. Counts, the
    first/last anomaly time, and a small sample give the LLM everything it
    needs to answer well, while staying small regardless of how many rows
    actually matched.
    """
    response = (
        supabase.table("bearing_readings")
        .select("reading_time, anomaly_type, severity, pct_above_baseline")
        .eq("bearing_id", bearing_id)
        .neq("anomaly_type", "Normal")
        .gte("reading_time", start_date)
        .lte("reading_time", end_date)
        .order("reading_time")
        .execute()
    )
    rows = response.data
    if not rows:
        return {
            "bearing_id": bearing_id, "start_date": start_date, "end_date": end_date,
            "total_anomalies": 0, "summary": "No anomalies found in this period.",
        }

    type_counts, severity_counts = {}, {}
    for r in rows:
        type_counts[r["anomaly_type"]] = type_counts.get(r["anomaly_type"], 0) + 1
        severity_counts[r["severity"]] = severity_counts.get(r["severity"], 0) + 1

    peak = max(rows, key=lambda r: r["pct_above_baseline"])

    return {
        "bearing_id": bearing_id,
        "start_date": start_date,
        "end_date": end_date,
        "total_anomalies": len(rows),
        "anomaly_type_counts": type_counts,
        "severity_counts": severity_counts,
        "first_anomaly_time": rows[0]["reading_time"],
        "last_anomaly_time": rows[-1]["reading_time"],
        "peak_anomaly": peak,
        "sample_first_5": rows[:5],
        "sample_last_5": rows[-5:],
    }


def get_threshold_status(bearing_id: str) -> dict:
    """
    Return the most recent reading vs. the configured alert threshold.

    Looks at the last RECENT_WINDOW_FOR_PEAK readings, not just the very
    last one — a single final reading near zero, right after a run of
    Severe readings, almost certainly means the equipment stopped/failed,
    not that it suddenly became healthy.
    """
    response = (
        supabase.table("bearing_readings")
        .select("reading_time, rms, severity")
        .eq("bearing_id", bearing_id)
        .order("reading_time", desc=True)
        .limit(RECENT_WINDOW_FOR_PEAK)
        .execute()
    )
    if not response.data:
        return {"bearing_id": bearing_id, "error": "No readings found."}

    recent = response.data  # most recent first
    latest = recent[0]
    recent_peak_rms = max(r["rms"] for r in recent)
    recent_peak_severity = "Severe" if any(r["severity"] == "Severe" for r in recent) else \
                            "Mild" if any(r["severity"] == "Mild" for r in recent) else "Normal"

    looks_like_dropout = (
        recent_peak_severity == "Severe"
        and latest["rms"] < DROPOUT_RATIO * recent_peak_rms
    )

    if looks_like_dropout:
        status_label = "Equipment Stoppage Suspected (sudden flatline after Severe readings)"
    else:
        status_label = latest["severity"]

    is_above = latest["rms"] > ALERT_THRESHOLD_RMS
    margin_pct = (latest["rms"] - ALERT_THRESHOLD_RMS) / ALERT_THRESHOLD_RMS * 100

    return {
        "bearing_id": bearing_id,
        "latest_reading_time": latest["reading_time"],
        "latest_rms": latest["rms"],
        "alert_threshold_rms": ALERT_THRESHOLD_RMS,
        "is_above_threshold": is_above,
        "margin_pct": round(margin_pct, 1),
        "recent_peak_rms": round(recent_peak_rms, 4),
        "status": status_label,
        "requires_attention": is_above or looks_like_dropout,
    }


def get_health_summary(bearing_id: str, window_days: int) -> dict:
    """
    Return a plain-data summary of health over the last N days of available data.

    Reports the PEAK severity/RMS observed in the window as the headline
    status, not just the literal final reading — a window that was 99%
    anomalous shouldn't be summarized as "Normal" just because the very
    last point happened to be a post-failure flatline.
    """
    response = (
        supabase.table("bearing_readings")
        .select("reading_time, rms, anomaly_type, severity")
        .eq("bearing_id", bearing_id)
        .order("reading_time")
        .execute()
    )
    all_rows = response.data
    if not all_rows:
        return {"bearing_id": bearing_id, "error": "No readings found."}

    cutoff = DATASET_END - timedelta(days=window_days)
    window_rows = [r for r in all_rows if r["reading_time"] >= cutoff.isoformat()]
    if not window_rows:
        window_rows = all_rows

    first_rms = window_rows[0]["rms"]
    peak_row = max(window_rows, key=lambda r: r["rms"])
    last_row = window_rows[-1]

    trend_direction = "rising" if peak_row["rms"] > first_rms * 1.2 else "stable"

    anomaly_count = sum(1 for r in window_rows if r["anomaly_type"] != "Normal")
    severe_count = sum(1 for r in window_rows if r["severity"] == "Severe")

    ended_abruptly = (
        peak_row["severity"] == "Severe"
        and last_row["rms"] < DROPOUT_RATIO * peak_row["rms"]
        and last_row["reading_time"] == all_rows[-1]["reading_time"]
    )

    return {
        "bearing_id": bearing_id,
        "window_days": window_days,
        "readings_in_window": len(window_rows),
        "trend_direction": trend_direction,
        "rms_start": round(first_rms, 4),
        "peak_rms": round(peak_row["rms"], 4),
        "peak_severity": peak_row["severity"],
        "peak_time": peak_row["reading_time"],
        "anomaly_count": anomaly_count,
        "severe_count": severe_count,
        "ended_with_suspected_equipment_stoppage": ended_abruptly,
    }


if __name__ == "__main__":
    print("get_trend sample (last 2 days of data):")
    result = get_trend("bearing_1", "2004-02-18", "2004-02-19")
    print(f"  Returned {len(result['readings'])} readings")

    print("\nget_anomalies sample (Feb 16-19):")
    result = get_anomalies("bearing_1", "2004-02-16", "2004-02-19")
    print(f"  Total anomalies: {result['total_anomalies']}")
    print(f"  By type: {result.get('anomaly_type_counts')}")
    print(f"  By severity: {result.get('severity_counts')}")

    print("\nget_threshold_status sample:")
    print(" ", get_threshold_status("bearing_1"))

    print("\nget_health_summary sample (last 3 days):")
    print(" ", get_health_summary("bearing_1", window_days=3))
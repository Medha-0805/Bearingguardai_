"""
Step 7 -- Live ML inference tools ("tools") the AI agent (and the API)
can call: load the trained models from backend/models/ once, and run
them against the latest real reading already stored in Supabase.

Companion to db_tools.py -- same idea (plain functions the agent can
call that return real data, no LLM involved in the computation itself)
but backed by the trained models in models/ instead of raw queries.

See train_model.py's module docstring for exactly what these models
were trained on and, importantly, what they are NOT (this is a binary
Normal-vs-Anomalous health-state classifier, not a multi-fault-type
classifier -- see there for why).
"""

import warnings
from pathlib import Path

import joblib

from db_tools import supabase

MODELS_DIR = Path(__file__).parent / "models"

# train_model.py fits both models on a pandas DataFrame (so it can label
# columns for feature-importance reporting), but the deployed API
# intentionally has no pandas dependency (requirements-server.txt keeps
# it out — nothing else at runtime needs it). Predicting from a plain
# list instead makes sklearn emit a cosmetic "X does not have valid
# feature names" warning on every call; the values still line up
# correctly (same order as train_model.py's FEATURE_COLS), so it's safe
# to silence rather than pull in pandas just to build a 1-row frame.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)

# Loaded lazily on first use and cached -- training these takes a
# couple of seconds, but the API process should only ever pay that
# cost once, not on every request.
_classifier = None
_regressor = None


def _load_models() -> tuple:
    global _classifier, _regressor
    if _classifier is None:
        _classifier = joblib.load(MODELS_DIR / "fault_classifier.joblib")
    if _regressor is None:
        _regressor = joblib.load(MODELS_DIR / "rul_regressor.joblib")
    return _classifier, _regressor


def _latest_reading(bearing_id: str) -> dict | None:
    response = (
        supabase.table("bearing_readings")
        .select("reading_time, rms, kurtosis, bpfo_energy")
        .eq("bearing_id", bearing_id)
        .order("reading_time", desc=True)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


def predict_fault_type(bearing_id: str) -> dict:
    """
    Classify the bearing's current health state from its latest stored
    reading, using the trained Random Forest health-state classifier.

    NOTE: despite the name (kept for API clarity on what question this
    answers), the model outputs "Normal" or "Anomalous" -- it does NOT
    identify which fault type (inner race / outer race / ball) is
    present. This project's genuine labeled data covers only one fault
    type (an outer-race defect in bearing_1), so a real multi-fault
    classifier can't honestly be claimed here. See train_model.py.
    """
    reading = _latest_reading(bearing_id)
    if reading is None:
        return {"bearing_id": bearing_id, "error": "No readings found."}

    classifier, _ = _load_models()
    features = [[reading["rms"], reading["kurtosis"], reading["bpfo_energy"]]]
    predicted = classifier.predict(features)[0]
    proba = dict(zip(classifier.classes_, classifier.predict_proba(features)[0]))

    return {
        "bearing_id": bearing_id,
        "reading_time": reading["reading_time"],
        "predicted_health_state": predicted,
        "confidence": round(float(proba[predicted]), 4),
        "class_probabilities": {k: round(float(v), 4) for k, v in proba.items()},
        "model_scope_note": (
            "Binary Normal-vs-Anomalous health-state classifier (Random Forest) "
            "trained on the NASA/IMS Test 2 bearing dataset -- not a multi-fault-type "
            "(inner race / outer race / ball) classifier. Only one genuine fault type "
            "(outer race) exists in this project's training data."
        ),
    }


def estimate_rul(bearing_id: str) -> dict:
    """
    Estimate Remaining Useful Life (in hours) from the bearing's latest
    stored reading, using the trained Random Forest RUL regressor.

    Trained on bearing_1's single genuine run-to-failure trajectory --
    treat this as a rough planning signal, not a precise countdown. See
    models/metrics_report.json and TRAINING_REPORT.md for the measured
    MAE/RMSE and the documented extrapolation caveat.
    """
    reading = _latest_reading(bearing_id)
    if reading is None:
        return {"bearing_id": bearing_id, "error": "No readings found."}

    _, regressor = _load_models()
    features = [[reading["rms"], reading["kurtosis"], reading["bpfo_energy"]]]
    predicted_hours = max(float(regressor.predict(features)[0]), 0.0)  # RUL can't be negative

    return {
        "bearing_id": bearing_id,
        "reading_time": reading["reading_time"],
        "estimated_rul_hours": round(predicted_hours, 1),
        "estimated_rul_days": round(predicted_hours / 24, 2),
        "model_scope_note": (
            "Trained on bearing_1's single genuine run-to-failure trajectory "
            "(NASA/IMS Test 2). Held-out validation MAE was ~28.5 hours on the final, "
            "hardest-to-predict portion of that trajectory -- treat this as a rough "
            "planning signal, not a precise countdown. See metrics_report.json / "
            "TRAINING_REPORT.md for full validation details and caveats."
        ),
    }


if __name__ == "__main__":
    print("predict_fault_type sample:")
    print(" ", predict_fault_type("bearing_1"))

    print("\nestimate_rul sample:")
    print(" ", estimate_rul("bearing_1"))

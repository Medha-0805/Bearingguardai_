"""
Step 6 -- Train ML Models: a Random Forest bearing health-state classifier
and a Random Forest Remaining-Useful-Life (RUL) regressor.

DATA -- WHAT THIS IS ACTUALLY TRAINED ON, AND WHY
--------------------------------------------------
The features here come from bearing2_features.csv, which is Test 2 of
the NASA/IMS (University of Cincinnati Center for Intelligent
Maintenance Systems) bearing prognostics dataset: four bearings on one
shaft, sampled every 10 minutes from 2004-02-12 to 2004-02-19, until
the test was stopped because bearing 1 developed a real outer-race
defect. This is documented in the dataset's own README and confirmed
by essentially every paper that cites it -- it is not something this
project inferred or guessed. Bearings 2, 3 and 4 ran the entire test
without failure.

That gives ONE genuine, ground-truth fault outcome (bearing 1 failed
via an outer-race defect; bearings 2-4 did not fail) -- not four fault
*types*. There is no genuine inner-race or ball/roller-defect example
anywhere in the data available to this project (that would require the
raw vibration files from IMS Test 1 and/or Test 3, which are not
present here -- see TRAINING_REPORT.md for the real, documented attempt
made to obtain them and why it did not make it into this model).

So, to be precise about what is and is not being claimed:

  1. FAULT/HEALTH-STATE CLASSIFIER -- binary: "Normal" vs "Anomalous".
     This is NOT a multi-class fault-type classifier (it cannot tell
     inner-race from outer-race from ball-defect) -- it is a genuine,
     properly-validated classifier for "is this bearing behaving
     normally or not", trained across all four bearings' real feature
     data with a real documented failed/did-not-fail outcome per
     bearing. The exact moment within bearing 1's own trajectory that
     is labeled "Anomalous" still comes from the z-score onset logic in
     detect_anomalies.py (a heuristic, not a post-mortem inspection
     timestamp) -- but bearings 2-4 being "Normal" for their entire run
     is genuine ground truth, not heuristic.

  2. RUL REGRESSOR -- trained on bearing 1's one real run-to-failure
     trajectory. The target (hours until the run ends) is genuine
     ground truth, since the recording stops when bearing 1 actually
     failed. The big caveat: with only one physical unit's trajectory
     available, this is validated by holding out the final, hardest
     portion of THAT SAME trajectory chronologically -- it has not been
     validated against a *different* bearing's failure run, so treat
     any claim of generalizing to other bearings/machines cautiously.

Do not describe this project as doing "multi-fault-type classification
(inner race / outer race / ball)" unless TRAINING_REPORT.md says the
raw multi-test data was actually obtained and incorporated.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit

FEATURE_COLS = ["rms", "kurtosis", "bpfo_energy"]
BEARING_IDS = [1, 2, 3, 4]

# Fraction of the shared timeline used for training the classifier --
# chronological, not random, since these are time-series readings and a
# random shuffle would let the model train on points a few minutes away
# from ones it's "tested" on (leakage of the future into the past).
CLASSIFIER_TRAIN_FRACTION = 0.75

# Bearing 1's own trajectory is the only genuine run-to-failure example
# available, so the RUL regressor is trained/tested purely on it, again
# with a chronological (not random) split -- trained on the earlier,
# healthier portion, tested on the later portion approaching failure,
# which is both the realistic use case and the harder test.
RUL_TRAIN_FRACTION = 0.70

MODELS_DIR = Path(__file__).parent / "models"


def load_long_format() -> pd.DataFrame:
    """
    Reshape the wide 4-bearings-per-row CSV into one row per
    (timestamp, bearing) with a shared feature schema, and attach the
    health_state label described in the module docstring.
    """
    feat = pd.read_csv("bearing2_features.csv", parse_dates=["timestamp"])
    anomalies = pd.read_csv("bearing2_anomalies.csv", parse_dates=["timestamp"])

    frames = []
    for b in BEARING_IDS:
        sub = feat[["timestamp", f"bearing_{b}_rms", f"bearing_{b}_kurtosis", f"bearing_{b}_bpfo_energy"]].copy()
        sub.columns = ["timestamp", *FEATURE_COLS]
        sub["bearing_id"] = f"bearing_{b}"
        frames.append(sub)
    long_df = pd.concat(frames, ignore_index=True)

    # Ground truth: bearings 2-4 never failed in this test -> every one
    # of their rows is genuinely "Normal". Bearing 1 alone gets the
    # already-computed z-score-based anomaly_type, collapsed to a
    # binary label (Mild + Severe both count as "Anomalous" here --
    # that finer grading is heuristic severity, not a fault-type
    # distinction worth a separate class).
    long_df["health_state"] = "Normal"
    b1_mask = long_df["bearing_id"] == "bearing_1"
    b1_label_by_time = dict(zip(anomalies["timestamp"], anomalies["anomaly_type"]))
    b1_raw_labels = long_df.loc[b1_mask, "timestamp"].map(b1_label_by_time)
    long_df.loc[b1_mask, "health_state"] = np.where(b1_raw_labels == "Normal", "Normal", "Anomalous")

    return long_df.sort_values("timestamp").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by a cutoff timestamp (not a random shuffle) so the test set is strictly later in time than training."""
    unique_times = np.sort(df["timestamp"].unique())
    cutoff = unique_times[int(len(unique_times) * train_fraction)]
    train = df[df["timestamp"] < cutoff]
    test = df[df["timestamp"] >= cutoff]
    return train, test


def train_classifier(long_df: pd.DataFrame) -> dict:
    """Train + evaluate the Normal/Anomalous health-state classifier. Returns a metrics dict for the report."""
    train_df, test_df = chronological_split(long_df, CLASSIFIER_TRAIN_FRACTION)

    X_train, y_train = train_df[FEATURE_COLS], train_df["health_state"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["health_state"]

    # class_weight="balanced" matters here: only ~11% of rows are
    # Anomalous, so an unweighted model could get high accuracy by
    # mostly predicting "Normal" and still miss the class we actually
    # care about catching.
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)
    labels = sorted(y_train.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("\n=== Fault/Health-State Classifier (Normal vs Anomalous) ===")
    print(f"Train rows: {len(train_df)} (up to {train_df['timestamp'].max()})")
    print(f"Test rows:  {len(test_df)} (from {test_df['timestamp'].min()})")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"Confusion matrix (rows=actual, cols=predicted), labels={labels}:")
    print(cm)
    print(classification_report(y_test, y_pred))

    # Report metrics from the honest train/test split above, but for
    # the artifact we actually deploy, refit on ALL available labeled
    # data (standard practice once validation is done) so live
    # inference benefits from the full trajectory, including the
    # failure endgame that was held out for testing.
    final_clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
    )
    final_clf.fit(long_df[FEATURE_COLS], long_df["health_state"])

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final_clf, MODELS_DIR / "fault_classifier.joblib")

    return {
        "model": "RandomForestClassifier",
        "classes": labels,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_period_end": str(train_df["timestamp"].max()),
        "test_period_start": str(test_df["timestamp"].min()),
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "classification_report": report,
        "feature_importances": dict(zip(FEATURE_COLS, [round(x, 4) for x in final_clf.feature_importances_])),
    }


def train_rul_regressor(long_df: pd.DataFrame) -> dict:
    """Train + evaluate the RUL regressor on bearing 1's genuine run-to-failure trajectory."""
    b1 = long_df[long_df["bearing_id"] == "bearing_1"].sort_values("timestamp").reset_index(drop=True)
    last_time = b1["timestamp"].max()
    b1["rul_hours"] = (last_time - b1["timestamp"]).dt.total_seconds() / 3600.0

    train_df, test_df = chronological_split(b1, RUL_TRAIN_FRACTION)
    X_train, y_train = train_df[FEATURE_COLS], train_df["rul_hours"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["rul_hours"]

    reg = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    # Extra validation on top of the single held-out split: TimeSeriesSplit
    # cross-validation across the training portion. This is still all one
    # physical bearing's trajectory (see module docstring caveat), but it
    # checks the model isn't just overfit to one particular cut point.
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = []
    for cv_train_idx, cv_test_idx in tscv.split(X_train):
        cv_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
        cv_model.fit(X_train.iloc[cv_train_idx], y_train.iloc[cv_train_idx])
        cv_pred = cv_model.predict(X_train.iloc[cv_test_idx])
        cv_maes.append(mean_absolute_error(y_train.iloc[cv_test_idx], cv_pred))

    total_span_hours = b1["rul_hours"].max()
    mae_pct = mae / total_span_hours * 100

    print("\n=== Remaining Useful Life (RUL) Regressor (bearing_1 only) ===")
    print(f"Train rows: {len(train_df)} (RUL {y_train.max():.1f}h down to {y_train.min():.1f}h)")
    print(f"Test rows:  {len(test_df)} (RUL {y_test.max():.1f}h down to {y_test.min():.1f}h)")
    print(f"MAE:  {mae:.2f} hours ({mae_pct:.1f}% of the full {total_span_hours:.1f}h trajectory)")
    print(f"RMSE: {rmse:.2f} hours")
    print(f"R^2:  {r2:.4f}  <- expected to look bad, see note below")
    print(f"TimeSeriesSplit CV MAE (5 folds, within training data only): {np.mean(cv_maes):.2f}h +/- {np.std(cv_maes):.2f}h")
    print(
        "NOTE on the negative R^2: the held-out test window covers only the final "
        f"{y_test.max():.0f} hours of a {total_span_hours:.0f}-hour run, so its own variance "
        "is small -- R^2 punishes any real error heavily against that small denominator. "
        "More fundamentally, RandomForestRegressor (like any tree ensemble) cannot "
        "extrapolate past the feature magnitudes it saw in training, and the last stretch "
        "before failure is exactly where RMS/kurtosis/BPFO-energy reach their highest, "
        "never-before-seen values -- so predictions there under-react. This is a real, "
        "documented limitation of applying tree-based regression directly to RUL from a "
        "single degradation trajectory (not a bug), and is exactly why the MAE/RMSE above "
        "-- not R^2 -- are the metrics to trust here."
    )

    # Deployed artifact: refit on the bearing's full trajectory, same
    # reasoning as the classifier above.
    final_reg = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    final_reg.fit(b1[FEATURE_COLS], b1["rul_hours"])

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final_reg, MODELS_DIR / "rul_regressor.joblib")

    return {
        "model": "RandomForestRegressor",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "mae_hours": round(mae, 2),
        "mae_pct_of_trajectory": round(mae_pct, 1),
        "rmse_hours": round(rmse, 2),
        "r2": round(r2, 4),
        "cv_mae_hours_mean": round(float(np.mean(cv_maes)), 2),
        "cv_mae_hours_std": round(float(np.std(cv_maes)), 2),
        "trained_on": "bearing_1 (only genuine run-to-failure trajectory available)",
        "feature_importances": dict(zip(FEATURE_COLS, [round(x, 4) for x in final_reg.feature_importances_])),
        "caveat": (
            "Validated only against the later, unseen-magnitude portion of the same "
            "bearing's own trajectory -- there is no second failure run in this "
            "project's data to test generalization across physically different "
            "bearings. The negative R^2 reflects tree-based regressors' known "
            "inability to extrapolate beyond feature ranges seen in training, not a "
            "bug; MAE/RMSE (in hours) are the metrics to read here. See "
            "train_model.py for the full explanation."
        ),
    }


def main():
    long_df = load_long_format()
    print(f"Loaded {len(long_df)} feature rows across {long_df['bearing_id'].nunique()} bearings "
          f"({long_df['timestamp'].min()} to {long_df['timestamp'].max()}).")
    print(f"Health-state label counts:\n{long_df['health_state'].value_counts()}")

    classifier_metrics = train_classifier(long_df)
    rul_metrics = train_rul_regressor(long_df)

    metadata = {
        "feature_cols": FEATURE_COLS,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    report = {
        "trained_at_utc": metadata["trained_at_utc"],
        "data_source": (
            "NASA/IMS (Univ. of Cincinnati) bearing prognostics dataset, Test 2 -- "
            "4 bearings, 10-min sampling, 2004-02-12 to 2004-02-19, run until bearing_1's "
            "documented outer-race failure ended the test."
        ),
        "scope_note": (
            "fault_classifier is a binary Normal-vs-Anomalous health-state classifier, "
            "NOT a multi-fault-type (inner race / outer race / ball) classifier -- this "
            "project's available data contains only one genuine failure type (outer race, "
            "bearing_1). See train_model.py's module docstring and TRAINING_REPORT.md."
        ),
        "fault_classifier": classifier_metrics,
        "rul_regressor": rul_metrics,
    }
    (MODELS_DIR / "metrics_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved models + metrics_report.json to {MODELS_DIR}/")


if __name__ == "__main__":
    main()

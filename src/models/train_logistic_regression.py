
import os
import glob
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

FEATURES = [
    "distance_m",
    "projection_position",
    "road_bearing",
    "gps_heading",
    "heading_difference",
    "gps_speed_mps",
    "time_delta_s",
    "large_or_invalid_time_gap"
]

TARGET = "is_matching_candidate"
GROUP_COLUMN = "trip_id"
OBSERVATION_COLUMN = "observation_id"

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# 2. CREATE LOGISTIC REGRESSION PIPELINE
# ============================================================

# Simple explanation: Build the Logistic Regression steps for filling gaps, scaling, and learning.
def create_model():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight=None,
                random_state=RANDOM_STATE
            )
        )
    ])


# ============================================================
# 3. LOAD LABELLED DATASET
# ============================================================

print("\n" + "=" * 75)
print("              PATHSNAP LOGISTIC REGRESSION")
print("=" * 75)

print("\nLoading labelled candidate datasets...")

files = sorted(
    glob.glob("experiments/labelled_candidates_*.csv")
)

if not files:
    raise FileNotFoundError(
        "No labelled candidate CSV files found in experiments/"
    )

data = pd.concat(
    [pd.read_csv(file) for file in files],
    ignore_index=True
)

required_columns = FEATURES + [
    TARGET,
    GROUP_COLUMN,
    OBSERVATION_COLUMN
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Convert features and target to numeric values.
for col in FEATURES + [TARGET]:
    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

# Remove rows without target or grouping identifiers.
data = data.dropna(
    subset=[
        TARGET,
        GROUP_COLUMN,
        OBSERVATION_COLUMN
    ]
).copy()

data[TARGET] = data[TARGET].astype(int)

if not set(data[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

X = data[FEATURES]
y = data[TARGET]
groups = data[GROUP_COLUMN]

total_trips = groups.nunique()

total_observations = data[
    [GROUP_COLUMN, OBSERVATION_COLUMN]
].drop_duplicates().shape[0]

print("\nDATASET INFORMATION")

dataset_table = pd.DataFrame({
    "Description": [
        "Total candidate rows",
        "Total trips",
        "Total GPS observations",
        "Number of features",
        "Matching candidates",
        "Non-matching candidates"
    ],
    "Value": [
        f"{len(data):,}",
        f"{total_trips:,}",
        f"{total_observations:,}",
        len(FEATURES),
        f"{(y == 1).sum():,}",
        f"{(y == 0).sum():,}"
    ]
})

print(dataset_table.to_string(index=False))


# ============================================================
# 4. FIVE-FOLD GROUP CROSS-VALIDATION
# ============================================================

if total_trips < N_SPLITS:
    raise ValueError(
        "Not enough trips for 5-fold GroupKFold."
    )

cv = GroupKFold(n_splits=N_SPLITS)

oof_predictions = np.zeros(len(data), dtype=int)
oof_probabilities = np.zeros(len(data), dtype=float)

fold_results = []

total_inference_time = 0.0

print("\nStarting 5-fold GroupKFold evaluation...\n")

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = create_model()

    # Train Logistic Regression on training trips.
    model.fit(X_train, y_train)

    # Measure probability prediction time.
    start_time = time.perf_counter()

    probabilities = model.predict_proba(X_test)[:, 1]

    inference_time = time.perf_counter() - start_time

    total_inference_time += inference_time

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    oof_predictions[test_idx] = predictions
    oof_probabilities[test_idx] = probabilities

    fold_accuracy = accuracy_score(
        y_test,
        predictions
    )

    fold_results.append({
        "Fold": fold,
        "Accuracy (%)": round(fold_accuracy * 100, 2),
        "Inference Time (sec)": round(inference_time, 4),
        "Validation Rows": len(test_idx)
    })

    print(
        f"Fold {fold} completed | "
        f"Accuracy: {fold_accuracy * 100:.2f}% | "
        f"Inference: {inference_time:.4f} sec"
    )


# ============================================================
# 5. CALCULATE CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y,
    oof_predictions
)

macro_f1 = f1_score(
    y,
    oof_predictions,
    average="macro",
    zero_division=0
)

precision = precision_score(
    y,
    oof_predictions,
    zero_division=0
)

recall = recall_score(
    y,
    oof_predictions,
    zero_division=0
)

logloss = log_loss(
    y,
    oof_probabilities,
    labels=[0, 1]
)


# ============================================================
# 6. CALCULATE TOP-1 ACCURACY
# ============================================================

evaluation_data = pd.DataFrame({
    "trip_id": data[GROUP_COLUMN].values,
    "observation_id": data[OBSERVATION_COLUMN].values,
    "actual": y.values,
    "probability": oof_probabilities
})

# Select the highest-probability candidate for each observation.
top_candidates = (
    evaluation_data
    .sort_values(
        "probability",
        ascending=False,
        kind="stable"
    )
    .groupby(
        ["trip_id", "observation_id"],
        sort=False
    )
    .head(1)
)

top1_accuracy = (
    top_candidates["actual"] == 1
).mean()


# ============================================================
# 7. CALCULATE CANDIDATE MISS RATE
# ============================================================

# An observation is a candidate miss if no generated
# candidate for that observation has a positive label.

observation_labels = (
    evaluation_data
    .groupby(
        ["trip_id", "observation_id"]
    )["actual"]
    .max()
)

missed_observations = (
    observation_labels == 0
).sum()

candidate_miss_rate = (
    missed_observations / len(observation_labels)
)


# ============================================================
# 8. CALCULATE INFERENCE TIME
# ============================================================

total_candidates = len(data)

average_time_per_candidate_ms = (
    total_inference_time / total_candidates
) * 1000

average_time_per_1000_candidates_ms = (
    average_time_per_candidate_ms * 1000
)


# ============================================================
# 9. CREATE FINAL METRICS TABLE
# ============================================================

metrics_table = pd.DataFrame([
    ["Macro F1-score", f"{macro_f1 * 100:.2f}%", "Higher is better"],
    ["Top-1 Accuracy", f"{top1_accuracy * 100:.2f}%", "Higher is better"],
    ["Log Loss", f"{logloss:.4f}", "Lower is better"],
    ["Candidate Miss Rate", f"{candidate_miss_rate * 100:.2f}%", "Lower is better"],
    ["Overall Accuracy", f"{accuracy * 100:.2f}%", "Higher is better"],
    ["Precision", f"{precision * 100:.2f}%", "Higher is better"],
    ["Recall", f"{recall * 100:.2f}%", "Higher is better"],
    ["Total Inference Time", f"{total_inference_time:.4f} sec", "Lower is better"],
    ["Average Time per Candidate", f"{average_time_per_candidate_ms:.6f} ms", "Lower is better"],
    ["Average Time per 1000 Candidates", f"{average_time_per_1000_candidates_ms:.4f} ms", "Lower is better"]
], columns=[
    "Evaluation Metric",
    "Final Result",
    "Interpretation"
])


# ============================================================
# 10. DISPLAY ALL RESULTS IN TABLES
# ============================================================

print("\n")
print("=" * 75)
print("                  FINAL EVALUATION REPORT")
print("=" * 75)

print("\nTABLE 1: MODEL PERFORMANCE METRICS")
print(metrics_table.to_string(index=False))

print("\nTABLE 2: FIVE-FOLD CROSS-VALIDATION RESULTS")

fold_table = pd.DataFrame(fold_results)

print(fold_table.to_string(index=False))

print("\nTABLE 3: CONFUSION MATRIX")

cm = confusion_matrix(
    y,
    oof_predictions,
    labels=[0, 1]
)

confusion_table = pd.DataFrame(
    cm,
    index=[
        "Actual Not Matching",
        "Actual Matching"
    ],
    columns=[
        "Predicted Not Matching",
        "Predicted Matching"
    ]
)

print(confusion_table.to_string())

print("\nTABLE 4: CLASSIFICATION REPORT")

report = classification_report(
    y,
    oof_predictions,
    labels=[0, 1],
    target_names=[
        "Not matching",
        "Matching"
    ],
    output_dict=True,
    zero_division=0
)

report_table = pd.DataFrame(report).transpose()

print(report_table.round(4).to_string())


# ============================================================
# 11. SAVE EVALUATION FILES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

# Save fold results.
fold_table.to_csv(
    "experiments/logistic_regression_fold_results.csv",
    index=False
)

# Save out-of-fold predictions.
predictions_table = pd.DataFrame({
    "trip_id": data[GROUP_COLUMN].values,
    "observation_id": data[OBSERVATION_COLUMN].values,
    "actual": y.values,
    "predicted": oof_predictions,
    "matching_probability": oof_probabilities
})

predictions_table.to_csv(
    "experiments/logistic_regression_oof_predictions.csv",
    index=False
)

# Save all metrics.
metrics_to_save = {
    "macro_f1_score": macro_f1,
    "top_1_accuracy": top1_accuracy,
    "log_loss": logloss,
    "candidate_miss_rate": candidate_miss_rate,
    "overall_accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "total_inference_time_seconds": total_inference_time,
    "average_time_per_candidate_ms": average_time_per_candidate_ms,
    "average_time_per_1000_candidates_ms": average_time_per_1000_candidates_ms
}

pd.DataFrame([metrics_to_save]).to_csv(
    "experiments/logistic_regression_metrics.csv",
    index=False
)


# ============================================================
# 12. TRAIN FINAL MODEL ON ALL 80 TRAINING TRIPS
# ============================================================

print("\nTraining final Logistic Regression model...")

final_model = create_model()

final_model.fit(X, y)

joblib.dump(
    final_model,
    "models/logistic_regression_model.joblib"
)


# ============================================================
# 13. DISPLAY FINAL RESULT SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("                    FINAL RESULT SUMMARY")
print("=" * 75)

summary_table = pd.DataFrame([
    ["Algorithm", "Logistic Regression"],
    ["Training trips", str(total_trips)],
    ["Dataset rows", f"{len(data):,}"],
    ["GPS observations", f"{total_observations:,}"],
    ["Features used", str(len(FEATURES))],
    ["Validation method", "5-Fold GroupKFold"],
    ["Macro F1-score", f"{macro_f1 * 100:.2f}%"],
    ["Top-1 Accuracy", f"{top1_accuracy * 100:.2f}%"],
    ["Log Loss", f"{logloss:.4f}"],
    ["Candidate Miss Rate", f"{candidate_miss_rate * 100:.2f}%"],
    ["Overall Accuracy", f"{accuracy * 100:.2f}%"],
    ["Precision", f"{precision * 100:.2f}%"],
    ["Recall", f"{recall * 100:.2f}%"],
    ["Total Inference Time", f"{total_inference_time:.4f} seconds"],
    ["Model Status", "SUCCESSFULLY TRAINED"],
    ["Saved Model", "models/logistic_regression_model.joblib"]
], columns=[
    "Parameter",
    "Final Value"
])

print(summary_table.to_string(index=False))

print("\n" + "=" * 75)
print("             MODEL TRAINING COMPLETED")
print("=" * 75)

print("\nSaved evaluation files:")
print("1. experiments/logistic_regression_metrics.csv")
print("2. experiments/logistic_regression_fold_results.csv")
print("3. experiments/logistic_regression_oof_predictions.csv")

print("\nSaved model:")
print("models/logistic_regression_model.joblib")

print("\nAll tasks completed successfully!")
print("=" * 75)

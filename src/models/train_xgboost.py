"""Train and evaluate an XGBoost model for road-candidate ranking.

The input CSVs contain one row per GPS-observation/road-arc candidate. The
model learns whether each candidate is the aligned reference arc. At
prediction time, the candidate with the highest positive-class score can be
selected for each GPS observation.

This script only reads the already processed training-trip files under
``experiments/``. It keeps whole trips together during cross-validation so
that observations from one trip cannot leak into both sides of a fold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier


# Find the repository root from this file, so the script works no matter
# which folder the user is in when they run it.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TRAINING_DATA_DIR = PROJECT_ROOT / "Student folder" / "train"

# These columns are inputs to the model. IDs, aligned labels, and the target
# column are deliberately not included because they would leak answers.
BASE_FEATURE_COLUMNS = [
    "distance_m",
    "projection_position",
    "road_bearing",
    "gps_heading",
    "heading_difference",
    "gps_speed_mps",
    "time_delta_s",
    "large_or_invalid_time_gap",
]

# The final model uses sine/cosine versions of angles. This makes bearings
# near 0 degrees and 360 degrees look close to each other to the model.
MODEL_FEATURE_COLUMNS = [
    "distance_m",
    "projection_position",
    "road_bearing_sin",
    "road_bearing_cos",
    "gps_heading_sin",
    "gps_heading_cos",
    "heading_difference",
    "gps_speed_mps",
    "time_delta_s",
    "large_or_invalid_time_gap",
]

LABEL_COLUMN = "is_matching_candidate"
ALIGNMENT_LABEL_COLUMN = "aligned_arc_id"
GROUP_COLUMN = "trip_id"
RANDOM_SEED = 42
NUMBER_OF_FOLDS = 5
PREDICTION_THRESHOLD = 0.5

# A single configuration is used for every fold and for the final model.
# This is a starting point, not a Bayesian-optimized configuration.
MODEL_PARAMETERS = {
    # Number of boosting rounds (trees). More trees can learn more detail,
    # but can also overfit; later tuning should compare this value.
    "n_estimators": 300,
    # Limit each tree's depth to control model complexity.
    "max_depth": 6,
    # Smaller values make each tree's contribution more gradual.
    "learning_rate": 0.05,
    # Use a fraction of rows and columns per tree to reduce overfitting.
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    # Require enough evidence in a leaf before making a split.
    "min_child_weight": 3,
    # L2 regularization discourages overly large leaf scores.
    "reg_lambda": 2.0,
    # This is a binary problem: matching candidate (1) or not (0).
    "objective": "binary:logistic",
    # Log loss tracks how good the predicted probabilities are.
    "eval_metric": "logloss",
    # Histogram training is efficient for numeric tabular features.
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def load_processed_training_data() -> tuple[pd.DataFrame, dict[str, int]]:
    """Load labeled candidates and observation counts from matching alignments.

    Returns:
        candidates: All candidate rows from the processed training trips.
        total_observations: Number of aligned GPS observations per trip. This
            lets evaluation count observations for which candidate generation
            found no correct road segment.
    """
    candidate_files = sorted(
        EXPERIMENTS_DIR.glob("labelled_candidates_*.csv")
    )
    if not candidate_files:
        raise FileNotFoundError(
            "No labelled_candidates_*.csv files were found in "
            f"{EXPERIMENTS_DIR}. Run the preprocessing/labeling pipeline first."
        )

    candidate_frames: list[pd.DataFrame] = []
    total_observations: dict[str, int] = {}

    for candidate_file in candidate_files:
        # The trip ID is also stored in every row. We check that it matches
        # the filename so that rows cannot silently be assigned to the wrong trip.
        trip_id_from_name = candidate_file.stem.removeprefix(
            "labelled_candidates_"
        )
        alignment_file = EXPERIMENTS_DIR / f"alignment_{trip_id_from_name}.csv"

        if not alignment_file.is_file():
            raise FileNotFoundError(
                f"Missing matching alignment file: {alignment_file.name}"
            )

        # Read trip IDs as text so leading zeroes (for example, 00000001)
        # are preserved and can be matched to folder names exactly.
        candidates = pd.read_csv(candidate_file, dtype={GROUP_COLUMN: str})
        alignments = pd.read_csv(alignment_file)

        required_columns = {
            GROUP_COLUMN,
            "observation_id",
            "arc_id",
            *BASE_FEATURE_COLUMNS,
            ALIGNMENT_LABEL_COLUMN,
            LABEL_COLUMN,
        }
        missing_columns = required_columns.difference(candidates.columns)
        if missing_columns:
            raise ValueError(
                f"{candidate_file.name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if candidates.empty:
            raise ValueError(f"{candidate_file.name} contains no candidate rows.")
        if set(candidates[GROUP_COLUMN].astype(str)) != {trip_id_from_name}:
            raise ValueError(
                f"Trip IDs in {candidate_file.name} do not match its filename."
            )

        # Only data under the training partition is allowed for model fitting.
        # This prevents student-test or evaluator labels from entering training.
        training_trip_dir = TRAINING_DATA_DIR / trip_id_from_name
        if not training_trip_dir.is_dir():
            raise ValueError(
                f"{trip_id_from_name} is not a trip in the training partition. "
                "Do not train on student-test or evaluator-test trips."
            )

        # Check that the label file still agrees with the alignment output.
        # These two columns are only used for this check; neither is a model input.
        expected_labels = (
            candidates["arc_id"].to_numpy()
            == candidates[ALIGNMENT_LABEL_COLUMN].to_numpy()
        ).astype("int8")
        actual_labels = candidates[LABEL_COLUMN].to_numpy(dtype="int8")
        if not np.array_equal(expected_labels, actual_labels):
            raise ValueError(
                f"Labels in {candidate_file.name} do not agree with arc alignment."
            )

        if not {"observation_id", "aligned_arc_id"}.issubset(alignments.columns):
            raise ValueError(
                f"{alignment_file.name} must contain observation_id and aligned_arc_id."
            )
        alignment_ids = alignments["observation_id"].to_numpy(dtype=int)
        if len(alignment_ids) and not np.array_equal(
            np.sort(alignment_ids), np.arange(len(alignment_ids))
        ):
            raise ValueError(
                f"{alignment_file.name} must contain each observation ID once, "
                "from zero through the last GPS observation."
            )

        # Each GPS observation has one aligned reference arc, so it should
        # create no more than one positive candidate row.
        positive_counts = candidates.groupby("observation_id")[LABEL_COLUMN].sum()
        if (positive_counts > 1).any():
            raise ValueError(
                f"{candidate_file.name} has more than one positive candidate "
                "for a GPS observation."
            )

        # The alignment has one row per GPS observation, including observations
        # for which candidate generation found no matching road segment.
        total_observations[trip_id_from_name] = len(alignments)
        candidate_frames.append(candidates)

    all_candidates = pd.concat(candidate_frames, ignore_index=True)
    all_candidates[GROUP_COLUMN] = all_candidates[GROUP_COLUMN].astype(str)
    all_candidates[LABEL_COLUMN] = all_candidates[LABEL_COLUMN].astype("int8")

    if all_candidates[LABEL_COLUMN].nunique() != 2:
        raise ValueError(
            "The processed files must contain both positive and negative "
            "candidate labels to train a binary classifier."
        )

    return all_candidates, total_observations


def make_model_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Create safe numeric model inputs from the processed candidate columns."""
    features = candidates[BASE_FEATURE_COLUMNS].copy()

    # Convert degrees to radians, then encode each direction with sine/cosine.
    # This avoids treating 359 degrees and 1 degree as far apart.
    for column in ("road_bearing", "gps_heading"):
        radians = np.deg2rad(features[column])
        features[f"{column}_sin"] = np.sin(radians)
        features[f"{column}_cos"] = np.cos(radians)

    # The two raw angle columns have been replaced by their circular encoding.
    features = features.drop(columns=["road_bearing", "gps_heading"])

    # XGBoost can use NaN values to learn how missing feature values behave.
    # Replace infinities with NaN as well, so bad values do not break training.
    features = features.replace([np.inf, -np.inf], np.nan)

    # Keep feature order fixed so training and later inference use the same order.
    return features[MODEL_FEATURE_COLUMNS]


def score_fold(
    validation_rows: pd.DataFrame,
    probabilities: np.ndarray,
    total_observations: dict[str, int],
) -> dict[str, float | int]:
    """Measure candidate classification and per-observation route selection."""
    labels = validation_rows[LABEL_COLUMN].to_numpy(dtype="int8")
    predicted_labels = (probabilities >= PREDICTION_THRESHOLD).astype("int8")

    # Candidate-level metrics treat every observation/road-arc pair as a row.
    candidate_macro_f1 = f1_score(
        labels, predicted_labels, average="macro", zero_division=0
    )
    candidate_log_loss = log_loss(labels, probabilities, labels=[0, 1])

    # For the route task, choose the highest-scoring candidate for each GPS point.
    ranked_rows = validation_rows[[GROUP_COLUMN, "observation_id"]].copy()
    ranked_rows["probability"] = probabilities
    ranked_rows[LABEL_COLUMN] = labels
    best_row_indices = ranked_rows.groupby(
        [GROUP_COLUMN, "observation_id"], sort=False
    )["probability"].idxmax()
    selected_candidates = ranked_rows.loc[best_row_indices]

    # Count every aligned GPS observation, even if candidate generation missed
    # its correct road segment. Such misses cannot be repaired by the classifier.
    validation_trip_ids = validation_rows[GROUP_COLUMN].unique()
    aligned_observation_count = sum(
        total_observations[trip_id] for trip_id in validation_trip_ids
    )
    observations_with_positive_candidate = int(
        selected_candidates[LABEL_COLUMN].sum()
    )
    observation_count_with_candidates = len(selected_candidates)
    candidate_misses = (
        aligned_observation_count - observations_with_positive_candidate
    )
    correctly_selected = observations_with_positive_candidate

    # This stricter accuracy counts a reference observation with no matching
    # candidate as incorrect, instead of silently dropping it from evaluation.
    end_to_end_top1_accuracy = (
        correctly_selected / aligned_observation_count
        if aligned_observation_count
        else 0.0
    )
    top1_accuracy_when_candidate_exists = (
        correctly_selected / observation_count_with_candidates
        if observation_count_with_candidates
        else 0.0
    )

    return {
        "candidate_rows": len(validation_rows),
        "aligned_observations": aligned_observation_count,
        "observations_with_candidates": observation_count_with_candidates,
        "candidate_misses": candidate_misses,
        "candidate_miss_rate": (
            candidate_misses / aligned_observation_count
            if aligned_observation_count
            else 0.0
        ),
        "candidate_macro_f1": float(candidate_macro_f1),
        "candidate_log_loss": float(candidate_log_loss),
        "top1_accuracy_when_candidate_exists": float(
            top1_accuracy_when_candidate_exists
        ),
        "end_to_end_top1_accuracy": float(end_to_end_top1_accuracy),
    }


def build_model() -> XGBClassifier:
    """Create one clear baseline XGBoost classifier configuration."""
    return XGBClassifier(**MODEL_PARAMETERS)


def run_grouped_cross_validation(
    candidates: pd.DataFrame,
    total_observations: dict[str, int],
) -> pd.DataFrame:
    """Evaluate the model while keeping each trip entirely in one fold."""
    unique_trip_count = candidates[GROUP_COLUMN].nunique()
    number_of_folds = min(NUMBER_OF_FOLDS, unique_trip_count)
    if number_of_folds < 2:
        raise ValueError("At least two different trips are needed for validation.")

    features = make_model_features(candidates)
    labels = candidates[LABEL_COLUMN].to_numpy(dtype="int8")
    groups = candidates[GROUP_COLUMN].to_numpy()
    group_cv = GroupKFold(n_splits=number_of_folds)
    fold_results: list[dict] = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        group_cv.split(features, labels, groups),
        start=1,
    ):
        # GroupKFold ensures every trip is either fully in training or fully
        # in validation for this fold.
        training_trip_ids = sorted(set(groups[train_indices]))
        validation_trip_ids = sorted(set(groups[validation_indices]))

        fold_model = build_model()
        fold_model.fit(
            features.iloc[train_indices],
            labels[train_indices],
            verbose=False,
        )

        probabilities = fold_model.predict_proba(
            features.iloc[validation_indices]
        )[:, 1]
        metrics = score_fold(
            candidates.iloc[validation_indices],
            probabilities,
            total_observations,
        )

        fold_results.append({
            "fold": fold_number,
            "training_trips": ",".join(training_trip_ids),
            "validation_trips": ",".join(validation_trip_ids),
            **metrics,
        })

    return pd.DataFrame(fold_results)


def save_final_model(
    candidates: pd.DataFrame,
    cv_results: pd.DataFrame,
    total_observations: dict[str, int],
) -> None:
    """Fit on all currently processed training trips and save model artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # After grouped validation is recorded, train the final model on all
    # available processed training-trip candidate rows.
    final_model = build_model()
    final_model.fit(
        make_model_features(candidates),
        candidates[LABEL_COLUMN].to_numpy(dtype="int8"),
        verbose=False,
    )

    model_path = ARTIFACTS_DIR / "xgboost_candidate_model.json"
    final_model.save_model(model_path)

    # Store the feature order and settings beside the model. Inference code
    # needs these same feature names and transformations to score new trips.
    metadata = {
        "model_type": "XGBClassifier",
        "target": LABEL_COLUMN,
        "positive_class_meaning": "candidate matches aligned reference arc",
        "input_files": [
            path.name
            for path in sorted(EXPERIMENTS_DIR.glob("labelled_candidates_*.csv"))
        ],
        "training_trip_ids": sorted(candidates[GROUP_COLUMN].unique().tolist()),
        "base_feature_columns": BASE_FEATURE_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "prediction_threshold_for_binary_metrics": PREDICTION_THRESHOLD,
        "number_of_group_folds": len(cv_results),
        "model_parameters": MODEL_PARAMETERS,
        "aligned_observations_by_trip": total_observations,
        "note": (
            "This model was trained only on the processed training-trip files "
            "available when the script ran. It has not been evaluated on "
            "student_test or evaluator_test data."
        ),
    }
    metadata_path = ARTIFACTS_DIR / "xgboost_candidate_model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    """Run data checks, grouped validation, and final model training."""
    print("Loading labeled candidate files...")
    candidates, total_observations = load_processed_training_data()

    print(f"Processed trips: {candidates[GROUP_COLUMN].nunique()}")
    print(f"Candidate rows: {len(candidates):,}")
    print(f"Matching candidate rows: {int(candidates[LABEL_COLUMN].sum()):,}")
    print(
        "Non-matching candidate rows: "
        f"{int((candidates[LABEL_COLUMN] == 0).sum()):,}"
    )
    print(
        "Warning: candidate misses are measured separately. A classifier "
        "cannot select a correct road segment if it was not generated."
    )

    # Summarize candidate coverage before training. This is a data-generation
    # diagnostic, not a model score.
    coverage_rows = []
    for trip_id, aligned_count in sorted(total_observations.items()):
        trip_candidates = candidates[candidates[GROUP_COLUMN] == trip_id]
        observations_with_candidates = int(
            trip_candidates["observation_id"].nunique()
        )
        observations_with_correct_candidate = int(
            trip_candidates.loc[
                trip_candidates[LABEL_COLUMN] == 1, "observation_id"
            ].nunique()
        )
        misses = aligned_count - observations_with_correct_candidate
        coverage_rows.append({
            "trip_id": trip_id,
            "aligned_observations": aligned_count,
            "observations_with_candidates": observations_with_candidates,
            "observations_with_correct_candidate": (
                observations_with_correct_candidate
            ),
            "candidate_misses": misses,
            "candidate_miss_rate": misses / aligned_count if aligned_count else 0.0,
        })

    coverage_path = EXPERIMENTS_DIR / "xgboost_candidate_coverage.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
    print("\nCandidate coverage before classification (data-quality measure):")
    print(pd.DataFrame(coverage_rows).to_string(index=False))
    print(f"Saved coverage report to: {coverage_path.relative_to(PROJECT_ROOT)}")

    print("\nRunning trip-grouped cross-validation...")
    cv_results = run_grouped_cross_validation(candidates, total_observations)
    results_path = EXPERIMENTS_DIR / "xgboost_group_cv.csv"
    cv_results.to_csv(results_path, index=False)
    print(cv_results.to_string(index=False))
    print(f"\nSaved fold results to: {results_path.relative_to(PROJECT_ROOT)}")

    # This summary makes the variation between held-out trips visible.
    numeric_metrics = cv_results.select_dtypes(include=[np.number]).drop(
        columns=["fold"], errors="ignore"
    )
    summary = pd.DataFrame({
        "mean": numeric_metrics.mean(),
        "standard_deviation": numeric_metrics.std(ddof=1).fillna(0.0),
    })
    summary_path = EXPERIMENTS_DIR / "xgboost_group_cv_summary.csv"
    summary.to_csv(summary_path, index_label="metric")
    print(f"Saved metric summary to: {summary_path.relative_to(PROJECT_ROOT)}")

    print("\nTraining final model on all available processed training trips...")
    save_final_model(candidates, cv_results, total_observations)
    print(
        "Saved model and metadata under: "
        f"{ARTIFACTS_DIR.relative_to(PROJECT_ROOT)}"
    )
    print(
        "This run covers only the processed trips currently in experiments/. "
        "Run preprocessing for the remaining training trips before treating "
        "these scores as final model evidence."
    )


if __name__ == "__main__":
    main()

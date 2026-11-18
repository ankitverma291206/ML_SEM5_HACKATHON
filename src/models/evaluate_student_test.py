"""Student-test evaluation script for Atharv Kulkarni's ML workstream.

Loads the 10 student test trips from 'Student folder/student_test', generates
candidate features and ground-truth sequence-aware labels, evaluates all three trained
models (Random Forest, KNN, Decision Tree) on the student test set, and calculates:
- Candidate-level Accuracy, Precision, Recall, Macro F1, Log Loss
- Observation-level Top-1 Accuracy
- Candidate Miss Rate
- Generalization gap (Student Test vs. 5-Fold CV)
"""

from pathlib import Path
from typing import Tuple
import json
import joblib
import numpy as np
import pandas as pd

from src.data.load_trip import load_trip
from src.features.candidates import generate_candidates
from src.features.labels import align_track_to_route, label_candidates
from src.models.config import FeatureConfig
from src.models.evaluate import (
    compute_candidate_metrics,
    compute_observation_top1_accuracy,
    compute_candidate_miss_rate,
)


def process_student_test_trips() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process all 10 student test trips into candidates and inventory."""
    test_dir = Path("Student folder/student_test")
    trip_dirs = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    print(f"Found {len(trip_dirs)} student test trips: {trip_dirs}")

    all_candidates = []
    all_inventories = []

    for trip_id in trip_dirs:
        data = load_trip(test_dir / trip_id, trip_id)
        n_obs = len(data["track"])

        # Inventory
        trip_inv = pd.DataFrame({
            "trip_id": trip_id,
            "observation_id": np.arange(n_obs),
        })
        all_inventories.append(trip_inv)

        # Candidates & alignment
        cands = generate_candidates(data, trip_id=trip_id, radius_m=50.0, max_candidates=8)
        align = align_track_to_route(data)
        lab = label_candidates(cands, align)

        if "large_or_invalid_time_gap" in lab.columns:
            lab["large_or_invalid_time_gap"] = lab["large_or_invalid_time_gap"].astype("int8")

        all_candidates.append(lab)

    df_test_cands = pd.concat(all_candidates, ignore_index=True)
    df_test_inv = pd.concat(all_inventories, ignore_index=True)

    return df_test_cands, df_test_inv


def evaluate_on_student_test():
    test_cands_path = Path("data/processed/test_candidates.parquet")
    test_inv_path = Path("data/processed/test_observation_inventory.parquet")

    if not test_cands_path.exists() or not test_inv_path.exists():
        print("Processing student test trips...")
        df_test_cands, df_test_inv = process_student_test_trips()
        test_cands_path.parent.mkdir(parents=True, exist_ok=True)
        df_test_cands.to_parquet(test_cands_path, index=False)
        df_test_inv.to_parquet(test_inv_path, index=False)
    else:
        print("Loading cached student test candidates...")
        df_test_cands = pd.read_parquet(test_cands_path)
        df_test_inv = pd.read_parquet(test_inv_path)

    print(f"Student test set: {len(df_test_cands):,} candidate rows, {len(df_test_inv):,} total observations across {df_test_cands['trip_id'].nunique()} trips.")

    config = FeatureConfig()
    config.validate_dataframe(df_test_cands, require_label=True)

    X_test = df_test_cands[config.feature_columns]
    y_test = df_test_cands[config.label_column].to_numpy()

    # Load artifacts
    models = {
        "Random Forest": joblib.load("artifacts/atharv_random_forest.pkl"),
        "K-Nearest Neighbours": joblib.load("artifacts/atharv_knn.pkl"),
        "Decision Tree": joblib.load("artifacts/atharv_decision_tree.pkl"),
    }

    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name} on student test set...")
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        df_eval = df_test_cands.copy()
        df_eval["prob_matching"] = probs
        df_eval["pred_matching"] = preds

        cls_metrics = compute_candidate_metrics(y_test, preds, probs)
        top1_acc = compute_observation_top1_accuracy(
            df=df_eval,
            prob_col="prob_matching",
            label_col="is_matching_candidate",
            all_observations=df_test_inv,
        )
        miss_rate = compute_candidate_miss_rate(
            df=df_eval,
            label_col="is_matching_candidate",
            all_observations=df_test_inv,
        )

        row = {
            "model_name": name,
            "test_accuracy": cls_metrics["accuracy"],
            "test_precision": cls_metrics["precision"],
            "test_recall": cls_metrics["recall"],
            "test_macro_f1": cls_metrics["macro_f1"],
            "test_top1_accuracy": top1_acc,
            "test_candidate_miss_rate": miss_rate,
            "test_log_loss": cls_metrics.get("log_loss"),
        }
        results.append(row)
        print(f"  Accuracy: {row['test_accuracy']*100:.2f}%")
        print(f"  Macro F1: {row['test_macro_f1']:.4f}")
        print(f"  Top-1 Accuracy: {row['test_top1_accuracy']*100:.2f}%")
        print(f"  Candidate Miss Rate: {row['test_candidate_miss_rate']*100:.2f}%")

    df_res = pd.DataFrame(results)
    df_res.to_csv("experiments/atharv_student_test_results.csv", index=False)
    print("\nSaved student test results to experiments/atharv_student_test_results.csv")


if __name__ == "__main__":
    evaluate_on_student_test()

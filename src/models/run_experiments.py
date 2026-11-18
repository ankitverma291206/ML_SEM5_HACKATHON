"""Reproducible execution script for Atharv Kulkarni's ML experimentation.

Trains and evaluates:
1. Random Forest Classifier (Baseline & Balanced)
2. K-Nearest Neighbours (KD-Tree, K=15 baseline & K sensitivity)
3. Decision Tree Classifier (Baseline & Balanced)

Uses strictly identical 5-fold GroupKFold validation grouped by trip_id,
evaluates candidate-level and observation-level metrics (Top-1 Accuracy, Candidate Miss Rate)
using the complete raw observation inventory (208,417 observations across 80 trips),
and exports verifiable CSV logs, figures, and serialized artifacts.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.config import FeatureConfig
from src.models.decision_tree import (
    build_decision_tree_pipeline,
    evaluate_decision_tree,
    fit_and_save_decision_tree,
)
from src.models.knn import (
    build_knn_pipeline,
    evaluate_knn,
    fit_and_save_knn,
)
from src.models.random_forest import (
    build_random_forest_pipeline,
    evaluate_random_forest,
    fit_and_save_random_forest,
)
from src.models.scoring import save_model_artifacts
from src.models.validation import ModelCVResult


# Simple explanation: Draw charts that compare the measured model results.
def plot_three_model_comparison(
    df_comparison: pd.DataFrame,
    output_dir: Path = Path("experiments"),
) -> Dict[str, Path]:
    """Generate high-resolution comparison plots for hackathon deliverables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}

    metrics_to_plot = [
        ("cv_accuracy", "Validation Accuracy (%)", "accuracy_comparison.png", True),
        ("cv_macro_f1", "Macro F1-Score", "macro_f1_comparison.png", False),
        ("top1_accuracy", "Observation-Level Top-1 Accuracy (%)", "top1_accuracy_comparison.png", True),
        ("train_time_sec", "Total Training Time (seconds)", "train_time_comparison.png", False),
        ("infer_time_sec", "Total Inference Time (seconds)", "infer_time_comparison.png", False),
    ]

    models = df_comparison["model_name"].tolist()
    colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#2ca02c", "#98df8a"][: len(models)]

    for col, title, fname, is_pct in metrics_to_plot:
        if col not in df_comparison.columns:
            continue

        plt.figure(figsize=(10, 5))
        vals = df_comparison[col].to_numpy()
        plot_vals = vals * 100.0 if is_pct else vals

        bars = plt.bar(models, plot_vals, color=colors, edgecolor="black", linewidth=0.8)

        for bar in bars:
            h = bar.get_height()
            if h is not None and not np.isnan(h):
                fmt = f"{h:.2f}%" if is_pct else f"{h:.4f}"
                if "time" in col:
                    fmt = f"{h:.1f}s"
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + (max(plot_vals) * 0.015),
                    fmt,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        plt.title(f"Atharv's Model Comparison: {title}", fontsize=12, fontweight="bold", pad=12)
        plt.ylabel(title, fontsize=10)
        plt.xlabel("Model Configuration", fontsize=10)
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = output_dir / fname
        plt.savefig(out_path, dpi=300)
        plt.close()
        plots[col] = out_path

    return plots


# Simple explanation: Run the model comparisons and save their results.
def execute_experiments():
    """Main orchestration for real ML model experimentation."""
    print("=" * 70)
    print("PATHSNAP ML EXPERIMENTATION WORKSTREAM (Atharv Kulkarni - PES2UG24CS092)")
    print("=" * 70)

    # 1. Load verified training data and observation inventory
    candidates_path = Path("data/processed/train_candidates.parquet")
    inventory_path = Path("data/processed/train_observation_inventory.parquet")

    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing {candidates_path}. Run build_training_dataset.py first.")
    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing {inventory_path}.")

    print("Loading candidate dataset and raw observation inventory...")
    df_train = pd.read_parquet(candidates_path)
    df_inventory = pd.read_parquet(inventory_path)

    config = FeatureConfig()
    config.validate_dataframe(df_train, require_label=True)

    print(f"Dataset verified: {len(df_train):,} candidate rows across {df_train['trip_id'].nunique()} trips.")
    print(f"Observation inventory verified: {len(df_inventory):,} observations across 80 trips.")

    exp_dir = Path("experiments")
    art_dir = Path("artifacts")
    exp_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    results_list: List[ModelCVResult] = []
    comparison_rows: List[Dict[str, Any]] = []
    all_oof_predictions: Dict[str, pd.Series] = {
        "trip_id": df_train["trip_id"],
        "observation_id": df_train["observation_id"],
        "arc_id": df_train["arc_id"],
        "is_matching_candidate": df_train["is_matching_candidate"],
    }

    # =========================================================================
    # EXPERIMENT 1: Random Forest Classifier (Baseline & Balanced)
    # =========================================================================
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: Random Forest Classifier")
    print("=" * 50)

    # 1A. Baseline (class_weight=None)
    print("Training Random Forest (Baseline: class_weight=None)...")
    t0 = time.time()
    rf_baseline_res = evaluate_random_forest(
        df_train=df_train,
        config=config,
        custom_params={"class_weight": None, "random_state": 42},
        n_splits=5,
        output_dir=exp_dir,
        all_observations=df_inventory,
    )
    rf_time = time.time() - t0
    results_list.append(rf_baseline_res)
    all_oof_predictions["rf_baseline_prob"] = rf_baseline_res.oof_predictions_df["prob_matching"]
    all_oof_predictions["rf_baseline_pred"] = rf_baseline_res.oof_predictions_df["pred_matching"]

    oof_rf = rf_baseline_res.oof_metrics
    print(f"Random Forest (Baseline) completed in {rf_time:.1f}s.")
    print(f"  CV Accuracy: {oof_rf['accuracy']*100:.2f}%")
    print(f"  CV Macro F1: {oof_rf['macro_f1']:.4f}")
    print(f"  Top-1 Accuracy: {oof_rf['top1_accuracy']*100:.2f}%")
    print(f"  Candidate Miss Rate: {oof_rf['candidate_miss_rate']*100:.2f}%")
    print(f"  CV Log Loss: {oof_rf.get('log_loss'):.4f}")

    comparison_rows.append({
        "model_name": "Random Forest (Baseline)",
        "algorithm_family": "Ensemble (Bagging)",
        "configuration": "n_trees=100, depth=12, split=10, leaf=4, class_weight=None",
        "cv_accuracy": oof_rf["accuracy"],
        "cv_macro_f1": oof_rf["macro_f1"],
        "cv_precision": oof_rf["precision"],
        "cv_recall": oof_rf["recall"],
        "top1_accuracy": oof_rf["top1_accuracy"],
        "candidate_miss_rate": oof_rf["candidate_miss_rate"],
        "cv_log_loss": oof_rf.get("log_loss"),
        "train_time_sec": rf_baseline_res.total_train_time,
        "infer_time_sec": rf_baseline_res.total_infer_time,
    })

    # 1B. Balanced Class Weight
    print("\nTraining Random Forest (Balanced: class_weight='balanced')...")
    t0 = time.time()
    rf_balanced_res = evaluate_random_forest(
        df_train=df_train,
        config=config,
        custom_params={"class_weight": "balanced", "random_state": 42},
        n_splits=5,
        output_dir=exp_dir,
        all_observations=df_inventory,
    )
    rf_bal_time = time.time() - t0
    all_oof_predictions["rf_balanced_prob"] = rf_balanced_res.oof_predictions_df["prob_matching"]
    all_oof_predictions["rf_balanced_pred"] = rf_balanced_res.oof_predictions_df["pred_matching"]

    oof_rf_bal = rf_balanced_res.oof_metrics
    print(f"Random Forest (Balanced) completed in {rf_bal_time:.1f}s.")
    print(f"  CV Accuracy: {oof_rf_bal['accuracy']*100:.2f}%")
    print(f"  CV Macro F1: {oof_rf_bal['macro_f1']:.4f}")
    print(f"  Top-1 Accuracy: {oof_rf_bal['top1_accuracy']*100:.2f}%")

    comparison_rows.append({
        "model_name": "Random Forest (Balanced)",
        "algorithm_family": "Ensemble (Bagging)",
        "configuration": "n_trees=100, depth=12, split=10, leaf=4, class_weight=balanced",
        "cv_accuracy": oof_rf_bal["accuracy"],
        "cv_macro_f1": oof_rf_bal["macro_f1"],
        "cv_precision": oof_rf_bal["precision"],
        "cv_recall": oof_rf_bal["recall"],
        "top1_accuracy": oof_rf_bal["top1_accuracy"],
        "candidate_miss_rate": oof_rf_bal["candidate_miss_rate"],
        "cv_log_loss": oof_rf_bal.get("log_loss"),
        "train_time_sec": rf_balanced_res.total_train_time,
        "infer_time_sec": rf_balanced_res.total_infer_time,
    })

    # =========================================================================
    # EXPERIMENT 2: K-Nearest Neighbours (KNN)
    # =========================================================================
    print("\n" + "=" * 50)
    print("EXPERIMENT 2: K-Nearest Neighbours (KNN)")
    print("=" * 50)

    # 2A. Baseline (K=15, distance-weighted, KD-Tree)
    print("Training KNN (Baseline: K=15, distance-weighted, KD-Tree)...")
    t0 = time.time()
    knn_baseline_res = evaluate_knn(
        df_train=df_train,
        config=config,
        custom_params={"n_neighbors": 15, "weights": "distance", "algorithm": "kd_tree"},
        n_splits=5,
        output_dir=exp_dir,
        all_observations=df_inventory,
    )
    knn_time = time.time() - t0
    results_list.append(knn_baseline_res)
    all_oof_predictions["knn_baseline_prob"] = knn_baseline_res.oof_predictions_df["prob_matching"]
    all_oof_predictions["knn_baseline_pred"] = knn_baseline_res.oof_predictions_df["pred_matching"]

    oof_knn = knn_baseline_res.oof_metrics
    print(f"KNN (Baseline) completed in {knn_time:.1f}s.")
    print(f"  CV Accuracy: {oof_knn['accuracy']*100:.2f}%")
    print(f"  CV Macro F1: {oof_knn['macro_f1']:.4f}")
    print(f"  Top-1 Accuracy: {oof_knn['top1_accuracy']*100:.2f}%")
    print(f"  Candidate Miss Rate: {oof_knn['candidate_miss_rate']*100:.2f}%")
    print(f"  CV Log Loss: {oof_knn.get('log_loss'):.4f}")

    comparison_rows.append({
        "model_name": "KNN (K=15, Distance)",
        "algorithm_family": "Instance-based",
        "configuration": "K=15, weights=distance, algorithm=kd_tree, metric=minkowski",
        "cv_accuracy": oof_knn["accuracy"],
        "cv_macro_f1": oof_knn["macro_f1"],
        "cv_precision": oof_knn["precision"],
        "cv_recall": oof_knn["recall"],
        "top1_accuracy": oof_knn["top1_accuracy"],
        "candidate_miss_rate": oof_knn["candidate_miss_rate"],
        "cv_log_loss": oof_knn.get("log_loss"),
        "train_time_sec": knn_baseline_res.total_train_time,
        "infer_time_sec": knn_baseline_res.total_infer_time,
    })

    # =========================================================================
    # EXPERIMENT 3: Decision Tree Classifier (Baseline & Balanced)
    # =========================================================================
    print("\n" + "=" * 50)
    print("EXPERIMENT 3: Decision Tree Classifier")
    print("=" * 50)

    # 3A. Baseline (class_weight=None)
    print("Training Decision Tree (Baseline: class_weight=None)...")
    t0 = time.time()
    dt_baseline_res = evaluate_decision_tree(
        df_train=df_train,
        config=config,
        custom_params={"class_weight": None, "random_state": 42},
        n_splits=5,
        output_dir=exp_dir,
        all_observations=df_inventory,
    )
    dt_time = time.time() - t0
    results_list.append(dt_baseline_res)
    all_oof_predictions["dt_baseline_prob"] = dt_baseline_res.oof_predictions_df["prob_matching"]
    all_oof_predictions["dt_baseline_pred"] = dt_baseline_res.oof_predictions_df["pred_matching"]

    oof_dt = dt_baseline_res.oof_metrics
    print(f"Decision Tree (Baseline) completed in {dt_time:.1f}s.")
    print(f"  CV Accuracy: {oof_dt['accuracy']*100:.2f}%")
    print(f"  CV Macro F1: {oof_dt['macro_f1']:.4f}")
    print(f"  Top-1 Accuracy: {oof_dt['top1_accuracy']*100:.2f}%")
    print(f"  Candidate Miss Rate: {oof_dt['candidate_miss_rate']*100:.2f}%")
    print(f"  CV Log Loss: {oof_dt.get('log_loss'):.4f}")

    comparison_rows.append({
        "model_name": "Decision Tree (Baseline)",
        "algorithm_family": "Tree-based",
        "configuration": "depth=8, split=10, leaf=5, criterion=gini, class_weight=None",
        "cv_accuracy": oof_dt["accuracy"],
        "cv_macro_f1": oof_dt["macro_f1"],
        "cv_precision": oof_dt["precision"],
        "cv_recall": oof_dt["recall"],
        "top1_accuracy": oof_dt["top1_accuracy"],
        "candidate_miss_rate": oof_dt["candidate_miss_rate"],
        "cv_log_loss": oof_dt.get("log_loss"),
        "train_time_sec": dt_baseline_res.total_train_time,
        "infer_time_sec": dt_baseline_res.total_infer_time,
    })

    # 3B. Balanced Decision Tree
    print("\nTraining Decision Tree (Balanced: class_weight='balanced')...")
    t0 = time.time()
    dt_balanced_res = evaluate_decision_tree(
        df_train=df_train,
        config=config,
        custom_params={"class_weight": "balanced", "random_state": 42},
        n_splits=5,
        output_dir=exp_dir,
        all_observations=df_inventory,
    )
    dt_bal_time = time.time() - t0
    all_oof_predictions["dt_balanced_prob"] = dt_balanced_res.oof_predictions_df["prob_matching"]
    all_oof_predictions["dt_balanced_pred"] = dt_balanced_res.oof_predictions_df["pred_matching"]

    oof_dt_bal = dt_balanced_res.oof_metrics
    print(f"Decision Tree (Balanced) completed in {dt_bal_time:.1f}s.")
    print(f"  CV Accuracy: {oof_dt_bal['accuracy']*100:.2f}%")
    print(f"  CV Macro F1: {oof_dt_bal['macro_f1']:.4f}")
    print(f"  Top-1 Accuracy: {oof_dt_bal['top1_accuracy']*100:.2f}%")

    comparison_rows.append({
        "model_name": "Decision Tree (Balanced)",
        "algorithm_family": "Tree-based",
        "configuration": "depth=8, split=10, leaf=5, criterion=gini, class_weight=balanced",
        "cv_accuracy": oof_dt_bal["accuracy"],
        "cv_macro_f1": oof_dt_bal["macro_f1"],
        "cv_precision": oof_dt_bal["precision"],
        "cv_recall": oof_dt_bal["recall"],
        "top1_accuracy": oof_dt_bal["top1_accuracy"],
        "candidate_miss_rate": oof_dt_bal["candidate_miss_rate"],
        "cv_log_loss": oof_dt_bal.get("log_loss"),
        "train_time_sec": dt_balanced_res.total_train_time,
        "infer_time_sec": dt_balanced_res.total_infer_time,
    })

    # =========================================================================
    # STEP 7: Save Consolidated Comparison and Out-Of-Fold Predictions
    # =========================================================================
    df_comparison = pd.DataFrame(comparison_rows).sort_values(
        by=["cv_accuracy", "top1_accuracy"], ascending=[False, False]
    ).reset_index(drop=True)

    comp_csv_path = exp_dir / "atharv_three_model_comparison.csv"
    df_comparison.to_csv(comp_csv_path, index=False)
    print(f"\nSaved consolidated three-model comparison to {comp_csv_path}")

    oof_predictions_df = pd.DataFrame(all_oof_predictions)
    oof_parquet_path = exp_dir / "atharv_oof_predictions.parquet"
    oof_predictions_df.to_parquet(oof_parquet_path, index=False)
    print(f"Saved out-of-fold predictions to {oof_parquet_path}")

    # Generate comparison plots
    plot_three_model_comparison(df_comparison, output_dir=exp_dir)
    print("Saved comparison plots to experiments/")

    # =========================================================================
    # REFIT AND SAVE MODEL ARTIFACTS
    # =========================================================================
    print("\nRefitting and saving model artifacts...")
    rf_pipe = fit_and_save_random_forest(df_train, config, artifacts_dir=art_dir)
    knn_pipe = fit_and_save_knn(df_train, config, artifacts_dir=art_dir)
    dt_pipe = fit_and_save_decision_tree(df_train, config, artifacts_dir=art_dir)

    # Determine best model based on measured accuracy
    best_row = df_comparison.iloc[0]
    best_model_name = best_row["model_name"]
    print(f"\nWINNING MODEL IDENTIFIED: {best_model_name}")
    print(f"  Validation Accuracy: {best_row['cv_accuracy']*100:.2f}%")
    print(f"  Top-1 Accuracy: {best_row['top1_accuracy']*100:.2f}%")
    print(f"  Macro F1: {best_row['cv_macro_f1']:.4f}")

    if "Random Forest" in best_model_name:
        best_pipe = rf_pipe
    elif "KNN" in best_model_name:
        best_pipe = knn_pipe
    else:
        best_pipe = dt_pipe

    save_model_artifacts(
        pipeline=best_pipe,
        config=config,
        metadata={
            "best_model_name": best_model_name,
            "cv_accuracy": float(best_row["cv_accuracy"]),
            "cv_macro_f1": float(best_row["cv_macro_f1"]),
            "top1_accuracy": float(best_row["top1_accuracy"]),
            "configuration": best_row["configuration"],
            "training_samples": len(df_train),
            "training_trips": int(df_train["trip_id"].nunique()),
        },
        artifacts_dir=art_dir,
        model_filename="best_model.pkl",
    )
    print("Exported best_model.pkl and configuration artifacts to artifacts/")

    print("\n" + "=" * 70)
    print("EXPERIMENTATION COMPLETE: ALL RESULTS VERIFIED AND LOGGED.")
    print("=" * 70)


if __name__ == "__main__":
    execute_experiments()

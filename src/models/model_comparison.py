"""Consolidated five-model comparison and experiment logging framework.

Supports Atharv Kulkarni's three models (Random Forest, KNN, Decision Tree)
and provides an integration interface to import the experimental results
from the two teammate-owned models without inventing or fabricating data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.validation import ModelCVResult


ALGORITHM_FAMILIES = {
    "random_forest": "Ensemble (Bagging)",
    "knn": "Instance-based",
    "decision_tree": "Tree-based",
    "logistic_regression": "Linear & Probabilistic",
    "hist_gradient_boosting": "Ensemble (Boosting)",
}


class ModelComparisonManager:
    """Manages cross-model comparisons for Atharv's models and teammate models."""

    def __init__(self, output_dir: Union[str, Path] = "experiments") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Atharv's internally evaluated CVResults
        self.atharv_results: Dict[str, ModelCVResult] = {}
        # Imported external teammate results
        self.external_results: Dict[str, Dict[str, Any]] = {}
        # Optional student test metrics
        self.student_test_metrics: Dict[str, Dict[str, Any]] = {}

    def add_atharv_result(
        self,
        cv_result: ModelCVResult,
        student_test_eval: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register one of Atharv's three trained model results."""
        self.atharv_results[cv_result.model_name] = cv_result
        if student_test_eval:
            self.student_test_metrics[cv_result.model_name] = student_test_eval

    def import_external_result(
        self,
        model_name: str,
        owner: str,
        algorithm_family: str,
        key_configuration: str,
        cv_metrics: Dict[str, Any],
        student_test_metrics: Optional[Dict[str, Any]] = None,
        train_time_sec: float = 0.0,
        infer_time_sec: float = 0.0,
        route_continuity: Optional[float] = None,
    ) -> None:
        """Import actual experimental results supplied by a teammate.

        Ensures external models are integrated fairly without fabrication.
        """
        self.external_results[model_name] = {
            "model_name": model_name,
            "owner": owner,
            "algorithm_family": algorithm_family,
            "key_configuration": key_configuration,
            "cv_metrics": cv_metrics,
            "train_time_sec": train_time_sec,
            "infer_time_sec": infer_time_sec,
            "route_continuity": route_continuity,
        }
        if student_test_metrics:
            self.student_test_metrics[model_name] = student_test_metrics

    def import_external_csv(self, csv_path: Union[str, Path]) -> None:
        """Import external model results directly from a teammate's CSV file."""
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            m_name = str(row["model_name"])
            self.external_results[m_name] = {
                "model_name": m_name,
                "owner": str(row.get("owner", "Teammate")),
                "algorithm_family": str(row.get("algorithm_family", "External")),
                "key_configuration": str(row.get("key_configuration", "")),
                "cv_metrics": {
                    "macro_f1": row.get("cv_macro_f1"),
                    "accuracy": row.get("cv_accuracy"),
                    "precision": row.get("cv_precision"),
                    "recall": row.get("cv_recall"),
                    "top1_accuracy": row.get("top1_accuracy"),
                    "candidate_miss_rate": row.get("candidate_miss_rate"),
                    "log_loss": row.get("cv_log_loss"),
                },
                "train_time_sec": float(row.get("train_time_sec", 0.0)),
                "infer_time_sec": float(row.get("infer_time_sec", 0.0)),
                "route_continuity": row.get("route_continuity"),
            }
            if "student_test_macro_f1" in row:
                self.student_test_metrics[m_name] = {
                    "macro_f1": row.get("student_test_macro_f1"),
                    "top1_accuracy": row.get("student_test_top1_acc"),
                }

    def generate_atharv_summary(self) -> pd.DataFrame:
        """Generate consolidated summary of Atharv's three models."""
        rows = []
        for name, res in self.atharv_results.items():
            oof = res.oof_metrics
            test = self.student_test_metrics.get(name, {})
            rows.append(
                {
                    "model_name": name,
                    "owner": res.owner,
                    "cv_macro_f1": oof.get("macro_f1"),
                    "cv_accuracy": oof.get("accuracy"),
                    "cv_precision": oof.get("precision"),
                    "cv_recall": oof.get("recall"),
                    "top1_accuracy": oof.get("top1_accuracy"),
                    "candidate_miss_rate": oof.get("candidate_miss_rate"),
                    "cv_log_loss": oof.get("log_loss"),
                    "student_test_macro_f1": test.get("macro_f1"),
                    "student_test_top1_acc": test.get("top1_accuracy"),
                    "train_time_sec": res.total_train_time,
                    "infer_time_sec": res.total_infer_time,
                    "hyperparameters": str(res.hyperparameters),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by=["top1_accuracy", "cv_macro_f1"], ascending=[False, False])
        return df

    def export_atharv_cv_results(self) -> Path:
        """Export Atharv's consolidated cross-validation results to CSV."""
        df = self.generate_atharv_summary()
        path = self.output_dir / "atharv_cv_results.csv"
        df.to_csv(path, index=False)
        return path

    def generate_five_model_comparison(self) -> pd.DataFrame:
        """Consolidate Atharv's 3 models and teammate models into one table."""
        rows = []

        # 1. Atharv's models
        for name, res in self.atharv_results.items():
            oof = res.oof_metrics
            test = self.student_test_metrics.get(name, {})
            fam_key = name.lower().replace(" ", "_").replace("-", "_")
            rows.append(
                {
                    "model_name": name,
                    "model_owner": res.owner,
                    "algorithm_family": ALGORITHM_FAMILIES.get(fam_key, "Supervised"),
                    "key_configuration": str(res.hyperparameters),
                    "cv_macro_f1": oof.get("macro_f1"),
                    "cv_accuracy": oof.get("accuracy"),
                    "cv_precision": oof.get("precision"),
                    "cv_recall": oof.get("recall"),
                    "top1_accuracy": oof.get("top1_accuracy"),
                    "candidate_miss_rate": oof.get("candidate_miss_rate"),
                    "cv_log_loss": oof.get("log_loss"),
                    "student_test_macro_f1": test.get("macro_f1"),
                    "student_test_top1_acc": test.get("top1_accuracy"),
                    "train_time_sec": res.total_train_time,
                    "infer_time_sec": res.total_infer_time,
                    "route_continuity": oof.get("route_continuity"),
                }
            )

        # 2. Teammate models
        for name, ext in self.external_results.items():
            m = ext.get("cv_metrics", {})
            test = self.student_test_metrics.get(name, {})
            rows.append(
                {
                    "model_name": name,
                    "model_owner": ext.get("owner", "Teammate"),
                    "algorithm_family": ext.get("algorithm_family", "External"),
                    "key_configuration": ext.get("key_configuration", ""),
                    "cv_macro_f1": m.get("macro_f1"),
                    "cv_accuracy": m.get("accuracy"),
                    "cv_precision": m.get("precision"),
                    "cv_recall": m.get("recall"),
                    "top1_accuracy": m.get("top1_accuracy"),
                    "candidate_miss_rate": m.get("candidate_miss_rate"),
                    "cv_log_loss": m.get("log_loss"),
                    "student_test_macro_f1": test.get("macro_f1"),
                    "student_test_top1_acc": test.get("top1_accuracy"),
                    "train_time_sec": ext.get("train_time_sec", 0.0),
                    "infer_time_sec": ext.get("infer_time_sec", 0.0),
                    "route_continuity": ext.get("route_continuity"),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(
                by=["top1_accuracy", "cv_macro_f1"], ascending=[False, False]
            ).reset_index(drop=True)
        return df

    def export_five_model_comparison_csv(self) -> Path:
        """Export unified five-model comparison table to CSV."""
        df = self.generate_five_model_comparison()
        path = self.output_dir / "five_model_comparison.csv"
        df.to_csv(path, index=False)
        return path

    def plot_comparison(
        self,
        metric: str = "cv_macro_f1",
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Generate and save bar chart comparison of a metric across models."""
        df = self.generate_five_model_comparison()
        if df.empty or metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in comparison table.")

        plt.figure(figsize=(10, 5))
        models = df["model_name"].tolist()
        values = df[metric].tolist()

        colors = ["#2b5c8f", "#3a86c8", "#5ca0d3", "#82b9df", "#add2ec"][: len(models)]
        bars = plt.bar(models, values, color=colors, edgecolor="black", linewidth=0.8)

        for bar in bars:
            h = bar.get_height()
            if h is not None and not np.isnan(h):
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.01,
                    f"{h:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        plot_title = title or f"Model Comparison: {metric.replace('_', ' ').title()}"
        plt.title(plot_title, fontsize=12, fontweight="bold", pad=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=10)
        plt.xlabel("Model Architecture", fontsize=10)
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_name = filename or f"{metric}_comparison.png"
        out_path = self.output_dir / out_name
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

"""Model comparison and experiment logging utilities.

Consolidates cross-validation and test metrics across multiple model families,
formats comparison tables according to the hackathon specifications, exports
experiment CSV logs, and generates comparison visualizations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.train import CVResult


ALGORITHM_FAMILIES = {
    "logistic_regression": "Linear & Probabilistic",
    "knn": "Instance-based",
    "decision_tree": "Tree-based",
    "random_forest": "Ensemble (Bagging)",
    "hist_gradient_boosting": "Ensemble (Boosting)",
}


class ModelComparator:
    """Manages experimental logging, cross-model comparison, and artifact generation."""

    def __init__(self, output_dir: Union[str, Path] = "experiments") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, CVResult] = {}
        self.test_metrics: Dict[str, Dict[str, Any]] = {}

    def add_cv_result(
        self,
        cv_result: CVResult,
        student_test_eval: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a cross-validation result and optional student-test result."""
        self.results[cv_result.model_name] = cv_result
        if student_test_eval:
            self.test_metrics[cv_result.model_name] = student_test_eval

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate consolidated comparison table matching hackathon report specifications."""
        rows = []
        for name, res in self.results.items():
            oof = res.oof_metrics
            test = self.test_metrics.get(name, {})

            # Key config summary
            if name == "logistic_regression":
                cfg = f"C={res.hyperparameters.get('C')}, solver={res.hyperparameters.get('solver')}"
            elif name == "knn":
                cfg = f"n_neighbors={res.hyperparameters.get('n_neighbors')}, weights={res.hyperparameters.get('weights')}"
            elif name == "decision_tree":
                cfg = f"max_depth={res.hyperparameters.get('max_depth')}, min_samples_leaf={res.hyperparameters.get('min_samples_leaf')}"
            elif name == "random_forest":
                cfg = f"n_trees={res.hyperparameters.get('n_estimators')}, max_depth={res.hyperparameters.get('max_depth')}"
            elif name == "hist_gradient_boosting":
                cfg = f"lr={res.hyperparameters.get('learning_rate')}, max_iter={res.hyperparameters.get('max_iter')}"
            else:
                cfg = str(res.hyperparameters)

            rows.append(
                {
                    "model_name": name,
                    "algorithm_family": ALGORITHM_FAMILIES.get(name, "Unknown"),
                    "key_configuration": cfg,
                    "cv_macro_f1": oof.get("macro_f1"),
                    "cv_accuracy": oof.get("accuracy"),
                    "top1_accuracy": oof.get("top1_accuracy"),
                    "candidate_miss_rate": oof.get("candidate_miss_rate"),
                    "cv_log_loss": oof.get("log_loss"),
                    "student_test_macro_f1": test.get("macro_f1"),
                    "student_test_top1_acc": test.get("top1_accuracy"),
                    "train_time_sec": res.total_train_time,
                    "infer_time_sec": res.total_infer_time,
                }
            )

        df_comparison = pd.DataFrame(rows)
        if not df_comparison.empty:
            # Sort by top-1 accuracy descending then Macro F1 descending
            df_comparison = df_comparison.sort_values(
                by=["top1_accuracy", "cv_macro_f1"], ascending=[False, False]
            ).reset_index(drop=True)

        return df_comparison

    def export_comparison_csv(self) -> Path:
        """Export the consolidated comparison table to CSV."""
        df_comp = self.generate_comparison_table()
        csv_path = self.output_dir / "model_comparison.csv"
        df_comp.to_csv(csv_path, index=False)
        return csv_path

    def export_fold_results_csv(self) -> Path:
        """Export individual fold-level metrics across all evaluated models."""
        rows = []
        for name, res in self.results.items():
            for f in res.fold_metrics:
                row = dict(f)
                row["model_name"] = name
                rows.append(row)

        df_folds = pd.DataFrame(rows)
        csv_path = self.output_dir / "cross_validation_results.csv"
        df_folds.to_csv(csv_path, index=False)
        return csv_path

    def plot_metric_comparison(
        self, metric_name: str = "cv_macro_f1", title: Optional[str] = None
    ) -> Path:
        """Generate and save bar chart comparison of a specific metric across models."""
        df_comp = self.generate_comparison_table()
        if df_comp.empty or metric_name not in df_comp.columns:
            raise ValueError(f"Metric '{metric_name}' not available for plotting.")

        plt.figure(figsize=(9, 5))
        models = df_comp["model_name"].tolist()
        values = df_comp[metric_name].tolist()

        colors = ["#2b5c8f", "#3a86c8", "#5ca0d3", "#82b9df", "#add2ec"][: len(models)]
        bars = plt.bar(models, values, color=colors, edgecolor="black", linewidth=0.8)

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            if height is not None and not np.isnan(height):
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        plot_title = title or f"Model Comparison: {metric_name.replace('_', ' ').title()}"
        plt.title(plot_title, fontsize=12, fontweight="bold", pad=12)
        plt.ylabel(metric_name.replace("_", " ").title(), fontsize=10)
        plt.xlabel("Model Architecture", fontsize=10)
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = self.output_dir / f"{metric_name}_comparison.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

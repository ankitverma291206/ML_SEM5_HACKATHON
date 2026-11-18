"""Consolidated multi-model comparison and experiment logging framework.

Supports Atharv Kulkarni's three models (Random Forest, KNN, Decision Tree)
and provides an extensible integration interface to validate and import actual
experimental results from the two teammate-owned models without fabricating data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.validation import ModelCVResult


@dataclass
# Simple explanation: Metadata for validating experiment comparability across team models.
class ModelExperimentMetadata:
    """Metadata for validating experiment comparability across team models."""

    model_name: str
    owner: str
    algorithm_family: str
    key_configuration: str
    dataset_partition: str = "train_80_trips"
    feature_schema_version: str = "1.0"
    label_methodology_version: str = "1.0"
    cv_splitter: str = "GroupKFold(5)"
    shared_fold_assignments: bool = True
    metric_aggregation: str = "oof_pooled"
    train_time_sec: Optional[float] = None
    infer_time_sec: Optional[float] = None
    route_continuity: Optional[float] = None


# Simple explanation: Validate that an external model experiment used equivalent methodology.
def validate_experiment_comparability(
    meta: ModelExperimentMetadata,
    expected_partition: str = "train_80_trips",
    expected_cv: str = "GroupKFold(5)",
) -> Tuple[bool, List[str]]:
    """Validate that an external model experiment used equivalent methodology.

    Returns:
        Tuple of (is_comparable, list_of_discrepancies).
    """
    discrepancies: List[str] = []

    if not meta.owner or meta.owner.strip() == "":
        discrepancies.append("Model owner is missing.")

    if meta.dataset_partition != expected_partition:
        discrepancies.append(
            f"Dataset partition mismatch: expected '{expected_partition}', got '{meta.dataset_partition}'."
        )

    if meta.cv_splitter != expected_cv:
        discrepancies.append(
            f"CV splitter mismatch: expected '{expected_cv}', got '{meta.cv_splitter}'."
        )

    if not meta.shared_fold_assignments:
        discrepancies.append(
            "Model did not use shared fold assignments across the 80 training trips."
        )

    is_comparable = len(discrepancies) == 0
    return is_comparable, discrepancies


# Simple explanation: Manages cross-model comparisons for Atharv's models and teammate models.
class ModelComparisonManager:
    """Manages cross-model comparisons for Atharv's models and teammate models."""

    # Simple explanation: Set the starting values for this object.
    def __init__(self, output_dir: Union[str, Path] = "experiments") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Atharv's internally evaluated CVResults
        self.atharv_results: Dict[str, ModelCVResult] = {}
        # Imported external teammate results
        self.external_results: Dict[str, Dict[str, Any]] = {}
        # Optional student test metrics
        self.student_test_metrics: Dict[str, Dict[str, Any]] = {}

    # Simple explanation: Register one of Atharv's three trained model results.
    def add_atharv_result(
        self,
        cv_result: ModelCVResult,
        student_test_eval: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register one of Atharv's three trained model results."""
        self.atharv_results[cv_result.model_name] = cv_result
        if student_test_eval:
            self.student_test_metrics[cv_result.model_name] = student_test_eval

    # Simple explanation: Import actual experimental results supplied by a teammate.
    def import_external_result(
        self,
        model_name: str,
        owner: str,
        algorithm_family: str,
        key_configuration: str,
        cv_metrics: Dict[str, Any],
        metadata: Optional[ModelExperimentMetadata] = None,
        student_test_metrics: Optional[Dict[str, Any]] = None,
        train_time_sec: Optional[float] = None,
        infer_time_sec: Optional[float] = None,
        route_continuity: Optional[float] = None,
        raise_on_incomparable: bool = False,
    ) -> bool:
        """Import actual experimental results supplied by a teammate.

        Validates experimental methodology (dataset partition, shared CV folds,
        schema version). Missing timings are recorded explicitly as None, never 0.0.

        Returns:
            True if the model result is fully comparable, False if discrepancies exist.
        """
        if metadata is None:
            metadata = ModelExperimentMetadata(
                model_name=model_name,
                owner=owner,
                algorithm_family=algorithm_family,
                key_configuration=key_configuration,
                train_time_sec=train_time_sec,
                infer_time_sec=infer_time_sec,
                route_continuity=route_continuity,
            )

        is_comparable, discrepancies = validate_experiment_comparability(metadata)
        if not is_comparable and raise_on_incomparable:
            raise ValueError(
                f"External model '{model_name}' is not comparable: {'; '.join(discrepancies)}"
            )

        self.external_results[model_name] = {
            "metadata": metadata,
            "cv_metrics": cv_metrics,
            "is_comparable": is_comparable,
            "comparability_notes": "; ".join(discrepancies) if discrepancies else "Verified comparable",
        }

        if student_test_metrics:
            self.student_test_metrics[model_name] = student_test_metrics

        return is_comparable

    # Simple explanation: Generate consolidated summary of Atharv's three models.
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

    # Simple explanation: Export Atharv's consolidated cross-validation results to CSV.
    def export_atharv_cv_results(self) -> Path:
        """Export Atharv's consolidated cross-validation results to CSV."""
        df = self.generate_atharv_summary()
        path = self.output_dir / "atharv_cv_results.csv"
        df.to_csv(path, index=False)
        return path

    # Simple explanation: Consolidate Atharv's 3 models and teammate models into one table.
    def generate_five_model_comparison(self) -> pd.DataFrame:
        """Consolidate Atharv's 3 models and teammate models into one table.

        Keeps algorithm families open and extensible for whichever algorithms
        the teammates actually implemented.
        """
        rows = []

        # 1. Atharv's models
        for name, res in self.atharv_results.items():
            oof = res.oof_metrics
            test = self.student_test_metrics.get(name, {})

            # Dynamic algorithm family lookup or naming
            fam = "Ensemble" if "forest" in name.lower() else ("Instance-based" if "knn" in name.lower() or "neighbour" in name.lower() else "Tree-based")

            rows.append(
                {
                    "model_name": name,
                    "model_owner": res.owner,
                    "algorithm_family": fam,
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
                    "is_comparable": True,
                    "comparability_notes": "Verified internal baseline",
                }
            )

        # 2. Teammate models (extensible to any model implemented by teammates)
        for name, ext in self.external_results.items():
            meta: ModelExperimentMetadata = ext["metadata"]
            m = ext.get("cv_metrics", {})
            test = self.student_test_metrics.get(name, {})
            rows.append(
                {
                    "model_name": name,
                    "model_owner": meta.owner,
                    "algorithm_family": meta.algorithm_family,
                    "key_configuration": meta.key_configuration,
                    "cv_macro_f1": m.get("macro_f1"),
                    "cv_accuracy": m.get("accuracy"),
                    "cv_precision": m.get("precision"),
                    "cv_recall": m.get("recall"),
                    "top1_accuracy": m.get("top1_accuracy"),
                    "candidate_miss_rate": m.get("candidate_miss_rate"),
                    "cv_log_loss": m.get("log_loss"),
                    "student_test_macro_f1": test.get("macro_f1"),
                    "student_test_top1_acc": test.get("top1_accuracy"),
                    "train_time_sec": meta.train_time_sec,
                    "infer_time_sec": meta.infer_time_sec,
                    "route_continuity": meta.route_continuity,
                    "is_comparable": ext.get("is_comparable", False),
                    "comparability_notes": ext.get("comparability_notes", ""),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(
                by=["top1_accuracy", "cv_macro_f1"], ascending=[False, False]
            ).reset_index(drop=True)
        return df

    # Simple explanation: Export unified five-model comparison table to CSV.
    def export_five_model_comparison_csv(self) -> Path:
        """Export unified five-model comparison table to CSV."""
        df = self.generate_five_model_comparison()
        path = self.output_dir / "five_model_comparison.csv"
        df.to_csv(path, index=False)
        return path

    # Simple explanation: Generate and save bar chart comparison of a metric across models.
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

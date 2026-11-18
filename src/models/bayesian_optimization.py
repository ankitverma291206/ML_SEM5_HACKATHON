"""Bayesian hyperparameter optimization framework using Optuna and TPE.

Implements leak-free, group-aware cross-validation objectives for hyperparameter tuning.
Maintains full trial history, records evaluation metrics per trial, and exports
tuning results to experiments/optimization_trials.csv.

Final Bayesian optimization is executed ONLY AFTER the initial five-model comparison
has identified the most promising candidate.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from src.models.config import FeatureConfig
from src.models.decision_tree import build_decision_tree_pipeline
from src.models.evaluate import Timer, evaluate_model_predictions
from src.models.knn import build_knn_pipeline
from src.models.random_forest import build_random_forest_pipeline


def sample_search_space(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """Define a scientifically principled hyperparameter search space for Optuna."""
    name = model_name.lower().replace(" ", "_").replace("-", "_")

    if name in {"random_forest", "rf"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=25),
            "max_depth": trial.suggest_int("max_depth", 4, 18),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 26, step=2),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15, step=1),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    elif name in {"knn", "k_nearest_neighbours"}:
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 45, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["minkowski", "manhattan"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
    elif name in {"decision_tree", "dt"}:
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30, step=2),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, step=1),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    else:
        raise ValueError(f"No search space defined for model: '{model_name}'.")


def get_pipeline_factory_for_model(model_name: str) -> Callable[..., Pipeline]:
    """Return pipeline builder for Atharv's models."""
    name = model_name.lower().replace(" ", "_").replace("-", "_")
    if name in {"random_forest", "rf"}:
        return build_random_forest_pipeline
    elif name in {"knn", "k_nearest_neighbours"}:
        return build_knn_pipeline
    elif name in {"decision_tree", "dt"}:
        return build_decision_tree_pipeline
    else:
        raise ValueError(f"Unknown model name: '{model_name}'")


class BayesianOptimizer:
    """Manages Optuna TPE Bayesian hyperparameter optimization."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        model_name: str,
        pipeline_factory: Optional[Callable[..., Pipeline]] = None,
        config: Optional[FeatureConfig] = None,
        metric_to_optimize: str = "top1_accuracy",  # "top1_accuracy" or "macro_f1"
        n_splits: int = 5,
        random_seed: int = 42,
        output_dir: Union[str, Path] = "experiments",
    ) -> None:
        self.df_train = df_train
        self.model_name = model_name
        self.pipeline_factory = pipeline_factory or get_pipeline_factory_for_model(model_name)
        self.config = config or FeatureConfig()
        self.metric_to_optimize = metric_to_optimize
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.study: Optional[optuna.Study] = None
        self.trials_df: Optional[pd.DataFrame] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_metric_value: Optional[float] = None
        self.optimization_time: float = 0.0

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function using GroupKFold cross-validation on training trips."""
        params = sample_search_space(trial, self.model_name)

        trips = self.df_train["trip_id"].values
        unique_trips = np.unique(trips)
        splits = min(self.n_splits, len(unique_trips))

        gkf = GroupKFold(n_splits=splits)
        fold_scores: List[float] = []

        X = self.df_train[self.config.feature_columns]
        y = self.df_train[self.config.label_column].values

        for train_idx, val_idx in gkf.split(X, y, groups=trips):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]

            pipeline = self.pipeline_factory(
                config=self.config, custom_params=params
            )
            pipeline.fit(X_tr, y_tr)

            val_df = self.df_train.iloc[val_idx].copy()
            y_pred = pipeline.predict(X_va)
            val_df["pred_matching"] = y_pred

            if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
                classes = list(pipeline.named_steps["classifier"].classes_)
                pos_idx = classes.index(1) if 1 in classes else 1
                val_df["prob_matching"] = pipeline.predict_proba(X_va)[:, pos_idx]
            else:
                val_df["prob_matching"] = y_pred.astype(float)

            res = evaluate_model_predictions(
                val_df,
                pred_col="pred_matching",
                prob_col="prob_matching",
                label_col=self.config.label_column,
                trip_col="trip_id",
                obs_col="observation_id",
            )

            score = res.get(self.metric_to_optimize)
            if score is None:
                score = res.get("macro_f1", 0.0)
            fold_scores.append(score)

        return float(np.mean(fold_scores))

    def run_optimization(
        self, n_trials: int = 30, timeout_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run Optuna TPE Bayesian optimization."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"{self.model_name}_optimization",
        )

        with Timer() as t:
            self.study.optimize(
                self._objective,
                n_trials=n_trials,
                timeout=timeout_seconds,
                show_progress_bar=False,
            )

        self.optimization_time = t.elapsed
        self.best_params = self.study.best_params
        self.best_metric_value = self.study.best_value

        trials_data = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                row = {
                    "trial_number": trial.number,
                    "value": trial.value,
                    "duration_sec": trial.duration.total_seconds() if trial.duration else None,
                    **trial.params,
                }
                trials_data.append(row)

        self.trials_df = pd.DataFrame(trials_data)

        return {
            "best_params": self.best_params,
            "best_metric_value": self.best_metric_value,
            "metric_name": self.metric_to_optimize,
            "n_completed_trials": len(trials_data),
            "optimization_time_sec": self.optimization_time,
        }

    def export_trials_csv(self) -> Path:
        """Export trial history to CSV."""
        if self.trials_df is None or self.trials_df.empty:
            raise RuntimeError("No optimization trials have been run yet.")

        csv_path = self.output_dir / "optimization_trials.csv"
        self.trials_df.to_csv(csv_path, index=False)
        return csv_path

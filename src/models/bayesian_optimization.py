"""Bayesian hyperparameter optimization framework using Optuna and TPE.

Implements leak-free, group-aware cross-validation objectives for hyperparameter tuning.
Maintains full trial history, records evaluation metrics per trial, and exports
tuning results for the hackathon deliverables.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.models.config import FeatureConfig
from src.models.evaluate import Timer, evaluate_model_predictions
from src.models.pipeline import build_model_pipeline, get_default_hyperparameters


def define_search_space(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """Define a scientifically principled hyperparameter search space for Optuna."""
    name = model_name.lower().strip()

    if name == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 50, 250, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100, step=10),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }
    elif name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 30, step=5),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20, step=2),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }
    elif name == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 40, step=5),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 25, step=2),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }
    elif name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        }
    elif name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 35, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
    else:
        raise ValueError(f"No search space defined for model: {model_name}")


class BayesianOptimizer:
    """Manages Optuna TPE Bayesian hyperparameter optimization."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        model_name: str,
        config: Optional[FeatureConfig] = None,
        metric_to_optimize: str = "macro_f1",  # "macro_f1" or "top1_accuracy"
        n_splits: int = 5,
        random_seed: int = 42,
        output_dir: Union[str, Path] = "experiments",
    ) -> None:
        self.df_train = df_train
        self.model_name = model_name
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
        """Optuna objective function using GroupKFold on training trips."""
        params = define_search_space(trial, self.model_name)

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

            pipeline = build_model_pipeline(
                self.model_name, self.config, custom_params=params
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

    def run_optimization(self, n_trials: int = 30, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Run the TPE Bayesian optimization process."""
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

        # Build trials DataFrame
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
        """Export the trial history to CSV."""
        if self.trials_df is None or self.trials_df.empty:
            raise RuntimeError("No trials have been run yet.")

        csv_path = self.output_dir / "optimization_trials.csv"
        self.trials_df.to_csv(csv_path, index=False)
        return csv_path

"""Trip-grouped cross-validation framework for PathSnap.

Provides reproducible, leak-free 5-fold GroupKFold cross-validation grouped
by trip_id. Guarantees identical fold assignments across all models (Atharv's
three models and teammate models) to enable fair, scientifically defensible
comparison.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from src.models.config import FeatureConfig
from src.models.evaluate import Timer, evaluate_model_predictions


def get_positive_class_index(pipeline: Pipeline, pos_label: int = 1) -> int:
    """Safely find the column index for the target positive class in predict_proba.

    Verifies the classifier's `classes_` attribute rather than assuming index 1.
    """
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "classes_"):
        classes = list(classifier.classes_)
        if pos_label in classes:
            return classes.index(pos_label)
    return 1


@dataclass
class ModelCVResult:
    """Container for Cross-Validation experimentation outputs."""

    model_name: str
    owner: str
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    n_splits: int
    fold_metrics: List[Dict[str, Any]]
    oof_metrics: Dict[str, Any]
    oof_predictions_df: pd.DataFrame
    total_train_time: float
    total_infer_time: float


def run_trip_group_cv(
    df_train: pd.DataFrame,
    pipeline_factory: Any,
    model_name: str,
    owner: str = "Atharv Kulkarni",
    config: Optional[FeatureConfig] = None,
    n_splits: int = 5,
    custom_params: Optional[Dict[str, Any]] = None,
) -> ModelCVResult:
    """Execute trip-grouped cross-validation for any scikit-learn compatible pipeline.

    Args:
        df_train: DataFrame containing candidate rows across training trips.
        pipeline_factory: Callable taking (config, custom_params) returning a Pipeline.
        model_name: Name of the model being evaluated.
        owner: Teammate who owns this model.
        config: Feature configuration object.
        n_splits: Number of GroupKFold splits (default: 5).
        custom_params: Hyperparameter overrides.

    Returns:
        ModelCVResult with out-of-fold metrics, fold records, and OOF predictions.
    """
    if config is None:
        config = FeatureConfig()

    config.validate_dataframe(df_train, require_label=True)

    trips = df_train["trip_id"].values
    unique_trips = np.unique(trips)
    splits = min(n_splits, len(unique_trips))
    if splits < 2:
        raise ValueError(
            f"Cannot run GroupKFold with only {len(unique_trips)} unique trips."
        )

    gkf = GroupKFold(n_splits=splits)

    fold_metrics: List[Dict[str, Any]] = []
    oof_dfs: List[pd.DataFrame] = []

    total_train_time = 0.0
    total_infer_time = 0.0

    X_all = df_train[config.feature_columns]
    y_all = df_train[config.label_column].values

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups=trips)):
        train_trips = set(trips[train_idx])
        val_trips = set(trips[val_idx])

        # Strict leakage verification
        leakage = train_trips.intersection(val_trips)
        if leakage:
            raise RuntimeError(f"Trip leakage detected in fold {fold_idx}: {leakage}")

        X_train = X_all.iloc[train_idx]
        y_train = y_all[train_idx]
        X_val = X_all.iloc[val_idx]
        y_val = y_all[val_idx]

        # Fresh pipeline instance for each fold (scaling & imputation fit strictly on train fold)
        pipeline = pipeline_factory(config=config, custom_params=custom_params)

        # Fit strictly on training fold
        with Timer() as t_train:
            pipeline.fit(X_train, y_train)
        fold_train_time = t_train.elapsed
        total_train_time += fold_train_time

        # Predict on validation fold
        with Timer() as t_infer:
            y_pred = pipeline.predict(X_val)
        fold_infer_time = t_infer.elapsed
        total_infer_time += fold_infer_time

        # Probability scoring
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            probs = pipeline.predict_proba(X_val)
            pos_idx = get_positive_class_index(pipeline, pos_label=1)
            y_prob = probs[:, pos_idx]
        else:
            y_prob = y_pred.astype(float)

        val_df = df_train.iloc[val_idx].copy()
        val_df["pred_matching"] = y_pred
        val_df["prob_matching"] = y_prob
        val_df["fold"] = fold_idx

        # Fold evaluation
        f_eval = evaluate_model_predictions(
            val_df,
            pred_col="pred_matching",
            prob_col="prob_matching",
            label_col=config.label_column,
            trip_col="trip_id",
            obs_col="observation_id",
        )
        f_eval["fold"] = fold_idx
        f_eval["train_time"] = fold_train_time
        f_eval["infer_time"] = fold_infer_time
        f_eval["n_train_trips"] = len(train_trips)
        f_eval["n_val_trips"] = len(val_trips)

        fold_metrics.append(f_eval)
        oof_dfs.append(val_df)

    oof_predictions_df = pd.concat(oof_dfs, axis=0).sort_index()

    oof_eval = evaluate_model_predictions(
        oof_predictions_df,
        pred_col="pred_matching",
        prob_col="prob_matching",
        label_col=config.label_column,
        trip_col="trip_id",
        obs_col="observation_id",
    )
    oof_eval["total_train_time"] = total_train_time
    oof_eval["total_infer_time"] = total_infer_time
    oof_eval["mean_fold_train_time"] = total_train_time / splits
    oof_eval["mean_fold_infer_time"] = total_infer_time / splits

    return ModelCVResult(
        model_name=model_name,
        owner=owner,
        hyperparameters=custom_params or {},
        feature_columns=list(config.feature_columns),
        n_splits=splits,
        fold_metrics=fold_metrics,
        oof_metrics=oof_eval,
        oof_predictions_df=oof_predictions_df,
        total_train_time=total_train_time,
        total_infer_time=total_infer_time,
    )

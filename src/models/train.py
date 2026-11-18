"""Group-aware cross-validation and model training framework.

Executes strictly leak-free GroupKFold cross-validation grouped by trip_id.
Tracks fold-level and out-of-fold evaluation metrics, training times,
and inference times.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from src.models.config import FeatureConfig
from src.models.evaluate import Timer, evaluate_model_predictions
from src.models.pipeline import build_model_pipeline, get_default_hyperparameters


def get_positive_class_index(pipeline: Pipeline, pos_label: int = 1) -> int:
    """Find the exact column index for the positive class (1) in predict_proba.

    Prevents bugs where class 1 is not at index 1 (e.g. if single class or ordered differently).
    """
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "classes_"):
        classes = list(classifier.classes_)
        if pos_label in classes:
            return classes.index(pos_label)
    return 1


@dataclass
class CVResult:
    """Container for Cross-Validation experimentation outputs."""

    model_name: str
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    n_splits: int
    fold_metrics: List[Dict[str, Any]]
    oof_metrics: Dict[str, Any]
    oof_predictions_df: pd.DataFrame
    total_train_time: float
    total_infer_time: float


def run_group_kfold_cv(
    df: pd.DataFrame,
    model_name: str,
    config: Optional[FeatureConfig] = None,
    n_splits: int = 5,
    custom_params: Optional[Dict[str, Any]] = None,
) -> CVResult:
    """Execute trip-grouped cross-validation for a candidate classification model.

    Args:
        df: DataFrame containing training candidate rows across trips.
        model_name: Name of model to train.
        config: FeatureConfig object specifying features, identifiers, and labels.
        n_splits: Number of GroupKFold splits (default: 5).
        custom_params: Optional hyperparameter overrides.

    Returns:
        CVResult containing fold metrics, out-of-fold metrics, and OOF predictions.
    """
    if config is None:
        config = FeatureConfig()

    config.validate_dataframe(df, require_label=True)

    trips = df["trip_id"].values
    unique_trips = np.unique(trips)
    actual_splits = min(n_splits, len(unique_trips))
    if actual_splits < 2:
        raise ValueError(
            f"Cannot run GroupKFold with only {len(unique_trips)} unique trips."
        )

    gkf = GroupKFold(n_splits=actual_splits)

    fold_metrics: List[Dict[str, Any]] = []
    oof_dfs: List[pd.DataFrame] = []

    total_train_time = 0.0
    total_infer_time = 0.0

    X_all = df[config.feature_columns]
    y_all = df[config.label_column].values

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

        # Build a fresh, isolated pipeline for each fold
        pipeline = build_model_pipeline(model_name, config, custom_params)

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

        # Extract positive class probability
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            probs = pipeline.predict_proba(X_val)
            pos_idx = get_positive_class_index(pipeline, pos_label=1)
            y_prob = probs[:, pos_idx]
        else:
            # Fallback if probability is not directly supported
            y_prob = y_pred.astype(float)

        val_df = df.iloc[val_idx].copy()
        val_df["pred_matching"] = y_pred
        val_df["prob_matching"] = y_prob
        val_df["fold"] = fold_idx

        # Evaluate current fold
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

    # Combine Out-Of-Fold predictions across all folds
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
    oof_eval["mean_fold_train_time"] = total_train_time / actual_splits
    oof_eval["mean_fold_infer_time"] = total_infer_time / actual_splits

    # Collect used hyperparameters
    used_params = get_default_hyperparameters(model_name, random_seed=config.random_seed)
    if custom_params:
        used_params.update(custom_params)

    return CVResult(
        model_name=model_name,
        hyperparameters=used_params,
        feature_columns=list(config.feature_columns),
        n_splits=actual_splits,
        fold_metrics=fold_metrics,
        oof_metrics=oof_eval,
        oof_predictions_df=oof_predictions_df,
        total_train_time=total_train_time,
        total_infer_time=total_infer_time,
    )


def train_full_model(
    df_train: pd.DataFrame,
    model_name: str,
    config: Optional[FeatureConfig] = None,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Pipeline, float]:
    """Train a final model on all permitted training trips.

    Args:
        df_train: Training DataFrame (all training trips).
        model_name: Name of model to train.
        config: Feature configuration.
        custom_params: Optional hyperparameter overrides.

    Returns:
        Tuple of (fitted Pipeline, training time in seconds).
    """
    if config is None:
        config = FeatureConfig()

    config.validate_dataframe(df_train, require_label=True)

    X_train = df_train[config.feature_columns]
    y_train = df_train[config.label_column].values

    pipeline = build_model_pipeline(model_name, config, custom_params)

    with Timer() as t:
        pipeline.fit(X_train, y_train)

    return pipeline, t.elapsed

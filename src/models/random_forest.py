"""Random Forest Classifier for Candidate Road-Segment Classification.

Atharv Kulkarni's Model 1.
Investigates:
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features
- class_weight
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.config import FeatureConfig
from src.models.evaluate import Timer
from src.models.validation import ModelCVResult, run_trip_group_cv


DEFAULT_RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "class_weight": None,  # Can also be "balanced" or balanced_subsample
    "n_jobs": -1,
    "random_state": 42,
}


# Simple explanation: Prepare data and build the Random Forest model.
def build_random_forest_pipeline(
    config: Optional[FeatureConfig] = None,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Build scikit-learn Pipeline with median imputation and RandomForestClassifier."""
    if config is None:
        config = FeatureConfig()

    params = dict(DEFAULT_RF_PARAMS)
    params["random_state"] = config.random_seed
    if custom_params:
        params.update(custom_params)

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if config.scale_features:
        steps.append(("scaler", StandardScaler()))

    steps.append(("classifier", RandomForestClassifier(**params)))
    return Pipeline(steps)


# Simple explanation: Check the Random Forest using trips held out from training.
def evaluate_random_forest(
    df_train: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    custom_params: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    output_dir: Union[str, Path] = "experiments",
    all_observations: Optional[Union[int, pd.DataFrame]] = None,
) -> ModelCVResult:
    """Run trip-grouped cross-validation for Random Forest and export fold metrics."""
    if config is None:
        config = FeatureConfig()

    params = dict(DEFAULT_RF_PARAMS)
    if custom_params:
        params.update(custom_params)

    result = run_trip_group_cv(
        df_train=df_train,
        pipeline_factory=build_random_forest_pipeline,
        model_name="Random Forest",
        owner="Atharv Kulkarni",
        config=config,
        n_splits=n_splits,
        custom_params=params,
        all_observations=all_observations,
    )

    # Export fold results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_folds = pd.DataFrame(result.fold_metrics)
    df_folds["model_name"] = "Random Forest"
    df_folds["owner"] = "Atharv Kulkarni"
    df_folds.to_csv(out_dir / "atharv_random_forest_results.csv", index=False)

    return result


# Simple explanation: Train the Random Forest on the available training data and save it.
def fit_and_save_random_forest(
    df_train: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    custom_params: Optional[Dict[str, Any]] = None,
    artifacts_dir: Union[str, Path] = "artifacts",
) -> Pipeline:
    """Fit full Random Forest model on all training trips and save artifact."""
    if config is None:
        config = FeatureConfig()

    config.validate_dataframe(df_train, require_label=True)
    pipeline = build_random_forest_pipeline(config, custom_params)

    X = df_train[config.feature_columns]
    y = df_train[config.label_column].values

    pipeline.fit(X, y)

    art_dir = Path(artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = art_dir / "atharv_random_forest.pkl"
    joblib.dump(pipeline, artifact_path)

    return pipeline

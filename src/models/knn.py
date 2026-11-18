"""K-Nearest Neighbours (KNN) Classifier for Candidate Road-Segment Classification.

Atharv Kulkarni's Model 2.
Investigates:
- n_neighbors
- weights ('uniform', 'distance')
- metric ('minkowski', 'euclidean', 'manhattan')
- p (1 for Manhattan, 2 for Euclidean)

Strictly uses StandardScaler fitted only on the training fold to prevent data leakage.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.config import FeatureConfig
from src.models.evaluate import Timer
from src.models.validation import ModelCVResult, run_trip_group_cv


DEFAULT_KNN_PARAMS: Dict[str, Any] = {
    "n_neighbors": 15,
    "weights": "distance",
    "algorithm": "kd_tree",
    "metric": "minkowski",
    "p": 2,
    "n_jobs": -1,
}


def build_knn_pipeline(
    config: Optional[FeatureConfig] = None,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Build scikit-learn Pipeline with median imputation, feature scaling, and KNN.

    StandardScaler is mandatory for distance-based nearest neighbours.
    """
    if config is None:
        config = FeatureConfig()

    params = dict(DEFAULT_KNN_PARAMS)
    if custom_params:
        params.update(custom_params)

    # Scaling is mandatory for distance metric validity
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(**params)),
    ]
    return Pipeline(steps)


def evaluate_knn(
    df_train: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    custom_params: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    output_dir: Union[str, Path] = "experiments",
    all_observations: Optional[Union[int, pd.DataFrame]] = None,
) -> ModelCVResult:
    """Run trip-grouped cross-validation for KNN and export fold metrics."""
    if config is None:
        config = FeatureConfig()

    params = dict(DEFAULT_KNN_PARAMS)
    if custom_params:
        params.update(custom_params)

    result = run_trip_group_cv(
        df_train=df_train,
        pipeline_factory=build_knn_pipeline,
        model_name="K-Nearest Neighbours",
        owner="Atharv Kulkarni",
        config=config,
        n_splits=n_splits,
        custom_params=params,
        all_observations=all_observations,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_folds = pd.DataFrame(result.fold_metrics)
    df_folds["model_name"] = "K-Nearest Neighbours"
    df_folds["owner"] = "Atharv Kulkarni"
    df_folds.to_csv(out_dir / "atharv_knn_results.csv", index=False)

    return result


def fit_and_save_knn(
    df_train: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    custom_params: Optional[Dict[str, Any]] = None,
    artifacts_dir: Union[str, Path] = "artifacts",
) -> Pipeline:
    """Fit full KNN model on all training trips and save artifact."""
    if config is None:
        config = FeatureConfig()

    config.validate_dataframe(df_train, require_label=True)
    pipeline = build_knn_pipeline(config, custom_params)

    X = df_train[config.feature_columns]
    y = df_train[config.label_column].values

    pipeline.fit(X, y)

    art_dir = Path(artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = art_dir / "atharv_knn.pkl"
    joblib.dump(pipeline, artifact_path)

    return pipeline

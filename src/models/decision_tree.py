"""Decision Tree Classifier for Candidate Road-Segment Classification.

Atharv Kulkarni's Model 3.
Investigates:
- max_depth
- min_samples_split
- min_samples_leaf
- criterion ('gini', 'entropy')
- class_weight (None, 'balanced')
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.models.config import FeatureConfig
from src.models.evaluate import Timer
from src.models.validation import ModelCVResult, run_trip_group_cv


DEFAULT_DT_PARAMS: Dict[str, Any] = {
    "max_depth": 8,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "criterion": "gini",
    "class_weight": None,
    "random_state": 42,
}


def build_decision_tree_pipeline(
    config: Optional[FeatureConfig] = None,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Build scikit-learn Pipeline with median imputation and DecisionTreeClassifier."""
    if config is None:
        config = FeatureConfig()

    params = dict(DEFAULT_DT_PARAMS)
    params["random_state"] = config.random_seed
    if custom_params:
        params.update(custom_params)

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if config.scale_features:
        steps.append(("scaler", StandardScaler()))

    steps.append(("classifier", DecisionTreeClassifier(**params)))
    return Pipeline(steps)


def evaluate_decision_tree(
    df_train: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    custom_params: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    output_dir: Union[str, Path] = "experiments",
    all_observations: Optional[Union[int, pd.DataFrame]] = None,
) -> ModelCVResult:
    """Run trip-grouped cross-validation for Decision Tree and export fold metrics."""
    if config is None:
        config = FeatureConfig()

    params = dict(DEFAULT_DT_PARAMS)
    if custom_params:
        params.update(custom_params)

    result = run_trip_group_cv(
        df_train=df_train,
        pipeline_factory=build_decision_tree_pipeline,
        model_name="Decision Tree",
        owner="Atharv Kulkarni",
        config=config,
        n_splits=n_splits,
        custom_params=params,
        all_observations=all_observations,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_folds = pd.DataFrame(result.fold_metrics)
    df_folds["model_name"] = "Decision Tree"
    df_folds["owner"] = "Atharv Kulkarni"
    df_folds.to_csv(out_dir / "atharv_decision_tree_results.csv", index=False)

    return result


def fit_and_save_decision_tree(
    df_train: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    custom_params: Optional[Dict[str, Any]] = None,
    artifacts_dir: Union[str, Path] = "artifacts",
) -> Pipeline:
    """Fit full Decision Tree model on all training trips and save artifact."""
    if config is None:
        config = FeatureConfig()

    config.validate_dataframe(df_train, require_label=True)
    pipeline = build_decision_tree_pipeline(config, custom_params)

    X = df_train[config.feature_columns]
    y = df_train[config.label_column].values

    pipeline.fit(X, y)

    art_dir = Path(artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = art_dir / "atharv_decision_tree.pkl"
    joblib.dump(pipeline, artifact_path)

    return pipeline

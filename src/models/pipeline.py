"""Pipeline factory and model builders for candidate classification.

Constructs scikit-learn Pipelines that encapsulate imputation, feature scaling,
and classification estimators. Ensures that all transformations are strictly fitted
on training folds without leaking data into validation folds.
"""

from typing import Any, Dict, Optional
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.models.config import FeatureConfig


SUPPORTED_MODELS = {
    "logistic_regression",
    "knn",
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
}


def get_default_hyperparameters(model_name: str, random_seed: int = 42) -> Dict[str, Any]:
    """Return verified baseline hyperparameters for a given model architecture."""
    name = model_name.lower().strip()
    if name == "logistic_regression":
        return {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": random_seed,
        }
    elif name == "knn":
        return {
            "n_neighbors": 15,
            "weights": "distance",
            "metric": "minkowski",
            "p": 2,
            "n_jobs": -1,
        }
    elif name == "decision_tree":
        return {
            "max_depth": 8,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "criterion": "gini",
            "random_state": random_seed,
        }
    elif name == "random_forest":
        return {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "n_jobs": -1,
            "random_state": random_seed,
        }
    elif name == "hist_gradient_boosting":
        return {
            "max_iter": 100,
            "learning_rate": 0.1,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
            "random_state": random_seed,
        }
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Supported: {SUPPORTED_MODELS}")


def build_model_pipeline(
    model_name: str,
    config: Optional[FeatureConfig] = None,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Build a complete, leakage-free scikit-learn Pipeline for the requested model.

    Args:
        model_name: One of the 5 supported model types.
        config: FeatureConfig object containing preprocessing options and seeds.
        custom_params: Optional dictionary of hyperparameter overrides.

    Returns:
        Pipeline instance containing imputer, scaler (if applicable), and classifier.
    """
    if config is None:
        config = FeatureConfig()

    name = model_name.lower().strip()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{name}'. Choose from: {sorted(SUPPORTED_MODELS)}")

    # Merge default hyperparameters with overrides
    params = get_default_hyperparameters(name, random_seed=config.random_seed)
    if custom_params:
        params.update(custom_params)

    steps = []

    # 1. Missing Value Imputation
    if config.impute_missing:
        steps.append(("imputer", SimpleImputer(strategy="median")))

    # 2. Feature Scaling
    # Logistic Regression and KNN strictly require scaling due to gradient/distance calculations.
    # For tree-based models, scaling is neutral, but can be enabled via config.
    requires_scaling = name in {"logistic_regression", "knn"} or config.scale_features
    if requires_scaling:
        steps.append(("scaler", StandardScaler()))

    # 3. Classifier Estimator
    if name == "logistic_regression":
        estimator = LogisticRegression(**params)
    elif name == "knn":
        estimator = KNeighborsClassifier(**params)
    elif name == "decision_tree":
        estimator = DecisionTreeClassifier(**params)
    elif name == "random_forest":
        estimator = RandomForestClassifier(**params)
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(**params)

    steps.append(("classifier", estimator))

    return Pipeline(steps)

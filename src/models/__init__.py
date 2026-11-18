"""PathSnap Machine Learning Models Package (Atharv Kulkarni).

Provides candidate road-segment classification, leak-free group cross-validation,
rigorous evaluation, Bayesian hyperparameter optimization, and inference scoring.
"""

from src.models.config import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_IDENTIFIER_COLUMNS,
    DEFAULT_LABEL_COLUMN,
    FeatureConfig,
)
from src.models.evaluate import (
    Timer,
    compute_candidate_metrics,
    compute_candidate_miss_rate,
    compute_observation_top1_accuracy,
    evaluate_model_predictions,
)
from src.models.inference import (
    load_model_artifacts,
    save_model_artifacts,
    score_candidates,
)
from src.models.model_comparison import (
    ALGORITHM_FAMILIES,
    ModelComparator,
)
from src.models.pipeline import (
    SUPPORTED_MODELS,
    build_model_pipeline,
    get_default_hyperparameters,
)
from src.models.train import (
    CVResult,
    run_group_kfold_cv,
    train_full_model,
)
from src.models.bayesian_optimization import (
    BayesianOptimizer,
    define_search_space,
)

__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "DEFAULT_IDENTIFIER_COLUMNS",
    "DEFAULT_LABEL_COLUMN",
    "FeatureConfig",
    "Timer",
    "compute_candidate_metrics",
    "compute_candidate_miss_rate",
    "compute_observation_top1_accuracy",
    "evaluate_model_predictions",
    "load_model_artifacts",
    "save_model_artifacts",
    "score_candidates",
    "ALGORITHM_FAMILIES",
    "ModelComparator",
    "SUPPORTED_MODELS",
    "build_model_pipeline",
    "get_default_hyperparameters",
    "CVResult",
    "run_group_kfold_cv",
    "train_full_model",
    "BayesianOptimizer",
    "define_search_space",
]

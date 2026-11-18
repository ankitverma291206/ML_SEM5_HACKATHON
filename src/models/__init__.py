"""PathSnap Machine Learning Models Package (Atharv Kulkarni).

Owns:
- Model 1: Random Forest Classifier
- Model 2: K-Nearest Neighbours (KNN)
- Model 3: Decision Tree Classifier
- Trip-grouped cross-validation framework (GroupKFold by trip_id)
- Multi-model evaluation and consolidated five-model comparison
- Optuna TPE Bayesian hyperparameter optimization engine
- Candidate scoring and artifact persistence interface
"""

from src.models.config import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_IDENTIFIER_COLUMNS,
    DEFAULT_LABEL_COLUMN,
    FeatureConfig,
)
from src.models.decision_tree import (
    build_decision_tree_pipeline,
    evaluate_decision_tree,
    fit_and_save_decision_tree,
)
from src.models.evaluate import (
    Timer,
    compute_candidate_metrics,
    compute_candidate_miss_rate,
    compute_observation_top1_accuracy,
    evaluate_model_predictions,
)
from src.models.knn import (
    build_knn_pipeline,
    evaluate_knn,
    fit_and_save_knn,
)
from src.models.model_comparison import (
    ALGORITHM_FAMILIES,
    ModelComparisonManager,
)
from src.models.random_forest import (
    build_random_forest_pipeline,
    evaluate_random_forest,
    fit_and_save_random_forest,
)
from src.models.scoring import (
    load_model_artifacts,
    save_model_artifacts,
    score_candidates,
)
from src.models.validation import (
    ModelCVResult,
    get_positive_class_index,
    run_trip_group_cv,
)
from src.models.bayesian_optimization import (
    BayesianOptimizer,
    sample_search_space,
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
    "ModelComparisonManager",
    "ModelCVResult",
    "get_positive_class_index",
    "run_trip_group_cv",
    "build_random_forest_pipeline",
    "evaluate_random_forest",
    "fit_and_save_random_forest",
    "build_knn_pipeline",
    "evaluate_knn",
    "fit_and_save_knn",
    "build_decision_tree_pipeline",
    "evaluate_decision_tree",
    "fit_and_save_decision_tree",
    "BayesianOptimizer",
    "sample_search_space",
]

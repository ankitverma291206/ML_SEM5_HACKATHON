"""Unit tests for Atharv Kulkarni's ML experimentation infrastructure.

Uses small SYNTHETIC fixtures strictly to verify software mechanics,
leak-free GroupKFold cross-validation for Atharv's three models
(Random Forest, KNN, Decision Tree), five-model consolidation, and the inference API.
THESE SYNTHETIC METRICS ARE NEVER INCLUDED IN HACKATHON EXPERIMENTAL RESULTS.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models import (
    BayesianOptimizer,
    FeatureConfig,
    ModelComparisonManager,
    build_decision_tree_pipeline,
    build_knn_pipeline,
    build_random_forest_pipeline,
    compute_candidate_metrics,
    compute_candidate_miss_rate,
    compute_observation_top1_accuracy,
    evaluate_decision_tree,
    evaluate_knn,
    evaluate_random_forest,
    fit_and_save_decision_tree,
    fit_and_save_knn,
    fit_and_save_random_forest,
    load_model_artifacts,
    run_trip_group_cv,
    save_model_artifacts,
    score_candidates,
)


@pytest.fixture
def synthetic_candidate_dataset() -> pd.DataFrame:
    """Generate a small SYNTHETIC candidate dataset across multiple trips for mechanics testing."""
    np.random.seed(42)
    rows = []
    # 6 synthetic trips, 5 observations each, 4 candidates per observation
    for trip_idx in range(6):
        trip_id = f"synthetic_trip_{trip_idx:03d}"
        for obs_idx in range(5):
            obs_id = obs_idx
            matching_candidate_idx = np.random.randint(0, 4)
            for cand_idx in range(4):
                arc_id = 1000 + obs_idx * 10 + cand_idx
                is_match = 1 if cand_idx == matching_candidate_idx else 0

                if is_match == 1:
                    distance = np.random.exponential(scale=5.0)
                    heading_diff = np.random.uniform(0.0, 20.0)
                else:
                    distance = np.random.exponential(scale=35.0) + 10.0
                    heading_diff = np.random.uniform(15.0, 180.0)

                rows.append(
                    {
                        "trip_id": trip_id,
                        "observation_id": obs_id,
                        "arc_id": arc_id,
                        "distance_m": distance,
                        "projection_position": np.random.uniform(0.0, 1.0),
                        "road_bearing": np.random.uniform(0.0, 360.0),
                        "gps_heading": np.random.uniform(0.0, 360.0),
                        "heading_difference": heading_diff,
                        "gps_speed": np.random.uniform(5.0, 25.0),
                        "time_delta": np.random.choice([1.0, 2.0, 5.0]),
                        "segment_length": np.random.uniform(50.0, 300.0),
                        "is_matching_candidate": is_match,
                    }
                )

    df = pd.DataFrame(rows)
    # Add a few intentional NaNs to test imputer robustness
    df.loc[3, "gps_speed"] = np.nan
    df.loc[7, "heading_difference"] = np.nan
    return df


def test_feature_config_validation(synthetic_candidate_dataset: pd.DataFrame):
    config = FeatureConfig()
    config.validate_dataframe(synthetic_candidate_dataset, require_label=True)

    bad_df = synthetic_candidate_dataset.drop(columns=["distance_m"])
    with pytest.raises(ValueError, match="missing configured feature columns"):
        config.validate_dataframe(bad_df, require_label=True)


def test_build_atharv_three_models():
    config = FeatureConfig()

    rf_pipe = build_random_forest_pipeline(config)
    assert isinstance(rf_pipe, Pipeline)
    assert "classifier" in rf_pipe.named_steps
    assert "imputer" in rf_pipe.named_steps

    knn_pipe = build_knn_pipeline(config)
    assert isinstance(knn_pipe, Pipeline)
    assert "scaler" in knn_pipe.named_steps
    assert "classifier" in knn_pipe.named_steps

    dt_pipe = build_decision_tree_pipeline(config)
    assert isinstance(dt_pipe, Pipeline)
    assert "classifier" in dt_pipe.named_steps


def test_trip_group_cv_for_atharv_three_models(
    synthetic_candidate_dataset: pd.DataFrame,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = FeatureConfig()

        # 1. Random Forest
        rf_res = evaluate_random_forest(
            synthetic_candidate_dataset, config=config, n_splits=3, output_dir=tmp_dir
        )
        assert rf_res.model_name == "Random Forest"
        assert rf_res.owner == "Atharv Kulkarni"
        assert len(rf_res.fold_metrics) == 3
        assert 0.0 <= rf_res.oof_metrics["macro_f1"] <= 1.0
        assert 0.0 <= rf_res.oof_metrics["top1_accuracy"] <= 1.0
        assert (Path(tmp_dir) / "atharv_random_forest_results.csv").exists()

        # 2. KNN
        knn_res = evaluate_knn(
            synthetic_candidate_dataset, config=config, n_splits=3, output_dir=tmp_dir
        )
        assert knn_res.model_name == "K-Nearest Neighbours"
        assert knn_res.owner == "Atharv Kulkarni"
        assert len(knn_res.fold_metrics) == 3
        assert 0.0 <= knn_res.oof_metrics["macro_f1"] <= 1.0
        assert (Path(tmp_dir) / "atharv_knn_results.csv").exists()

        # 3. Decision Tree
        dt_res = evaluate_decision_tree(
            synthetic_candidate_dataset, config=config, n_splits=3, output_dir=tmp_dir
        )
        assert dt_res.model_name == "Decision Tree"
        assert dt_res.owner == "Atharv Kulkarni"
        assert len(dt_res.fold_metrics) == 3
        assert 0.0 <= dt_res.oof_metrics["macro_f1"] <= 1.0
        assert (Path(tmp_dir) / "atharv_decision_tree_results.csv").exists()


def test_five_model_comparison_integration(
    synthetic_candidate_dataset: pd.DataFrame,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = FeatureConfig()
        manager = ModelComparisonManager(output_dir=tmp_dir)

        # Register Atharv's 3 models
        rf_res = evaluate_random_forest(
            synthetic_candidate_dataset, config=config, n_splits=3, output_dir=tmp_dir
        )
        knn_res = evaluate_knn(
            synthetic_candidate_dataset, config=config, n_splits=3, output_dir=tmp_dir
        )
        dt_res = evaluate_decision_tree(
            synthetic_candidate_dataset, config=config, n_splits=3, output_dir=tmp_dir
        )

        manager.add_atharv_result(rf_res)
        manager.add_atharv_result(knn_res)
        manager.add_atharv_result(dt_res)

        # Export Atharv's 3-model summary
        atharv_cv_path = manager.export_atharv_cv_results()
        assert atharv_cv_path.exists()

        # Import 2 external teammate model results (simulated actual handoff from teammates)
        manager.import_external_result(
            model_name="Logistic Regression",
            owner="Teammate 1",
            algorithm_family="Linear & Probabilistic",
            key_configuration="C=1.0, solver=lbfgs",
            cv_metrics={
                "macro_f1": 0.8123,
                "accuracy": 0.8450,
                "precision": 0.8200,
                "recall": 0.8050,
                "top1_accuracy": 0.7890,
                "candidate_miss_rate": 0.0210,
                "log_loss": 0.3850,
            },
            train_time_sec=1.25,
            infer_time_sec=0.15,
        )
        manager.import_external_result(
            model_name="HistGradientBoosting",
            owner="Teammate 2",
            algorithm_family="Ensemble (Boosting)",
            key_configuration="max_iter=100, lr=0.1",
            cv_metrics={
                "macro_f1": 0.8845,
                "accuracy": 0.9010,
                "precision": 0.8900,
                "recall": 0.8790,
                "top1_accuracy": 0.8650,
                "candidate_miss_rate": 0.0210,
                "log_loss": 0.2450,
            },
            train_time_sec=3.40,
            infer_time_sec=0.22,
        )

        # Generate unified 5-model comparison
        five_model_table = manager.generate_five_model_comparison()
        assert len(five_model_table) == 5
        assert set(five_model_table["model_name"]) == {
            "Random Forest",
            "K-Nearest Neighbours",
            "Decision Tree",
            "Logistic Regression",
            "HistGradientBoosting",
        }
        assert "model_owner" in five_model_table.columns
        assert "top1_accuracy" in five_model_table.columns

        csv_path = manager.export_five_model_comparison_csv()
        assert csv_path.exists()

        plot_path = manager.plot_comparison(metric="cv_macro_f1")
        assert plot_path.exists()


def test_scoring_interface_and_artifacts(
    synthetic_candidate_dataset: pd.DataFrame,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = FeatureConfig()

        # Fit and save Atharv's 3 artifacts
        rf_pipe = fit_and_save_random_forest(
            synthetic_candidate_dataset, config, artifacts_dir=tmp_dir
        )
        knn_pipe = fit_and_save_knn(
            synthetic_candidate_dataset, config, artifacts_dir=tmp_dir
        )
        dt_pipe = fit_and_save_decision_tree(
            synthetic_candidate_dataset, config, artifacts_dir=tmp_dir
        )

        assert (Path(tmp_dir) / "atharv_random_forest.pkl").exists()
        assert (Path(tmp_dir) / "atharv_knn.pkl").exists()
        assert (Path(tmp_dir) / "atharv_decision_tree.pkl").exists()

        # Test inference scoring contract
        candidates = synthetic_candidate_dataset.drop(columns=["is_matching_candidate"])
        scored = score_candidates(rf_pipe, candidates, config)

        assert "trip_id" in scored.columns
        assert "observation_id" in scored.columns
        assert "arc_id" in scored.columns
        assert "prob_matching" in scored.columns
        assert "pred_matching" in scored.columns
        assert len(scored) == len(candidates)
        assert (scored["prob_matching"] >= 0.0).all()
        assert (scored["prob_matching"] <= 1.0).all()


def test_bayesian_optimization_atharv_models(
    synthetic_candidate_dataset: pd.DataFrame,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = FeatureConfig()
        optimizer = BayesianOptimizer(
            df_train=synthetic_candidate_dataset,
            model_name="random_forest",
            config=config,
            metric_to_optimize="top1_accuracy",
            n_splits=2,
            output_dir=tmp_dir,
        )

        # Run 2 quick trials to test mechanics
        res = optimizer.run_optimization(n_trials=2)
        assert "best_params" in res
        assert "best_metric_value" in res
        assert res["n_completed_trials"] == 2

        trials_csv = optimizer.export_trials_csv()
        assert trials_csv.exists()

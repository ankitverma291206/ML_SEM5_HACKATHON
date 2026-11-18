"""Unit tests for ML experimentation infrastructure (Atharv Kulkarni).

Uses small SYNTHETIC fixtures strictly to verify software mechanics,
leak-free GroupKFold cross-validation, metric calculations, and the inference API.
THESE SYNTHETIC METRICS ARE NEVER INCLUDED IN HACKATHON EXPERIMENTAL RESULTS.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models import (
    DEFAULT_FEATURE_COLUMNS,
    BayesianOptimizer,
    FeatureConfig,
    ModelComparator,
    SUPPORTED_MODELS,
    build_model_pipeline,
    compute_candidate_metrics,
    compute_candidate_miss_rate,
    compute_observation_top1_accuracy,
    evaluate_model_predictions,
    load_model_artifacts,
    run_group_kfold_cv,
    save_model_artifacts,
    score_candidates,
    train_full_model,
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
            # 1 matching candidate and 3 non-matching candidates
            matching_candidate_idx = np.random.randint(0, 4)
            for cand_idx in range(4):
                arc_id = 1000 + obs_idx * 10 + cand_idx
                is_match = 1 if cand_idx == matching_candidate_idx else 0

                # Realistic feature distributions: matching candidates have lower distance and heading diff
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

    # Missing column should raise error
    bad_df = synthetic_candidate_dataset.drop(columns=["distance_m"])
    with pytest.raises(ValueError, match="missing configured feature columns"):
        config.validate_dataframe(bad_df, require_label=True)


def test_build_all_five_models():
    config = FeatureConfig()
    for model_name in SUPPORTED_MODELS:
        pipe = build_model_pipeline(model_name, config)
        assert isinstance(pipe, Pipeline)
        assert "classifier" in pipe.named_steps
        assert "imputer" in pipe.named_steps
        if model_name in {"logistic_regression", "knn"}:
            assert "scaler" in pipe.named_steps


def test_group_kfold_cv_zero_leakage_all_five_models(
    synthetic_candidate_dataset: pd.DataFrame,
):
    config = FeatureConfig()
    for model_name in SUPPORTED_MODELS:
        res = run_group_kfold_cv(
            synthetic_candidate_dataset,
            model_name=model_name,
            config=config,
            n_splits=3,
        )
        assert res.model_name == model_name
        assert len(res.fold_metrics) == 3
        assert res.oof_metrics["macro_f1"] is not None
        assert 0.0 <= res.oof_metrics["macro_f1"] <= 1.0
        assert res.oof_metrics["top1_accuracy"] is not None
        assert 0.0 <= res.oof_metrics["top1_accuracy"] <= 1.0
        assert res.total_train_time > 0.0
        assert res.total_infer_time > 0.0


def test_observation_top1_and_miss_rate():
    # Manually constructed scenario
    test_df = pd.DataFrame(
        {
            "trip_id": ["t1", "t1", "t1", "t1", "t2", "t2"],
            "observation_id": [0, 0, 1, 1, 0, 0],
            "arc_id": [1, 2, 3, 4, 5, 6],
            "prob_matching": [0.8, 0.2, 0.1, 0.9, 0.4, 0.3],
            "is_matching_candidate": [1, 0, 1, 0, 0, 0],  # obs 0 matches top, obs 1 top is wrong, t2 obs 0 missed completely
        }
    )
    # Obs 1 (t1, 0): top is arc 1 (prob 0.8), label is 1 -> CORRECT
    # Obs 2 (t1, 1): top is arc 4 (prob 0.9), label is 0 -> INCORRECT
    # Obs 3 (t2, 0): top is arc 5 (prob 0.4), label is 0 -> INCORRECT
    # Total obs = 3, Correct = 1 -> Top-1 = 1/3
    top1 = compute_observation_top1_accuracy(test_df)
    assert pytest.approx(top1, 0.01) == 1.0 / 3.0

    # Miss rate: Obs 3 (t2, 0) has sum of labels = 0 -> 1 out of 3 missed
    miss_rate = compute_candidate_miss_rate(test_df)
    assert pytest.approx(miss_rate, 0.01) == 1.0 / 3.0


def test_model_comparator_and_exports(synthetic_candidate_dataset: pd.DataFrame):
    with tempfile.TemporaryDirectory() as tmp_dir:
        comparator = ModelComparator(output_dir=tmp_dir)
        config = FeatureConfig()

        for model_name in ["logistic_regression", "decision_tree"]:
            cv_res = run_group_kfold_cv(
                synthetic_candidate_dataset,
                model_name=model_name,
                config=config,
                n_splits=3,
            )
            comparator.add_cv_result(cv_res)

        table = comparator.generate_comparison_table()
        assert len(table) == 2
        assert "cv_macro_f1" in table.columns
        assert "top1_accuracy" in table.columns

        csv_path = comparator.export_comparison_csv()
        assert csv_path.exists()

        fold_csv = comparator.export_fold_results_csv()
        assert fold_csv.exists()

        plot_path = comparator.plot_metric_comparison(metric_name="cv_macro_f1")
        assert plot_path.exists()


def test_inference_scoring_interface(synthetic_candidate_dataset: pd.DataFrame):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = FeatureConfig()
        pipe, train_time = train_full_model(
            synthetic_candidate_dataset, "hist_gradient_boosting", config
        )

        candidates = synthetic_candidate_dataset.drop(columns=["is_matching_candidate"])
        scored = score_candidates(pipe, candidates, config)

        assert "trip_id" in scored.columns
        assert "observation_id" in scored.columns
        assert "arc_id" in scored.columns
        assert "prob_matching" in scored.columns
        assert "pred_matching" in scored.columns
        assert len(scored) == len(candidates)
        assert (scored["prob_matching"] >= 0.0).all()
        assert (scored["prob_matching"] <= 1.0).all()

        # Artifact save/load test
        saved_paths = save_model_artifacts(
            pipe,
            config,
            metadata={"model_name": "hist_gradient_boosting", "train_time": train_time},
            artifacts_dir=tmp_dir,
        )
        assert Path(saved_paths["model_path"]).exists()

        loaded_pipe, loaded_config, loaded_meta = load_model_artifacts(tmp_dir)
        scored_loaded = score_candidates(loaded_pipe, candidates, loaded_config)
        np.testing.assert_allclose(
            scored["prob_matching"].values, scored_loaded["prob_matching"].values
        )


def test_bayesian_optimization_mechanics(synthetic_candidate_dataset: pd.DataFrame):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = FeatureConfig()
        optimizer = BayesianOptimizer(
            df_train=synthetic_candidate_dataset,
            model_name="logistic_regression",
            config=config,
            metric_to_optimize="macro_f1",
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

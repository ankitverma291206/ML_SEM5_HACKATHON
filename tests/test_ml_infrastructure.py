"""Unit tests for Atharv Kulkarni's ML experimentation infrastructure.

Uses small SYNTHETIC fixtures strictly to verify software mechanics,
leak-free GroupKFold cross-validation, candidate coverage with zero-candidate
observations, circular angle encoding, positive class validation, and comparison metadata.

THESE SYNTHETIC METRICS ARE NEVER INCLUDED IN HACKATHON EXPERIMENTAL RESULTS.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.models import (
    BayesianOptimizer,
    FeatureConfig,
    ModelComparisonManager,
    ModelExperimentMetadata,
    build_decision_tree_pipeline,
    build_knn_pipeline,
    build_random_forest_pipeline,
    compute_candidate_metrics,
    compute_candidate_miss_rate,
    compute_observation_top1_accuracy,
    encode_circular_angles,
    evaluate_decision_tree,
    evaluate_knn,
    evaluate_model_predictions,
    evaluate_random_forest,
    fit_and_save_decision_tree,
    fit_and_save_knn,
    fit_and_save_random_forest,
    get_positive_class_index,
    load_model_artifacts,
    run_trip_group_cv,
    save_model_artifacts,
    score_candidates,
    validate_experiment_comparability,
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
                    distance = float(np.random.exponential(scale=5.0))
                    heading_diff = float(np.random.uniform(0.0, 20.0))
                else:
                    distance = float(np.random.exponential(scale=35.0) + 10.0)
                    heading_diff = float(np.random.uniform(15.0, 180.0))

                rows.append(
                    {
                        "trip_id": trip_id,
                        "observation_id": obs_id,
                        "arc_id": arc_id,
                        "distance_m": distance,
                        "projection_position": float(np.random.uniform(0.0, 1.0)),
                        "road_bearing": float(np.random.uniform(0.0, 360.0)),
                        "gps_heading": float(np.random.uniform(0.0, 360.0)),
                        "heading_difference": heading_diff,
                        "gps_speed_mps": float(np.random.uniform(5.0, 25.0)),
                        "time_delta_s": float(np.random.choice([1.0, 2.0, 5.0])),
                        "large_or_invalid_time_gap": 0,
                        "is_matching_candidate": is_match,
                    }
                )

    df = pd.DataFrame(rows)
    # Add a few intentional NaNs in numeric features to test imputer robustness
    df.loc[3, "gps_speed_mps"] = np.nan
    df.loc[7, "heading_difference"] = np.nan
    return df


def test_circular_angular_feature_encoding():
    """Verify that angles near 0° and 360° are continuous and close in engineered feature space."""
    angles = np.array([1.0, 359.0, 180.0])
    sin_comp, cos_comp = encode_circular_angles(angles, period=360.0)

    # Coordinates for 1 deg and 359 deg
    p1 = np.array([sin_comp[0], cos_comp[0]])
    p359 = np.array([sin_comp[1], cos_comp[1]])

    # Linear degree distance would be |1 - 359| = 358 deg
    linear_diff = abs(angles[0] - angles[1])
    assert linear_diff == 358.0

    # Euclidean distance in circular representation must be small (~ 2 * sin(1 deg) ≈ 0.0349)
    circ_dist = np.linalg.norm(p1 - p359)
    assert circ_dist < 0.05
    assert circ_dist == pytest.approx(2.0 * np.sin(np.radians(1.0)), rel=1e-3)


def test_feature_config_leakage_rejection():
    """Verify that config rejects identifier or target leakage."""
    # 1. Identifier leakage rejection
    with pytest.raises(ValueError, match="Identifier leakage hazard"):
        FeatureConfig(feature_columns=["distance_m", "trip_id"])

    # 2. Target label leakage rejection
    with pytest.raises(ValueError, match="Target leakage hazard"):
        FeatureConfig(feature_columns=["distance_m", "is_matching_candidate"])


def test_feature_config_dataframe_validation(synthetic_candidate_dataset: pd.DataFrame):
    """Verify dataframe validation catches missing IDs, nulls, non-numeric types, non-binary labels."""
    config = FeatureConfig()

    # Valid dataframe passes
    config.validate_dataframe(synthetic_candidate_dataset, require_label=True)

    # Null in identifier
    bad_id_df = synthetic_candidate_dataset.copy()
    bad_id_df.loc[0, "trip_id"] = np.nan
    with pytest.raises(ValueError, match="contains 1 null values"):
        config.validate_dataframe(bad_id_df)

    # Non-numeric predictive feature
    bad_type_df = synthetic_candidate_dataset.copy()
    bad_type_df["distance_m"] = bad_type_df["distance_m"].astype(str)
    with pytest.raises(TypeError, match="has non-numeric dtype"):
        config.validate_dataframe(bad_type_df)

    # Non-binary target labels
    bad_label_df = synthetic_candidate_dataset.copy()
    bad_label_df.loc[0, "is_matching_candidate"] = 2
    with pytest.raises(ValueError, match="contains non-binary values"):
        config.validate_dataframe(bad_label_df, require_label=True)


def test_candidate_coverage_with_zero_candidate_observations():
    """Verify candidate miss-rate denominator with complete observation inventory."""
    # 5 total GPS observations in raw track inventory: (t1, 0), (t1, 1), (t1, 2), (t1, 3), (t1, 4)
    raw_inventory = pd.DataFrame(
        {
            "trip_id": ["t1"] * 5,
            "observation_id": [0, 1, 2, 3, 4],
        }
    )

    # Candidate table only contains candidates for obs 0, 1, 2 (obs 3 and 4 had ZERO candidates!)
    candidate_df = pd.DataFrame(
        {
            "trip_id": ["t1", "t1", "t1", "t1", "t1"],
            "observation_id": [0, 0, 1, 1, 2],
            "arc_id": [10, 11, 20, 21, 30],
            "is_matching_candidate": [1, 0, 0, 0, 1],  # obs 0 and 2 have matches, obs 1 has no match
            "prob_matching": [0.9, 0.1, 0.4, 0.6, 0.8],
        }
    )

    # Without inventory: only looks at obs 0, 1, 2 (3 obs total, 1 missed -> miss rate = 1/3)
    miss_rate_no_inv = compute_candidate_miss_rate(candidate_df)
    assert pytest.approx(miss_rate_no_inv, 0.01) == 1.0 / 3.0

    # With raw inventory: 5 total obs:
    # Obs 0: has match (1)
    # Obs 1: candidates proposed, but none match (miss)
    # Obs 2: has match (1)
    # Obs 3: 0 candidates generated (miss)
    # Obs 4: 0 candidates generated (miss)
    # Total missed = 3 out of 5 -> miss rate = 3 / 5 = 0.6
    miss_rate_with_inv = compute_candidate_miss_rate(
        candidate_df, all_observations=raw_inventory
    )
    assert pytest.approx(miss_rate_with_inv, 0.01) == 0.60

    # Top-1 accuracy with raw inventory: 2 correct out of 5 total obs = 0.40
    top1_with_inv = compute_observation_top1_accuracy(
        candidate_df, all_observations=raw_inventory
    )
    assert pytest.approx(top1_with_inv, 0.01) == 0.40


def test_positive_class_index_validation():
    """Verify positive class index validation and error handling."""
    # Unfitted pipeline
    pipe = Pipeline([("classifier", DecisionTreeClassifier())])
    with pytest.raises(AttributeError, match="has no 'classes_' attribute"):
        get_positive_class_index(pipe, pos_label=1)

    # Fit with only single class [0]
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 0])
    pipe.fit(X, y)

    # Requesting class 1 when only 0 is present must raise ValueError (not silently return 1)
    with pytest.raises(ValueError, match="Requested positive class '1' was not found"):
        get_positive_class_index(pipe, pos_label=1)

    # Fit with binary classes [0, 1]
    y_binary = np.array([0, 1])
    pipe.fit(X, y_binary)
    assert get_positive_class_index(pipe, pos_label=1) == 1


def test_trip_group_cv_and_inference_timing(synthetic_candidate_dataset: pd.DataFrame):
    """Verify GroupKFold CV measures inference timing encompassing predict and predict_proba."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = FeatureConfig()
        rf_res = evaluate_random_forest(
            synthetic_candidate_dataset, config=config, n_splits=3, output_dir=tmp_dir
        )

        assert rf_res.total_infer_time > 0.0
        for f in rf_res.fold_metrics:
            assert f["infer_time"] > 0.0
            assert f["train_time"] > 0.0


def test_cross_model_comparison_metadata():
    """Verify metadata validation for external model integration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = ModelComparisonManager(output_dir=tmp_dir)

        # Comparable external result
        valid_meta = ModelExperimentMetadata(
            model_name="SGDClassifier",
            owner="Teammate 1",
            algorithm_family="Linear & Probabilistic",
            key_configuration="loss=log_loss, alpha=0.001",
            dataset_partition="train_80_trips",
            cv_splitter="GroupKFold(5)",
            shared_fold_assignments=True,
            train_time_sec=1.1,
            infer_time_sec=0.2,
        )
        is_comp = manager.import_external_result(
            model_name="SGDClassifier",
            owner="Teammate 1",
            algorithm_family="Linear & Probabilistic",
            key_configuration="loss=log_loss",
            cv_metrics={"macro_f1": 0.82, "top1_accuracy": 0.79},
            metadata=valid_meta,
        )
        assert is_comp is True

        # Incomparable external result (e.g. trained on wrong partition)
        invalid_meta = ModelExperimentMetadata(
            model_name="TestModel",
            owner="Teammate 2",
            algorithm_family="Custom",
            key_configuration="default",
            dataset_partition="random_unsplit",
            cv_splitter="KFold(5)",
            shared_fold_assignments=False,
            train_time_sec=None,  # Explicitly None, not 0.0
        )
        is_comp_invalid = manager.import_external_result(
            model_name="TestModel",
            owner="Teammate 2",
            algorithm_family="Custom",
            key_configuration="default",
            cv_metrics={"macro_f1": 0.85},
            metadata=invalid_meta,
        )
        assert is_comp_invalid is False

        # Table preserves metadata and does not default missing time to 0.0
        table = manager.generate_five_model_comparison()
        test_model_row = table[table["model_name"] == "TestModel"].iloc[0]
        assert bool(test_model_row["is_comparable"]) is False
        assert pd.isna(test_model_row["train_time_sec"])


def test_bayesian_optimization_gatekeeper_and_oof(synthetic_candidate_dataset: pd.DataFrame):
    """Verify Optuna optimization gatekeeper and OOF pooled objective."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = FeatureConfig()
        optimizer = BayesianOptimizer(
            df_train=synthetic_candidate_dataset,
            model_name="decision_tree",
            config=config,
            metric_to_optimize="top1_accuracy",
            n_splits=2,
            output_dir=tmp_dir,
        )

        # Calling run_optimization before 5-model comparison must raise RuntimeError
        with pytest.raises(RuntimeError, match="Bayesian optimization is locked"):
            optimizer.run_optimization(n_trials=2)

        # Unlocking optimization executes successfully with pooled OOF objective
        optimizer.enable_optimization()
        res = optimizer.run_optimization(n_trials=2)
        assert "best_params" in res
        assert "best_metric_value" in res
        assert res["n_completed_trials"] == 2


def test_build_and_score_atharv_three_models(synthetic_candidate_dataset: pd.DataFrame):
    """Verify model building, artifact persistence, and scoring interface for Arya and Ankit."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = FeatureConfig()

        rf_pipe = fit_and_save_random_forest(synthetic_candidate_dataset, config, artifacts_dir=tmp_dir)
        knn_pipe = fit_and_save_knn(synthetic_candidate_dataset, config, artifacts_dir=tmp_dir)
        dt_pipe = fit_and_save_decision_tree(synthetic_candidate_dataset, config, artifacts_dir=tmp_dir)

        assert (Path(tmp_dir) / "atharv_random_forest.pkl").exists()
        assert (Path(tmp_dir) / "atharv_knn.pkl").exists()
        assert (Path(tmp_dir) / "atharv_decision_tree.pkl").exists()

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


def test_feature_leakage_rejection_reference_columns():
    """Verify that reference alignment metadata (aligned_arc_id, etc.) are rejected as features."""
    for ref_col in ["aligned_arc_id", "aligned_route_position", "alignment_distance_m"]:
        with pytest.raises(ValueError, match="Reference alignment leakage hazard"):
            FeatureConfig(feature_columns=["distance_m", ref_col])


def test_schema_compatibility_with_real_trip_data():
    """Verify that real labelled candidates generated from trip 00000001 validate against FeatureConfig."""
    sample_csv = Path("experiments/labelled_candidates_00000001.csv")
    if not sample_csv.exists():
        pytest.skip("experiments/labelled_candidates_00000001.csv not present.")

    df_sample = pd.read_csv(sample_csv, nrows=100)
    config = FeatureConfig()
    # Should validate without any schema or type errors
    config.validate_dataframe(df_sample, require_label=True)
    assert set(config.feature_columns).issubset(set(df_sample.columns))
    assert set(config.identifier_columns).issubset(set(df_sample.columns))
    assert config.label_column in df_sample.columns


def test_missing_positive_candidate_coverage_accounting():
    """Verify that observations without any positive candidate are quantified as misses."""
    df_obs = pd.DataFrame(
        {
            "trip_id": ["00000001"] * 4,
            "observation_id": [0, 0, 1, 1],
            "arc_id": [10, 11, 20, 21],
            "is_matching_candidate": [1, 0, 0, 0],  # obs 0 has positive candidate, obs 1 has NO positive candidate
            "prob_matching": [0.8, 0.2, 0.6, 0.4],
            "distance_m": [1.5, 12.0, 5.0, 18.0],
            "projection_position": [0.1, 0.5, 0.2, 0.9],
            "road_bearing": [10.0, 20.0, 30.0, 40.0],
            "gps_heading": [12.0, 12.0, 35.0, 35.0],
            "heading_difference": [2.0, 8.0, 5.0, 5.0],
            "gps_speed_mps": [10.0, 10.0, 12.0, 12.0],
            "time_delta_s": [1.0, 1.0, 1.0, 1.0],
            "large_or_invalid_time_gap": [0, 0, 0, 0],
        }
    )

    # obs 0 has positive candidate, obs 1 does not
    miss_rate = compute_candidate_miss_rate(df_obs)
    assert miss_rate == 0.5  # 1 of 2 observations lacks a positive candidate



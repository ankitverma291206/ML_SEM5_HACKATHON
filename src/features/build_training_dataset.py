"""Reproducible generation and comprehensive audit of the 80-trip training dataset.

Processes all 80 training trips through Aum's candidate generation and sequence-aware
route alignment pipeline. Generates:
- Complete training candidate dataset (saved locally to data/processed/train_candidates.parquet)
- Per-trip candidate coverage and positive-label audit (experiments/candidate_coverage_audit.csv)
- Label distribution and class imbalance report (experiments/label_distribution_summary.csv)
- High alignment distance and missing positive candidate investigation (experiments/high_alignment_distance_investigation.csv)
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.data.load_trip import load_trip
from src.features.candidates import generate_candidates
from src.features.labels import align_track_to_route, label_candidates


def audit_single_trip(
    trip_id: str,
    trip_dir: Path,
    radius_m: float = 50.0,
    max_candidates: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """Process one trip and collect audit diagnostics."""
    data = load_trip(trip_dir, trip_id)
    n_obs = len(data["track"])

    # 1. Candidate generation
    candidates = generate_candidates(
        data, trip_id=trip_id, radius_m=radius_m, max_candidates=max_candidates
    )

    # 2. Sequence-aware alignment
    alignment = align_track_to_route(data)

    # 3. Label assignment
    labelled = label_candidates(candidates, alignment)

    # Convert large_or_invalid_time_gap to integer 0/1 for consistent numerical modeling
    if "large_or_invalid_time_gap" in labelled.columns:
        labelled["large_or_invalid_time_gap"] = labelled["large_or_invalid_time_gap"].astype("int8")

    # Diagnostic calculations
    align_summary = alignment.attrs.get("summary", {})
    align_distances = alignment["alignment_distance_m"].to_numpy()

    # Per-observation matching stats
    obs_grouped = labelled.groupby("observation_id")["is_matching_candidate"].agg(["count", "max"])
    obs_with_cand = len(obs_grouped)
    obs_with_pos = int((obs_grouped["max"] == 1).sum()) if obs_with_cand > 0 else 0
    obs_without_pos = n_obs - obs_with_pos

    n_pos_rows = int((labelled["is_matching_candidate"] == 1).sum())
    n_neg_rows = int((labelled["is_matching_candidate"] == 0).sum())
    n_total_rows = len(labelled)

    imbalance_ratio = float(n_neg_rows / n_pos_rows) if n_pos_rows > 0 else float("inf")

    # Duplicate check on (trip_id, observation_id, arc_id)
    n_duplicates = int(labelled.duplicated(subset=["trip_id", "observation_id", "arc_id"]).sum())

    # High alignment distance observations (> 50m)
    high_dist_obs = alignment[alignment["alignment_distance_m"] > radius_m]
    high_dist_records = []
    for _, h_row in high_dist_obs.iterrows():
        high_dist_records.append(
            {
                "trip_id": trip_id,
                "observation_id": int(h_row["observation_id"]),
                "aligned_arc_id": int(h_row["aligned_arc_id"]),
                "alignment_distance_m": float(h_row["alignment_distance_m"]),
                "had_candidates": int(h_row["observation_id"] in obs_grouped.index),
                "has_positive_candidate": int(
                    h_row["observation_id"] in obs_grouped.index
                    and obs_grouped.loc[h_row["observation_id"], "max"] == 1
                ),
            }
        )

    trip_audit = {
        "trip_id": trip_id,
        "total_gps_observations": n_obs,
        "observations_with_candidates": obs_with_cand,
        "candidate_coverage_pct": round(100.0 * obs_with_cand / n_obs, 2) if n_obs > 0 else 0.0,
        "observations_with_positive_candidate": obs_with_pos,
        "positive_coverage_pct": round(100.0 * obs_with_pos / n_obs, 2) if n_obs > 0 else 0.0,
        "observations_without_positive_candidate": obs_without_pos,
        "total_candidate_rows": n_total_rows,
        "positive_candidate_rows": n_pos_rows,
        "negative_candidate_rows": n_neg_rows,
        "class_imbalance_ratio": round(imbalance_ratio, 2) if np.isfinite(imbalance_ratio) else -1.0,
        "duplicate_candidate_rows": n_duplicates,
        "mean_alignment_distance_m": round(float(np.mean(align_distances)), 2),
        "median_alignment_distance_m": round(float(np.median(align_distances)), 2),
        "p95_alignment_distance_m": round(float(np.percentile(align_distances, 95)), 2),
        "max_alignment_distance_m": round(float(np.max(align_distances)), 2),
        "observations_alignment_gt_50m": len(high_dist_obs),
    }

    return labelled, trip_audit, high_dist_records


def build_and_audit_training_dataset(
    train_dir: Path = Path("Student folder/train"),
    output_data_dir: Path = Path("data/processed"),
    output_exp_dir: Path = Path("experiments"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process all 80 training trips, generate dataset, and save audit reports."""
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_exp_dir.mkdir(parents=True, exist_ok=True)

    trip_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    print(f"Starting audit and dataset generation for {len(trip_dirs)} training trips...")

    all_labelled_dfs: List[pd.DataFrame] = []
    audit_rows: List[Dict[str, Any]] = []
    all_high_dist_records: List[Dict[str, Any]] = []

    t_start = time.perf_counter()

    for idx, trip_path in enumerate(trip_dirs):
        trip_id = trip_path.name
        t0 = time.perf_counter()
        labelled_df, audit_info, high_dist = audit_single_trip(trip_id, trip_path)
        elapsed = time.perf_counter() - t0

        all_labelled_dfs.append(labelled_df)
        audit_rows.append(audit_info)
        all_high_dist_records.extend(high_dist)

        if (idx + 1) % 10 == 0 or idx == len(trip_dirs) - 1:
            print(
                f"[{idx+1:02d}/{len(trip_dirs)}] Trip {trip_id}: "
                f"{audit_info['total_gps_observations']} obs, "
                f"{audit_info['total_candidate_rows']} candidates, "
                f"Pos coverage: {audit_info['positive_coverage_pct']}%, "
                f"Elapsed: {elapsed:.2f}s"
            )

    total_time = time.perf_counter() - t_start
    print(f"\nCompleted processing 80 trips in {total_time:.2f}s ({total_time/60:.2f} min).")

    # Combine into single training candidate dataset
    full_train_df = pd.concat(all_labelled_dfs, ignore_index=True)
    audit_df = pd.DataFrame(audit_rows)
    high_dist_df = pd.DataFrame(all_high_dist_records)

    # 1. Save local full dataset (parquet + compressed csv)
    parquet_path = output_data_dir / "train_candidates.parquet"
    try:
        full_train_df.to_parquet(parquet_path, index=False)
        print(f"Saved full training dataset to {parquet_path} ({len(full_train_df):,} rows)")
    except Exception as e:
        print(f"Parquet export failed ({e}), saving compressed CSV...")
    
    csv_gz_path = output_data_dir / "train_candidates.csv.gz"
    full_train_df.to_csv(csv_gz_path, index=False, compression="gzip")
    print(f"Saved compressed CSV dataset to {csv_gz_path}")

    # 2. Save per-trip candidate coverage audit report
    audit_path = output_exp_dir / "candidate_coverage_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"Saved coverage audit to {audit_path}")

    # 3. Save high alignment distance investigation report
    if not high_dist_df.empty:
        high_dist_path = output_exp_dir / "high_alignment_distance_investigation.csv"
        high_dist_df.to_csv(high_dist_path, index=False)
        print(f"Saved high alignment distance investigation to {high_dist_path} ({len(high_dist_df):,} rows)")

    # 4. Save label distribution summary report
    tot_obs = audit_df["total_gps_observations"].sum()
    tot_cand_obs = audit_df["observations_with_candidates"].sum()
    tot_pos_obs = audit_df["observations_with_positive_candidate"].sum()
    tot_missing_pos_obs = audit_df["observations_without_positive_candidate"].sum()
    tot_cand_rows = audit_df["total_candidate_rows"].sum()
    tot_pos_rows = audit_df["positive_candidate_rows"].sum()
    tot_neg_rows = audit_df["negative_candidate_rows"].sum()
    overall_imbalance = tot_neg_rows / tot_pos_rows if tot_pos_rows > 0 else float("inf")

    label_summary_df = pd.DataFrame(
        [
            {"metric": "total_training_trips", "value": len(audit_df)},
            {"metric": "total_gps_observations", "value": int(tot_obs)},
            {"metric": "observations_with_candidates", "value": int(tot_cand_obs)},
            {"metric": "overall_candidate_coverage_pct", "value": round(100.0 * tot_cand_obs / tot_obs, 2)},
            {"metric": "observations_with_positive_candidate", "value": int(tot_pos_obs)},
            {"metric": "overall_positive_coverage_pct", "value": round(100.0 * tot_pos_obs / tot_obs, 2)},
            {"metric": "observations_without_positive_candidate", "value": int(tot_missing_pos_obs)},
            {"metric": "overall_candidate_miss_rate", "value": round(tot_missing_pos_obs / tot_obs, 4)},
            {"metric": "total_candidate_rows", "value": int(tot_cand_rows)},
            {"metric": "positive_candidate_rows", "value": int(tot_pos_rows)},
            {"metric": "negative_candidate_rows", "value": int(tot_neg_rows)},
            {"metric": "overall_class_imbalance_ratio", "value": round(overall_imbalance, 2)},
            {"metric": "positive_class_percentage", "value": round(100.0 * tot_pos_rows / tot_cand_rows, 2)},
            {"metric": "negative_class_percentage", "value": round(100.0 * tot_neg_rows / tot_cand_rows, 2)},
            {"metric": "total_duplicate_rows", "value": int(audit_df['duplicate_candidate_rows'].sum())},
            {"metric": "mean_p95_alignment_distance_m", "value": round(float(audit_df['p95_alignment_distance_m'].mean()), 2)},
            {"metric": "total_observations_alignment_gt_50m", "value": int(audit_df['observations_alignment_gt_50m'].sum())},
        ]
    )
    label_summary_path = output_exp_dir / "label_distribution_summary.csv"
    label_summary_df.to_csv(label_summary_path, index=False)
    print(f"Saved label distribution summary to {label_summary_path}")

    return full_train_df, audit_df


if __name__ == "__main__":
    build_and_audit_training_dataset()

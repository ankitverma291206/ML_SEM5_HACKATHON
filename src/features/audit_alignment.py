"""Audit candidate coverage and sequence-alignment labels across selected trips."""

from pathlib import Path
import pandas as pd

from src.data.load_trip import load_trip
from src.features.candidates import generate_candidates
from src.features.labels import align_track_to_route, label_candidates

ROOT = Path("data/Student folder/train")
TRIP_IDS = [
    "00000001", "00000002", "00000019",
    "00000033", "00000042", "00000069"
]

summaries = []
all_missing = []

for trip_id in TRIP_IDS:
    print(f"\n========== TRIP {trip_id} ==========", flush=True)

    try:
        data = load_trip(ROOT / trip_id, trip_id)
        candidates = generate_candidates(data, trip_id)
        alignment = align_track_to_route(data)
        labelled = label_candidates(candidates, alignment)

        n_obs = len(data["track"])
        positive_counts = (
            labelled.groupby("observation_id")["is_matching_candidate"]
            .sum()
            .reindex(range(n_obs), fill_value=0)
        )
        n_positive = int((positive_counts > 0).sum())
        n_missing = int((positive_counts == 0).sum())

        # For each observation, check whether the aligned arc was in
        # the generated candidate list, regardless of its distance rank.
        candidate_arc_sets = labelled.groupby("observation_id")["arc_id"].agg(set)
        aligned_by_obs = alignment.set_index("observation_id")["aligned_arc_id"]

        missing_ids = positive_counts[positive_counts == 0].index
        for obs_id in missing_ids:
            rows = labelled[labelled.observation_id == obs_id]
            all_missing.append({
                "trip_id": trip_id,
                "observation_id": int(obs_id),
                "aligned_arc_id": int(aligned_by_obs.loc[obs_id]),
                "alignment_distance_m": float(
                    alignment.loc[
                        alignment.observation_id == obs_id,
                        "alignment_distance_m"
                    ].iloc[0]
                ),
                "nearest_candidate_distance_m": (
                    float(rows.distance_m.min()) if len(rows) else None
                ),
                "candidate_count": len(rows),
            })

        summary = {
            "trip_id": trip_id,
            "gps_observations": n_obs,
            "route_positions": len(data["route"]),
            "candidate_rows": len(candidates),
            "candidate_coverage": candidates.attrs["summary"]["candidate_coverage"],
            "observations_with_positive": n_positive,
            "observations_without_positive": n_missing,
            "positive_label_rate_per_observation": n_positive / n_obs,
            "mean_alignment_distance_m": alignment.attrs["summary"]["mean_alignment_distance_m"],
            "median_alignment_distance_m": alignment.attrs["summary"]["median_alignment_distance_m"],
            "p95_alignment_distance_m": alignment.attrs["summary"]["p95_alignment_distance_m"],
            "fraction_alignment_within_25m": alignment.attrs["summary"]["fraction_within_25m"],
            "monotonic_alignment": alignment.attrs["summary"]["monotonic_alignment"],
            "degenerate_route_arcs": alignment.attrs["summary"].get("degenerate_route_arcs"),
        }
        summaries.append(summary)

        print("Candidate coverage:", f'{summary["candidate_coverage"]:.3%}')
        print("Positive observations:", f'{n_positive}/{n_obs}')
        print("Missing positive observations:", n_missing)
        print("Median alignment distance:", f'{summary["median_alignment_distance_m"]:.2f} m')
        print("95th percentile alignment distance:", f'{summary["p95_alignment_distance_m"]:.2f} m')
        print("Monotonic:", summary["monotonic_alignment"])
        print("Degenerate route arcs:", summary["degenerate_route_arcs"])

        # Save per-trip outputs for reproducibility.
        labelled.to_csv(
            Path("experiments") / f"labelled_candidates_{trip_id}.csv",
            index=False
        )
        alignment.to_csv(
            Path("experiments") / f"alignment_{trip_id}.csv",
            index=False
        )

    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        summaries.append({
            "trip_id": trip_id,
            "error": f"{type(exc).__name__}: {exc}"
        })

Path("experiments").mkdir(exist_ok=True)
pd.DataFrame(summaries).to_csv(
    "experiments/multi_trip_alignment_audit.csv", index=False
)
pd.DataFrame(all_missing).to_csv(
    "experiments/missing_positive_observations.csv", index=False
)

print("\n========== AUDIT COMPLETE ==========")
print(pd.DataFrame(summaries).to_string(index=False))
print("\nSaved experiments/multi_trip_alignment_audit.csv")
print("Saved experiments/missing_positive_observations.csv")

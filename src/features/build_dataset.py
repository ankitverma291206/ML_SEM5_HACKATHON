"""Build the labelled candidate-feature table for ALL trips (resumable).

Run from the repository root:

    python -m src.features.build_dataset                    # all 80 train + 10 student_test
    python -m src.features.build_dataset --trips 00000033   # a subset
    python -m src.features.build_dataset --force            # ignore cached per-trip files

Outputs (under experiments/):
    features/<trip_id>.csv.gz          per-trip table (cached; delete or --force to rebuild)
    features/<trip_id>.report.json     per-trip candidate-miss summary
    candidate_features_all.csv.gz      everything concatenated
    candidate_miss_report.csv          one row per trip (miss rate, alignment quality)
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from src.data.load_trip import load_trip
from src.features.candidates import generate_candidates
from src.features.finalize import (
    FEATURE_COLUMNS, add_features, candidate_report,
)
from src.features.labels import align_track_to_route, label_candidates

PARTITIONS = ("train", "student_test")


def find_root(arg):
    """Dataset root: --root if given, else 'Student folder' or 'data/Student folder'."""
    candidates = [Path(arg)] if arg else [Path("Student folder"), Path("data/Student folder")]
    for c in candidates:
        if c.is_dir():
            return c
    raise SystemExit(
        "Dataset folder not found. Pass --root <path to 'Student folder'> "
        f"(looked for: {', '.join(map(str, candidates))})"
    )


def process_trip(trip_dir, partition, out_dir, radius_m, max_candidates, force):
    trip_id = trip_dir.name
    table_path = out_dir / f"{trip_id}.csv.gz"
    report_path = out_dir / f"{trip_id}.report.json"

    if table_path.exists() and report_path.exists() and not force:
        return json.loads(report_path.read_text()), True

    data = load_trip(trip_dir, trip_id)
    n_obs = len(data["track"])

    candidates = generate_candidates(
        data, trip_id, radius_m=radius_m, max_candidates=max_candidates
    )
    alignment = align_track_to_route(data)
    labelled = label_candidates(candidates, alignment)
    table = add_features(labelled, alignment, data)
    table.insert(0, "partition", partition)

    report = candidate_report(table, alignment, n_obs, trip_id, partition)

    table.to_csv(table_path, index=False, compression="gzip")
    report_path.write_text(json.dumps(report))
    return report, False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="path to the 'Student folder' dataset directory")
    ap.add_argument("--partitions", nargs="+", default=list(PARTITIONS))
    ap.add_argument("--trips", nargs="*", help="only these trip ids")
    ap.add_argument("--radius", type=float, default=50.0, help="candidate search radius, metres")
    ap.add_argument("--max-candidates", type=int, default=24, help="max candidates per observation")
    ap.add_argument("--out", default="experiments", help="output directory")
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()

    root = find_root(args.root)
    out = Path(args.out)
    feat_dir = out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    reports, failures = [], []
    t0 = time.time()
    for partition in args.partitions:
        pdir = root / partition
        for trip_dir in sorted(p for p in pdir.iterdir() if p.is_dir()):
            if args.trips and trip_dir.name not in args.trips:
                continue
            t1 = time.time()
            try:
                rep, cached = process_trip(
                    trip_dir, partition, feat_dir, args.radius, args.max_candidates, args.force
                )
                reports.append(rep)
                print(
                    f"[{partition}] {trip_dir.name}: {rep['gps_observations']:>6} obs, "
                    f"{rep['candidate_rows']:>7} rows, miss {rep['candidate_miss_rate']:.1%} "
                    f"({'cached' if cached else f'{time.time() - t1:.1f}s'})",
                    flush=True,
                )
            except Exception as exc:  # keep going; report at the end
                failures.append((partition, trip_dir.name, f"{type(exc).__name__}: {exc}"))
                print(f"[{partition}] {trip_dir.name}: FAILED {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()

    if not reports:
        raise SystemExit("No trips processed.")

    report_df = pd.DataFrame(reports)
    report_df.to_csv(out / "candidate_miss_report.csv", index=False)

    tables = [
        pd.read_csv(feat_dir / f"{r['trip_id']}.csv.gz", dtype={"trip_id": str})
        for r in reports
    ]
    allrows = pd.concat(tables, ignore_index=True)
    allrows.to_csv(out / "candidate_features_all.csv.gz", index=False, compression="gzip")

    print("\n================ SUMMARY ================")
    print(f"trips: {len(reports)} ok, {len(failures)} failed   ({time.time() - t0:.0f}s)")
    print(f"candidate rows: {len(allrows):,}   features: {len(FEATURE_COLUMNS)}")
    for part, g in report_df.groupby("partition"):
        n = g["gps_observations"].sum()
        pos = g["observations_with_positive"].sum()
        print(f"{part:13s} obs {n:>7,}  with true arc among candidates {pos / n:.1%}  (miss {1 - pos / n:.1%})")
    bad = report_df[report_df["candidate_miss_rate"] > 0.10]
    if len(bad):
        print("\nTrips with >10% candidate miss (check these; exclude/mask for training):")
        print(bad[["trip_id", "gps_observations", "candidate_miss_rate", "share_aligned_over_50m"]]
              .round(3).to_string(index=False))
    if failures:
        print("\nFAILED trips:")
        for f in failures:
            print(" ", f)
    print(f"\nSaved {out / 'candidate_features_all.csv.gz'} and {out / 'candidate_miss_report.csv'}")
    print("Train with:  from src.features.finalize import FEATURE_COLUMNS, training_view")


if __name__ == "__main__":
    main()

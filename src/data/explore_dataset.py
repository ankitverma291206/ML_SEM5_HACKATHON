"""Exploratory data analysis for the PathSnap training dataset."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.load_trip import load_trip


EARTH_RADIUS_M = 6_371_000


def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate distance in metres between two longitude/latitude points."""
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def analyse_trip(trip_dir: Path) -> dict:
    """Calculate summary statistics for one training trip."""
    trip_id = trip_dir.name
    data = load_trip(trip_dir, trip_id)

    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]
    route = data["route"]

    timestamps = track["timestamp_s"].to_numpy(dtype=float)
    intervals = np.diff(timestamps)

    if len(timestamps) >= 2:
        duration_s = float(timestamps[-1] - timestamps[0])
    else:
        duration_s = np.nan

    if len(intervals) > 0:
        mean_interval_s = float(np.mean(intervals))
        median_interval_s = float(np.median(intervals))
    else:
        mean_interval_s = np.nan
        median_interval_s = np.nan

    if len(track) >= 2:
        distances = haversine_distance(
            track["longitude"].to_numpy()[:-1],
            track["latitude"].to_numpy()[:-1],
            track["longitude"].to_numpy()[1:],
            track["latitude"].to_numpy()[1:],
        )
        gps_distance_m = float(np.sum(distances))
    else:
        gps_distance_m = 0.0

    # Count incoming and outgoing directed arcs for each road-network node.
    if len(arcs) > 0 and len(nodes) > 0:
        from_ids = arcs["from_node_id"].to_numpy(dtype=int)
        to_ids = arcs["to_node_id"].to_numpy(dtype=int)

        out_degree = np.bincount(from_ids, minlength=len(nodes))
        in_degree = np.bincount(to_ids, minlength=len(nodes))

        mean_in_degree = float(np.mean(in_degree))
        mean_out_degree = float(np.mean(out_degree))
        isolated_nodes = int(np.sum((in_degree + out_degree) == 0))
    else:
        mean_in_degree = 0.0
        mean_out_degree = 0.0
        isolated_nodes = len(nodes)

    return {
        "trip_id": trip_id,
        "gps_points": len(track),
        "road_nodes": len(nodes),
        "directed_arcs": len(arcs),
        "reference_route_arcs": len(route),
        "duration_s": duration_s,
        "mean_sampling_interval_s": mean_interval_s,
        "median_sampling_interval_s": median_interval_s,
        "gps_trajectory_distance_m": gps_distance_m,
        "mean_in_degree": mean_in_degree,
        "mean_out_degree": mean_out_degree,
        "isolated_nodes": isolated_nodes,
    }


def create_plots(summary: pd.DataFrame, output_dir: Path):
    """Save basic distributions for the training dataset."""
    plots = [
        ("gps_points", "GPS points per trip", "Number of GPS points",
         "gps_points_distribution.png"),
        ("duration_s", "Trip duration distribution", "Duration (seconds)",
         "trip_duration_distribution.png"),
        ("gps_trajectory_distance_m", "GPS trajectory distance distribution",
         "Distance (metres)", "gps_distance_distribution.png"),
        ("road_nodes", "Road network nodes per trip", "Number of nodes",
         "road_nodes_distribution.png"),
        ("directed_arcs", "Directed road arcs per trip", "Number of arcs",
         "road_arcs_distribution.png"),
    ]

    for column, title, xlabel, filename in plots:
        values = summary[column].dropna()

        if values.empty:
            print(f"Skipping empty plot: {filename}")
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=15, edgecolor="black")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Number of trips")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Explore the PathSnap training dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/Student folder"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments"),
    )
    args = parser.parse_args()

    train_dir = args.dataset_root / "train"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    trip_dirs = sorted(
        path for path in train_dir.iterdir() if path.is_dir()
    )

    if not trip_dirs:
        raise RuntimeError(f"No training trips found in {train_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, trip_dir in enumerate(trip_dirs, start=1):
        rows.append(analyse_trip(trip_dir))

        if index % 10 == 0 or index == len(trip_dirs):
            print(f"Analysed {index}/{len(trip_dirs)} training trips")

    summary = pd.DataFrame(rows)

    csv_path = args.output_dir / "training_eda_summary.csv"
    summary.to_csv(csv_path, index=False)

    numeric_summary = summary.drop(columns=["trip_id"]).describe().T
    numeric_path = args.output_dir / "training_eda_statistics.csv"
    numeric_summary.to_csv(numeric_path)

    create_plots(summary, args.output_dir)

    print("\n===== TRAINING DATASET EDA =====")
    print(f"Trips analysed: {len(summary)}")
    print(f"Total GPS points: {summary['gps_points'].sum():,}")
    print(f"Total road nodes: {summary['road_nodes'].sum():,}")
    print(f"Total directed arcs: {summary['directed_arcs'].sum():,}")
    print(
        "Total reference-route arcs: "
        f"{summary['reference_route_arcs'].sum():,}"
    )

    print("\n===== NUMERICAL SUMMARY =====")
    print(numeric_summary.round(2).to_string())

    print("\n===== OUTPUT FILES =====")
    print(f"Trip-level report: {csv_path}")
    print(f"Statistical report: {numeric_path}")
    print(f"Charts directory: {args.output_dir}")


if __name__ == "__main__":
    main()

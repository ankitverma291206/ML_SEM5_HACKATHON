"""Sequence-aware GPS-to-reference-route alignment and candidate labels."""

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import LineString, Point, distance

from src.features.candidates import generate_candidates


def align_track_to_route(data, skip_penalty_m=2.0):
    """Align GPS observations monotonically to ordered route arc positions."""

    track = data["track"].reset_index(drop=True)
    nodes = data["nodes"]
    arcs = data["arcs"]
    route_ids = data["route"]["arc_id"].to_numpy(dtype=int)

    if len(track) == 0 or len(route_ids) == 0:
        raise ValueError("Track and reference route must be non-empty.")

    if (
        (route_ids < 0).any()
        or (route_ids >= len(arcs)).any()
    ):
        raise ValueError("Invalid arc ID in reference route.")

    lon = track["longitude"].to_numpy(float)
    lat = track["latitude"].to_numpy(float)

    centre_lon = float(np.median(lon))
    centre_lat = float(np.median(lat))
    zone = int(np.floor((centre_lon + 180.0) / 6.0) + 1)
    epsg = (32600 if centre_lat >= 0 else 32700) + zone

    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )

    gps_x, gps_y = transformer.transform(lon, lat)

    node_lon = nodes["longitude"].to_numpy(float)
    node_lat = nodes["latitude"].to_numpy(float)
    node_x, node_y = transformer.transform(node_lon, node_lat)

    source = arcs["from_node_id"].to_numpy(int)
    target = arcs["to_node_id"].to_numpy(int)

    route_source = source[route_ids]
    route_target = target[route_ids]

    lines = []
    for s, t in zip(route_source, route_target):
        line = LineString([
            (node_x[s], node_y[s]),
            (node_x[t], node_y[t]),
        ])
        lines.append(line)

    if any(line.length <= 0.01 for line in lines):
        raise ValueError(
            "Reference route contains zero-length geometry. "
            "Handle degenerate route arcs before alignment."
        )

    n_obs = len(track)
    n_route = len(route_ids)

    # Each row contains distances from one GPS point to all route arcs.
    distance_matrix = np.empty((n_obs, n_route), dtype=np.float32)

    for i in range(n_obs):
        point = Point(float(gps_x[i]), float(gps_y[i]))
        distance_matrix[i] = distance(point, lines)

    # DP: route position may stay or move forward, never backwards.
    # Advancing over route positions incurs skip_penalty_m per position.
    previous = (
        distance_matrix[0]
        + skip_penalty_m * np.arange(n_route)
    ).astype(np.float64)

    back = np.zeros((n_obs, n_route), dtype=np.int32)

    for i in range(1, n_obs):
        adjusted = previous - skip_penalty_m * np.arange(n_route)
        prefix_best = np.minimum.accumulate(adjusted)
        prefix_arg = np.empty(n_route, dtype=np.int32)

        best_index = 0
        best_value = adjusted[0]

        for j in range(n_route):
            if adjusted[j] < best_value:
                best_value = adjusted[j]
                best_index = j
            prefix_arg[j] = best_index

        current = (
            distance_matrix[i]
            + prefix_best
            + skip_penalty_m * np.arange(n_route)
        )

        back[i] = prefix_arg
        previous = current

    aligned_positions = np.empty(n_obs, dtype=np.int32)
    aligned_positions[-1] = int(np.argmin(previous))

    for i in range(n_obs - 1, 0, -1):
        aligned_positions[i - 1] = back[
            i, aligned_positions[i]
        ]

    aligned_arc_ids = route_ids[aligned_positions]
    aligned_distances = distance_matrix[
        np.arange(n_obs), aligned_positions
    ]

    result = pd.DataFrame({
        "observation_id": np.arange(n_obs),
        "aligned_route_position": aligned_positions,
        "aligned_arc_id": aligned_arc_ids,
        "alignment_distance_m": aligned_distances,
    })

    result.attrs["summary"] = {
        "gps_observations": n_obs,
        "route_positions": n_route,
        "unique_aligned_route_positions": int(
            np.unique(aligned_positions).size
        ),
        "monotonic_alignment": bool(
            np.all(np.diff(aligned_positions) >= 0)
        ),
        "mean_alignment_distance_m": float(
            np.mean(aligned_distances)
        ),
        "median_alignment_distance_m": float(
            np.median(aligned_distances)
        ),
        "p95_alignment_distance_m": float(
            np.percentile(aligned_distances, 95)
        ),
        "fraction_within_25m": float(
            np.mean(aligned_distances <= 25)
        ),
        "fraction_within_50m": float(
            np.mean(aligned_distances <= 50)
        ),
        "skip_penalty_m": skip_penalty_m,
        "utm_epsg": epsg,
    }

    return result


def label_candidates(candidates, alignment):
    """Attach provisional labels without using labels as input features."""

    result = candidates.merge(
        alignment[["observation_id", "aligned_arc_id"]],
        on="observation_id",
        how="left",
        validate="many_to_one",
    )

    result["is_matching_candidate"] = (
        result["arc_id"] == result["aligned_arc_id"]
    ).astype("int8")

    result["is_matching_candidate"] = result[
        "is_matching_candidate"
    ].astype("int8")

    return result


if __name__ == "__main__":
    from src.data.load_trip import load_trip

    trip_id = "00000001"
    trip_dir = Path("data/Student folder/train") / trip_id
    data = load_trip(trip_dir, trip_id)

    print("Generating candidates...")
    candidates = generate_candidates(data, trip_id)

    print("Aligning GPS sequence to reference route...")
    alignment = align_track_to_route(data)

    print("\n===== ALIGNMENT SUMMARY =====")
    for key, value in alignment.attrs["summary"].items():
        print(f"{key}: {value}")

    labelled = label_candidates(candidates, alignment)

    print("\n===== LABEL SUMMARY =====")
    print(labelled["is_matching_candidate"].value_counts().to_string())

    positive_per_observation = labelled.groupby(
        "observation_id"
    )["is_matching_candidate"].sum()

    print("\nObservations with a positive candidate:",
          int((positive_per_observation > 0).sum()))
    print("Observations without a positive candidate:",
          int((positive_per_observation == 0).sum()))

    print("\n===== FIRST 10 LABELLED CANDIDATES =====")
    print(labelled.head(10).to_string(index=False))

    out = Path("experiments/labelled_candidates_00000001.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(out, index=False)

    alignment_out = Path("experiments/alignment_00000001.csv")
    alignment.to_csv(alignment_out, index=False)

    print("\nSaved:", out)
    print("Saved:", alignment_out)

"""Spatial candidate generation and feature extraction for PathSnap."""

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import LineString, Point, STRtree

from src.graph.road_graph import haversine_array


def _bearing_difference(a, b):
    """Smallest absolute angular difference in degrees."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _gps_bearings(x, y):
    """Estimate heading from consecutive GPS observations."""
    dx = np.diff(x)
    dy = np.diff(y)

    # Bearing clockwise from north.
    bearings = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

    result = np.full(len(x), np.nan)
    if len(x) > 1:
        result[1:] = bearings
    return result


def generate_candidates(
    data,
    trip_id,
    radius_m=50.0,
    max_candidates=8,
    max_speed_gap_s=60.0,
):
    """Generate nearby directed road-arc candidates for every GPS point."""

    track = data["track"].reset_index(drop=True)
    nodes = data["nodes"]
    arcs = data["arcs"]

    gps_lon = track["longitude"].to_numpy(dtype=float)
    gps_lat = track["latitude"].to_numpy(dtype=float)
    timestamps = track["timestamp_s"].to_numpy(dtype=float)

    node_lon = nodes["longitude"].to_numpy(dtype=float)
    node_lat = nodes["latitude"].to_numpy(dtype=float)

    if not (
        np.isfinite(gps_lon).all()
        and np.isfinite(gps_lat).all()
        and np.isfinite(timestamps).all()
    ):
        raise ValueError(f"{trip_id}: GPS data contains non-finite values.")

    if len(track) == 0:
        raise ValueError(f"{trip_id}: empty GPS track.")

    # Select a UTM zone from the trip's approximate centre.
    centre_lon = float(np.median(gps_lon))
    centre_lat = float(np.median(gps_lat))
    zone = int(np.floor((centre_lon + 180.0) / 6.0) + 1)
    epsg = (32600 if centre_lat >= 0 else 32700) + zone

    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )

    gps_x, gps_y = transformer.transform(gps_lon, gps_lat)
    node_x, node_y = transformer.transform(node_lon, node_lat)

    gps_x = np.asarray(gps_x)
    gps_y = np.asarray(gps_y)
    node_x = np.asarray(node_x)
    node_y = np.asarray(node_y)

    source = arcs["from_node_id"].to_numpy(dtype=int)
    target = arcs["to_node_id"].to_numpy(dtype=int)

    # Preserve original arc IDs, but omit degenerate geometry from the index.
    dx = node_x[target] - node_x[source]
    dy = node_y[target] - node_y[source]
    lengths = np.hypot(dx, dy)

    valid_arc_ids = np.flatnonzero(
        np.isfinite(lengths) & (lengths > 0.01)
    )

    lines = [
        LineString([
            (node_x[source[i]], node_y[source[i]]),
            (node_x[target[i]], node_y[target[i]]),
        ])
        for i in valid_arc_ids
    ]

    if not lines:
        raise ValueError(f"{trip_id}: no non-degenerate road arcs.")

    tree = STRtree(lines)

    road_bearings = (
        np.degrees(np.arctan2(dx[valid_arc_ids], dy[valid_arc_ids]))
        + 360.0
    ) % 360.0

    gps_heading = _gps_bearings(gps_x, gps_y)

    dt = np.diff(timestamps, prepend=np.nan)
    gps_distance = np.full(len(track), np.nan)

    if len(track) > 1:
        gps_distance[1:] = np.hypot(
            np.diff(gps_x), np.diff(gps_y)
        )

    valid_speed_interval = (
        np.isfinite(dt) & (dt > 0) & (dt <= max_speed_gap_s)
    )

    gps_speed = np.full(len(track), np.nan)
    gps_speed[valid_speed_interval] = (
        gps_distance[valid_speed_interval] / dt[valid_speed_interval]
    )

    rows = []
    observations_with_candidates = 0

    for obs_id, (x, y) in enumerate(zip(gps_x, gps_y)):
        point = Point(float(x), float(y))
        nearby_indices = tree.query(point.buffer(radius_m))

        matches = []

        for tree_index in nearby_indices:
            tree_index = int(tree_index)
            line = lines[tree_index]
            distance = float(point.distance(line))

            if distance > radius_m:
                continue

            position = float(line.project(point) / line.length)
            arc_id = int(valid_arc_ids[tree_index])
            road_heading = float(road_bearings[tree_index])

            heading_diff = (
                float(_bearing_difference(
                    gps_heading[obs_id], road_heading
                ))
                if np.isfinite(gps_heading[obs_id])
                else np.nan
            )

            matches.append({
                "trip_id": trip_id,
                "observation_id": obs_id,
                "arc_id": arc_id,
                "distance_m": distance,
                "projection_position": position,
                "road_bearing": road_heading,
                "gps_heading": gps_heading[obs_id],
                "heading_difference": heading_diff,
                "gps_speed_mps": gps_speed[obs_id],
                "time_delta_s": dt[obs_id],
                "large_or_invalid_time_gap": not bool(
                    valid_speed_interval[obs_id]
                ),
            })

        matches.sort(key=lambda row: row["distance_m"])
        matches = matches[:max_candidates]

        if matches:
            observations_with_candidates += 1
            rows.extend(matches)

    result = pd.DataFrame(rows)

    if result.empty:
        result = pd.DataFrame(columns=[
            "trip_id", "observation_id", "arc_id", "distance_m",
            "projection_position", "road_bearing", "gps_heading",
            "heading_difference", "gps_speed_mps", "time_delta_s",
            "large_or_invalid_time_gap",
        ])

    result.attrs["summary"] = {
        "trip_id": trip_id,
        "gps_observations": len(track),
        "observations_with_candidates": observations_with_candidates,
        "observations_without_candidates": (
            len(track) - observations_with_candidates
        ),
        "candidate_coverage": (
            observations_with_candidates / len(track)
        ),
        "candidate_rows": len(result),
        "mean_candidates_per_covered_observation": (
            len(result) / observations_with_candidates
            if observations_with_candidates else 0.0
        ),
        "radius_m": radius_m,
        "max_candidates": max_candidates,
        "utm_epsg": epsg,
        "degenerate_arcs_excluded_from_geometry": (
            len(arcs) - len(valid_arc_ids)
        ),
    }

    return result


if __name__ == "__main__":
    from src.data.load_trip import load_trip

    trip_id = "00000001"
    trip_dir = Path("data/Student folder/train") / trip_id
    data = load_trip(trip_dir, trip_id)

    candidates = generate_candidates(data, trip_id)
    summary = candidates.attrs["summary"]

    print("\n===== CANDIDATE SUMMARY =====")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n===== FIRST 10 CANDIDATES =====")
    print(candidates.head(10).to_string(index=False))

    output = Path("experiments/candidates_00000001.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output, index=False)
    print("\nSaved:", output)

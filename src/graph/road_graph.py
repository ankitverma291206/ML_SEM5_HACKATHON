"""Build directed road graphs while preserving original PathSnap arc IDs."""

import numpy as np
import networkx as nx


EARTH_RADIUS_M = 6_371_000


def haversine_array(lon1, lat1, lon2, lat2):
    """Vectorized distance in metres between coordinate arrays."""
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2 * EARTH_RADIUS_M
        * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )


def build_road_graph(data):
    """Build a NetworkX MultiDiGraph; edge keys are original arc IDs."""
    nodes = data["nodes"]
    arcs = data["arcs"]

    if not nodes.index.equals(
        __import__("pandas").RangeIndex(len(nodes))
    ):
        raise ValueError("Node IDs must be zero-based row indices.")

    if not arcs.index.equals(
        __import__("pandas").RangeIndex(len(arcs))
    ):
        raise ValueError("Arc IDs must be zero-based row indices.")

    node_lon = nodes["longitude"].to_numpy(dtype=float)
    node_lat = nodes["latitude"].to_numpy(dtype=float)

    if not np.isfinite(node_lon).all() or not np.isfinite(node_lat).all():
        raise ValueError("Road nodes contain non-finite coordinates.")

    if (
        (np.abs(node_lon) > 180).any()
        or (np.abs(node_lat) > 90).any()
    ):
        raise ValueError("Road nodes contain out-of-range coordinates.")

    source = arcs["from_node_id"].to_numpy(dtype=int)
    target = arcs["to_node_id"].to_numpy(dtype=int)

    if (
        (source < 0).any()
        or (target < 0).any()
        or (source >= len(nodes)).any()
        or (target >= len(nodes)).any()
    ):
        raise ValueError("At least one arc has an invalid node endpoint.")

    lengths = haversine_array(
        node_lon[source],
        node_lat[source],
        node_lon[target],
        node_lat[target],
    )

    graph = nx.MultiDiGraph()

    graph.add_nodes_from(
        (
            int(node_id),
            {
                "longitude": float(node_lon[node_id]),
                "latitude": float(node_lat[node_id]),
            },
        )
        for node_id in range(len(nodes))
    )

    graph.add_edges_from(
        (
            int(source[i]),
            int(target[i]),
            int(i),
            {"arc_id": int(i), "length_m": float(lengths[i])},
        )
        for i in range(len(arcs))
    )

    return graph, lengths


def route_continuity(data):
    """Check whether consecutive reference arcs share connecting nodes."""
    arcs = data["arcs"]
    route_ids = data["route"]["arc_id"].to_numpy(dtype=int)

    if len(route_ids) == 0:
        return 0, 0

    if (
        (route_ids < 0).any()
        or (route_ids >= len(arcs)).any()
    ):
        raise ValueError("Reference route contains invalid arc IDs.")

    if len(route_ids) < 2:
        return 0, 0

    sources = arcs["from_node_id"].to_numpy(dtype=int)
    targets = arcs["to_node_id"].to_numpy(dtype=int)

    previous_targets = targets[route_ids[:-1]]
    next_sources = sources[route_ids[1:]]

    mismatches = int(
        np.count_nonzero(previous_targets != next_sources)
    )

    return len(route_ids) - 1, mismatches


def graph_diagnostics(data, trip_id):
    """Return graph statistics and the constructed graph."""
    graph, lengths = build_road_graph(data)
    arcs = data["arcs"]

    transitions, mismatches = route_continuity(data)

    report = {
        "trip_id": trip_id,
        "nodes": graph.number_of_nodes(),
        "directed_arcs": graph.number_of_edges(),
        "weak_components": nx.number_weakly_connected_components(graph),
        "isolated_nodes": len(list(nx.isolates(graph))),
        "duplicate_endpoint_pairs": int(
            arcs.duplicated(
                subset=["from_node_id", "to_node_id"]
            ).sum()
        ),
        "self_loops": nx.number_of_selfloops(graph),
        "route_transitions": transitions,
        "route_discontinuities": mismatches,
        "route_discontinuity_rate": (
            mismatches / transitions if transitions else 0.0
        ),
        "mean_arc_length_m": float(np.mean(lengths)),
        "median_arc_length_m": float(np.median(lengths)),
        "zero_length_arcs": int(np.count_nonzero(lengths == 0)),
    }

    return report, graph

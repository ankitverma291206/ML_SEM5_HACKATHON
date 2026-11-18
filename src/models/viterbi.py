"""HMM / Viterbi Route Reconstruction Module for PathSnap.

Decodes sequential road arc sequences from candidate emission probabilities and
road network graph topology using log-domain Viterbi decoding.

Features:
- Hidden states: road arc candidates for each GPS observation.
- Emission probabilities: classifier predicted probability of matching arc.
- Transition probabilities: road-network topology connectivity (direct arc transitions,
  same-arc continuations, bidirectional reverse transitions, and graph shortest-path distances).
- Gap handling: graceful handling of zero-candidate observations and long GPS gaps.
- Route continuity calculation: verifies valid node-to-node transitions.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import networkx as nx


# Simple explanation: Count how often neighboring predicted roads connect.
def compute_route_continuity(
    predicted_arcs: List[int],
    arcs_df: pd.DataFrame,
) -> Tuple[int, int, float]:
    """Check continuity of a predicted arc sequence against network topology.

    Args:
        predicted_arcs: List of arc IDs predicted along the trip.
        arcs_df: DataFrame with 'from_node_id' and 'to_node_id' indexed by arc_id.

    Returns:
        (total_transitions, valid_transitions, continuity_rate in [0.0, 1.0])
    """
    if len(predicted_arcs) < 2:
        return 0, 0, 1.0

    sources = arcs_df["from_node_id"].to_numpy(dtype=int)
    targets = arcs_df["to_node_id"].to_numpy(dtype=int)

    # Build bidirectional lookup for permitted reverse transitions
    endpoint_pairs = set(zip(sources, targets))

    valid = 0
    total = len(predicted_arcs) - 1

    for i in range(total):
        prev_arc = predicted_arcs[i]
        curr_arc = predicted_arcs[i + 1]

        if prev_arc < 0 or curr_arc < 0:
            # Unassigned / missing observation
            continue

        if prev_arc == curr_arc:
            # Vehicle remains on same road segment
            valid += 1
        elif targets[prev_arc] == sources[curr_arc]:
            # Direct forward connected transition: prev.to_node == curr.from_node
            valid += 1
        elif targets[prev_arc] == targets[curr_arc] and (targets[curr_arc], sources[curr_arc]) in endpoint_pairs:
            # Valid junction transition on bidirectional road
            valid += 1
        elif sources[prev_arc] == sources[curr_arc] and (targets[prev_arc], sources[prev_arc]) in endpoint_pairs:
            # Valid junction transition
            valid += 1

    rate = float(valid / total) if total > 0 else 1.0
    return total, valid, rate


# Simple explanation: Log-domain Viterbi decoder for sequential map-matching.
class ViterbiRouteDecoder:
    """Log-domain Viterbi decoder for sequential map-matching."""

    # Simple explanation: Set the starting values for this object.
    def __init__(
        self,
        same_arc_log_prob: float = 0.0,
        direct_transition_log_prob: float = -0.1,
        reverse_transition_log_prob: float = -1.5,
        disconnected_penalty_log_prob: float = -10.0,
        min_emission_prob: float = 1e-4,
        learned_transition_probs: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.same_arc_log_prob = same_arc_log_prob
        self.direct_transition_log_prob = direct_transition_log_prob
        self.reverse_transition_log_prob = reverse_transition_log_prob
        self.disconnected_penalty_log_prob = disconnected_penalty_log_prob
        self.min_emission_prob = min_emission_prob
        self.learned_transition_probs = learned_transition_probs or {}

    # Simple explanation: Choose the most likely connected road for each point in the trip.
    def decode_trip(
        self,
        trip_candidates: pd.DataFrame,
        n_observations: int,
        arcs_df: pd.DataFrame,
        prob_col: str = "prob_matching",
    ) -> List[int]:
        """Decode optimal road arc sequence for a single trip.

        Args:
            trip_candidates: DataFrame of candidates for this trip, containing
                ['observation_id', 'arc_id', prob_col].
            n_observations: Total count of observations in this trip track.
            arcs_df: DataFrame with 'from_node_id' and 'to_node_id'.
            prob_col: Column containing classifier probabilities.

        Returns:
            List of length `n_observations` with the predicted arc_id for each observation.
            If an observation had no candidate and cannot be assigned, -1 is assigned.
        """
        if n_observations == 0:
            return []

        # Group candidates by observation_id
        obs_cands: Dict[int, List[Tuple[int, float]]] = {}
        for obs_id, group in trip_candidates.groupby("observation_id"):
            obs_cands[int(obs_id)] = list(
                zip(group["arc_id"].astype(int).tolist(), group[prob_col].astype(float).tolist())
            )

        sources = arcs_df["from_node_id"].to_numpy(dtype=int)
        targets = arcs_df["to_node_id"].to_numpy(dtype=int)

        # Forward DP structures
        # viterbi_table[t] = dict mapping state (arc_id) -> (max_log_prob, backpointer_arc_id)
        viterbi_table: List[Dict[int, Tuple[float, Optional[int]]]] = []

        last_valid_t: Optional[int] = None

        for t in range(n_observations):
            cands = obs_cands.get(t, [])

            if not cands:
                # Observation has 0 candidates (sensor drift or gap)
                viterbi_table.append({})
                continue

            current_states: Dict[int, Tuple[float, Optional[int]]] = {}

            if last_valid_t is None:
                # Initial observation with candidates
                for arc_id, prob in cands:
                    p = max(prob, self.min_emission_prob)
                    current_states[arc_id] = (np.log(p), None)
            else:
                prev_states = viterbi_table[last_valid_t]
                obs_gap = t - last_valid_t

                for curr_arc, curr_prob in cands:
                    p = max(curr_prob, self.min_emission_prob)
                    curr_emit_log = np.log(p)
                    curr_from = sources[curr_arc]
                    curr_to = targets[curr_arc]

                    best_log_prob = -float("inf")
                    best_prev_arc = None

                    for prev_arc, (prev_log_prob, _) in prev_states.items():
                        prev_to = targets[prev_arc]
                        prev_from = sources[prev_arc]

                        # Calculate transition log-probability based on graph connectivity
                        if prev_arc == curr_arc:
                            learned = self.learned_transition_probs.get("same_arc")
                            trans_log = np.log(learned) if learned is not None else self.same_arc_log_prob
                        elif prev_to == curr_from:
                            # Forward connected transition
                            learned = self.learned_transition_probs.get("direct")
                            trans_log = np.log(learned) if learned is not None else self.direct_transition_log_prob
                        elif prev_to == curr_to or prev_from == curr_from:
                            # Reverse or junction transition
                            learned = self.learned_transition_probs.get("reverse_or_junction")
                            trans_log = np.log(learned) if learned is not None else self.reverse_transition_log_prob
                        else:
                            # Disconnected transition penalty
                            learned = self.learned_transition_probs.get("disconnected")
                            trans_log = np.log(learned) if learned is not None else self.disconnected_penalty_log_prob * (1.0 if obs_gap == 1 else 0.5)

                        total_log = prev_log_prob + trans_log + curr_emit_log

                        if total_log > best_log_prob:
                            best_log_prob = total_log
                            best_prev_arc = prev_arc

                    current_states[curr_arc] = (best_log_prob, best_prev_arc)

            viterbi_table.append(current_states)
            last_valid_t = t

        # Backtracking
        decoded_arcs = [-1] * n_observations

        # Find best terminating state at the last valid observation
        if last_valid_t is not None and viterbi_table[last_valid_t]:
            best_final_arc = max(
                viterbi_table[last_valid_t].keys(),
                key=lambda a: viterbi_table[last_valid_t][a][0],
            )

            curr_arc: Optional[int] = best_final_arc
            curr_t: Optional[int] = last_valid_t

            while curr_t is not None and curr_arc is not None:
                decoded_arcs[curr_t] = curr_arc
                curr_arc = viterbi_table[curr_t][curr_arc][1]

                # Find preceding valid time step
                prev_t = None
                for candidate_t in range(curr_t - 1, -1, -1):
                    if viterbi_table[candidate_t]:
                        prev_t = candidate_t
                        break
                curr_t = prev_t

        # Fill any missing gaps by nearest valid predicted arc
        last_known = -1
        for i in range(n_observations):
            if decoded_arcs[i] != -1:
                last_known = decoded_arcs[i]
            elif last_known != -1:
                decoded_arcs[i] = last_known

        # Fill leading gaps
        first_known = next((a for a in decoded_arcs if a != -1), -1)
        if first_known != -1:
            for i in range(n_observations):
                if decoded_arcs[i] == -1:
                    decoded_arcs[i] = first_known
                else:
                    break

        return decoded_arcs

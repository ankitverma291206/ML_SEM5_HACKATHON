# PathSnap Feature Schema & Candidate Matching Contract

**Author:** Atharv Kulkarni (PES2UG24CS092) — Machine Learning Workstream  
**Upstream Provider:** Aum Lavkumar Padia — Data Engineering Workstream  
**Downstream Consumers:** Arya M G (HMM Decoder) & Ankit Verma (FastAPI Backend)  
**Version:** 1.0 (Verified against Aum's candidate pipeline)

---

## 1. Overview

Each sample in the candidate dataset represents one **(GPS observation, candidate road arc)** pair.
For an observation with $K$ candidate road segments proposed by the spatial index within a 50m search radius, exactly $K$ candidate rows are generated.

---

## 2. Identifier Columns (Mandatory, Non-Predictive)

> [!CAUTION]
> These columns identify samples and must NEVER be passed to machine learning estimators as predictive features.

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `trip_id` | `str` / `int` | Global trip identifier (e.g. `"00000001"`). Groups observations for leak-free cross-validation. |
| `observation_id` | `int` | Sequential observation index within the trip track ($0, 1, \dots, N-1$). |
| `arc_id` | `int` | Local directed arc identifier in the road network graph. |

---

## 3. Predictive Feature Columns

| Column Name | Type | Physical Unit | Description & Engineering Details |
| :--- | :--- | :--- | :--- |
| `distance_m` | `float32` / `float64` | Meters ($m$) | Perpendicular distance from projected GPS coordinate to the candidate directed road arc geometry. |
| `projection_position` | `float32` / `float64` | Ratio $[0, 1]$ | Normalized fractional distance along the directed segment from `from_node` to `to_node`. |
| `road_bearing` | `float32` / `float64` | Degrees $[0, 360)$ | Azimuth bearing of the directed road arc clockwise from true North. |
| `gps_heading` | `float32` / `float64` | Degrees $[0, 360)$ | Estimated instantaneous heading computed from successive GPS positions. NaN at $t=0$. |
| `heading_difference` | `float32` / `float64` | Degrees $[0, 180]$ | Smallest acute angular difference between vehicle travel heading and road segment direction. |
| `gps_speed_mps` | `float32` / `float64` | $m/s$ | Instantaneous GPS speed calculated as displacement over time interval. NaN at $t=0$. |
| `time_delta_s` | `float32` / `float64` | Seconds ($s$) | Elapsed sampling interval from previous GPS observation. |
| `large_or_invalid_time_gap` | `int8` ($0$ or $1$) | Binary flag | Indicator flagging irregular time gaps ($>60s$), negative intervals, or stationary drift. |

---

## 4. Target Label Column (Training & Evaluation Only)

| Column Name | Type | Values | Description |
| :--- | :--- | :--- | :--- |
| `is_matching_candidate` | `int8` | $0$ or $1$ | Positive class ($1$) indicates this candidate arc corresponds to the vehicle's true reference route at this observation; negative class ($0$) indicates a non-matching candidate. |

---

## 5. Reference Alignment Metadata (Strictly Forbidden in Feature Inputs)

> [!WARNING]
> The following columns are generated during sequence alignment for validation purposes. They contain ground-truth reference route data and are strictly forbidden from entering any feature matrix:
> - `aligned_arc_id`
> - `aligned_route_position`
> - `alignment_distance_m`

---

## 6. Preprocessing & Handling Rules

1. **Missing Value Imputation**:
   First observations ($t=0$) naturally have missing `gps_heading`, `heading_difference`, `gps_speed_mps`, and `time_delta_s`. Missing values are imputed using median values strictly fitted on training folds via `sklearn.impute.SimpleImputer(strategy='median')`.
2. **Feature Scaling**:
   Standardization via `StandardScaler()` is strictly fitted on training folds. Required for distance-based and gradient-based models (KNN, Logistic Regression); neutral for tree-based ensembles.
3. **Circular Transformation Support**:
   Optional sine/cosine decomposition $\sin(2\pi \cdot \text{angle} / 360)$, $\cos(2\pi \cdot \text{angle} / 360)$ is supported via `encode_circular_angles()` for circular features (`road_bearing`, `gps_heading`).

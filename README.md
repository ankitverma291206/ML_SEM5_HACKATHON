# PathSnap — Team Work Plan

**Project:** Distributed Map-Matching & Route Reconstruction  
**Course:** UE24CS352A Machine Learning Hackathon  
**Team size:** 4

This README is the team's working plan: it assigns ownership, defines component handoffs, and tracks the work toward the hackathon submission. The supplied hackathon PDFs define the mandatory requirements. Architecture choices and example models below are the team's proposed approach and should be validated against real experiments.

## FastAPI and Docker quick start

The repository includes an initial FastAPI service and Docker Compose setup. It can report health, list trips in the configured partition, and return paginated GPS observations. **The model and HMM have not been implemented yet**, so `POST /predict` currently returns HTTP 503 with a clear message instead of inventing a route. Replace the integration point in `api/pipeline.py` when the real inference pipeline is ready.

### Run with Docker Compose

From the repository root, with Docker Desktop/Docker Engine and the Compose plugin installed:

```bash
docker compose up --build
```

The API listens on port 8000. Open `http://localhost:8000/docs` for the interactive API page. The default trip directory is the supplied `Student folder/student_test` partition, mounted read-only into the container. The repository's optional `data/` directory is also mounted read-only for later evaluator data.

To point the service at another partition, set these values in a local `.env` file (do not commit it):

```dotenv
PATHSNAP_TRIPS_DIR=/app/data/evaluator_test
PATHSNAP_PARTITION=evaluator_test
```

Place the authorized evaluator trip folders under `data/evaluator_test/` on the host. Keep reference labels server-side; the trip-list and observation endpoints do not return `.route` contents.

### Run locally without Docker

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Useful endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service, data-directory, and inference readiness |
| `GET` | `/trips` | IDs in the configured partition |
| `GET` | `/trips/{trip_id}/observations?offset=0&limit=500` | A page of GPS observations, without reference labels |
| `POST` | `/predict` | Prediction integration point; returns 503 until the real pipeline is connected |

Example request body for `/predict`:

```json
{"trip_id": "00000000"}
```

When Laptop 2 hosts the UI, set `PATHSNAP_CORS_ORIGINS` to its exact browser origin(s), comma-separated. For a UI on another laptop, use the backend laptop's LAN IP and port 8000 as the API base URL; allow inbound TCP port 8000 on that laptop's firewall as needed.

## XGBoost candidate-classification model

The processed files currently available in `experiments/` are six labeled training trips: `00000001`, `00000002`, `00000019`, `00000033`, `00000042`, and `00000069`. They contain **165,381 candidate rows** and **21,680 positive candidate labels**. Their matching alignment files contain **26,562 GPS observations**. In these six trips, **4,882 observations (18.4%) have no correct candidate** in the generated candidate set. Trip `00000033` has the largest miss rate (about 67%), so candidate generation should be reviewed there.

These are preliminary processed data, not all 80 training trips. Any cross-validation result from them is an early development result, not final evidence for choosing the project model. Run the preprocessing and labeling pipeline for the remaining training trips before drawing final conclusions. The model script refuses to train from candidate files whose trip IDs are not present in `Student folder/train/`.

The readable, heavily commented training script is `src/models/train_xgboost.py`. It:

1. Loads each labeled candidate file and its matching alignment file.
2. Checks that trip IDs, candidate labels, and alignment files agree.
3. Uses only geometric/movement features. It excludes trip/arc identifiers and reference labels from model inputs.
4. Encodes bearings with sine and cosine so directions around 0°/360° are treated as neighbors. XGBoost handles missing feature values.
5. Measures candidate Macro F1 and Log Loss, observation-level Top-1 accuracy, and candidate miss rate.
6. Uses `GroupKFold` by trip. A trip is kept entirely in either training or validation for each fold.
7. Fits one final model on the processed training trips and saves the model plus metadata.

Run from the repository root:

```bash
pip install -r requirements.txt
python -m src.models.train_xgboost
```

The script writes fold results and candidate-coverage diagnostics under `experiments/`, and saves `artifacts/xgboost_candidate_model.json` plus its feature/configuration metadata. Candidate miss rate measures the upstream candidate generator; a classifier cannot predict an arc that was never offered as a candidate. Sequence continuity will be evaluated separately after the HMM/Viterbi decoder is connected.

## Project goal

Build an end-to-end system that takes noisy GPS observations, identifies likely road segments, and reconstructs a coherent vehicle route using road-network information and an HMM/Viterbi decoder. The final application uses a two-laptop setup: a FastAPI backend on Laptop 1 and an evaluator-facing UI on Laptop 2.

## Dataset and handling rules

The supplied student dataset contains 80 `train` trips and 10 `student_test` trips. Each trip is organized with four whitespace-separated, headerless files:

| File | Contents |
| --- | --- |
| `.track` | GPS longitude, latitude, timestamp observations |
| `.nodes` | Road-network node longitude and latitude |
| `.arcs` | Directed connections as `from_node_id` and `to_node_id`; row index is the `arc_id` |
| `.route` | Ordered reference `arc_id` sequence |

The number of GPS observations need not equal the number of route arcs. Establish and validate a sequence-aware alignment; do not pair rows by index. Arc IDs are local to a trip, so use `(trip_id, arc_id)` wherever IDs cross trip boundaries.

The handout describes a further 10-trip `evaluator_test` partition, which is not included in the supplied ZIP. Keep this partition separate when it is provided. Use the 80 training trips for fitting, cross-validation, and optimization; use the 10 student-testing trips for development comparisons. Do not train on evaluator trips, tune against evaluator labels, expose evaluator ground truth to the model or UI, or hardcode expected predictions.

Before modeling, validate file availability, parsing, coordinates, timestamps and gaps, route-to-arc references, graph connectivity, and candidate coverage. Treat the supplied routes as reference labels with possible limitations, and document assumptions. Fit learned preprocessing on permitted training data only.

## Team ownership

Each person owns one primary workstream. The whole team remains responsible for understanding and being able to explain the complete system.

### Ankit Verma — Backend, UI, deployment, and integration

**Primary output:** A working two-laptop application that runs the complete prediction pipeline.

- Implement a FastAPI backend. Suggested endpoints: `GET /health`, `GET /trips`, and `POST /predict`.
- Build the evaluator UI for trip selection, prediction, map display, result summaries, loading states, and errors. Streamlit is a proposed lightweight option; the handout does not mandate a UI framework.
- Integrate data loading, candidate generation, feature extraction, the saved classifier, and route decoder.
- Keep prediction separate from evaluation: inference receives GPS and road-network inputs, while reference labels remain server-side for evaluation. Never return evaluator ground-truth labels or routes to the UI.
- Configure Laptop 1 (API) and Laptop 2 (UI) communication; make API request logs visible on Laptop 1.
- Exercise the complete pipeline, invalid trip handling, and all 10 evaluator trips when they become available.
- Provide deployment instructions, UI screenshots, and screenshots of successful API responses for the report.

**Completion condition:** The UI can request a real prediction from the API and display its returned route and permitted results.

### Atharv Kulkarni — Machine learning and experimentation

**Primary output:** A reproducible, empirically selected classifier with saved artifacts and experiment records.

- Train and compare at least five distinct models. Suggested starting set: Logistic Regression, K-Nearest Neighbours, Decision Tree, Random Forest, and HistGradientBoosting. These are suggestions, not prescribed algorithms.
- Work with Aum to agree on candidate labels and feature-table schema before full training. A candidate-level positive/negative relevance classifier is one possible formulation; validate it against route alignment and the assignment metrics.
- Use trip-grouped cross-validation (for example, `GroupKFold` by `trip_id`) so observations from a trip cannot leak across folds. Fit preprocessing within each training fold.
- Record model configurations, seeds, training and inference times, fold results, and student-test results.
- Report candidate-level and observation/route-level metrics distinctly. Include applicable accuracy, precision, recall, Macro F1, log loss, Top-1 accuracy, candidate miss rate, route continuity, and confusion matrices.
- Select a promising model using measured generalization, probability quality, route relevance, and runtime—not training accuracy alone.
- Perform Bayesian hyperparameter optimization on the selected model with group-aware CV on training trips; retain the search space, objective, trial history, best parameters, and computational cost. Compare tuned and untuned models on the same partitions and metrics.
- Save the final model, preprocessing components, feature/model configuration, and experiment results for Ankit and Arya to consume.
- Own the report sections on model experimentation, model comparison, visualizations, optimization, and selection rationale.

**Completion condition:** Five or more models have genuinely been trained and compared, Bayesian optimization is reproducible, and the selected model supplies candidate scores/probabilities to the decoder.

### Arya M G — HMM, Viterbi, and route reconstruction

**Primary output:** A complete, evaluated sequence decoder that consumes classifier candidate scores.

- Define hidden states as candidate road segments for each GPS observation and preserve candidate-to-arc correspondence.
- Use classifier probabilities consistently as emission scores; document any additional emission evidence.
- Model directed graph connectivity and transition likelihood. Consider graph travel distance, GPS displacement, time gaps, and heading consistency.
- Implement Viterbi decoding with log probabilities and explicit handling for missing candidates, invalid transitions, long gaps, and disconnected components.
- Return a predicted road segment for each relevant observation.
- Compare classifier-only and classifier-plus-HMM predictions; report observation-level and sequence-level performance, continuity, and cases where decoding does not help. Tune decoder parameters using training and student-testing data only.
- Generate actual Folium maps showing GPS observations, road-network geometry, reference route when available, and predicted route in distinct layers.
- Own the report section on hidden states, emissions, transitions, Viterbi, gap handling, continuity, and decoder comparison.

**Completion condition:** Given candidate scores and a trip graph, the decoder returns and evaluates a complete route sequence.

### Aum Lavkumar Padia — Data engineering, EDA, and feature engineering

**Primary output:** A validated data and candidate-feature pipeline for training and inference.

- Build loaders and validation for `.track`, `.nodes`, `.arcs`, and `.route`; produce a trip inventory and data-quality summaries.
- Investigate observations per trip, graph sizes and connectivity, isolated/disconnected components, timestamp intervals and large gaps, coordinate plausibility, route references, and candidate miss rate.
- Build road-segment geometry from nodes and arcs. Investigate and validate a sequence-aware, monotonic GPS-to-reference-route alignment. Do not assume independent nearest-edge matches always form a correct sequence.
- Generate nearby road-segment candidates efficiently with a spatial index. Use geodesic distances or suitable per-trip projected coordinates; do not use raw degree distances as physical distances.
- Develop and document candidate features such as perpendicular distance, projection position, road bearing, GPS heading/speed, heading difference, time delta, and segment length. These are candidates for investigation, not a fixed required list.
- Produce dataset inventory, data-quality and sampling summaries, trajectory/network visualizations, and alignment/candidate visualizations. Use Folium for geographic maps.
- Hand off stable preprocessing, alignment, candidate-generation and feature APIs, labels, schema documentation, and validation examples to Atharv and Arya; provide the same inference-time functions to Ankit.
- Own the report section on dataset understanding and EDA.

**Completion condition:** Any permitted trip can be loaded, its graph reconstructed, GPS observations processed, and a validated candidate-feature dataset produced.

## Component handoffs and shared contract

Agree on function signatures and schemas early. Proposed end-to-end flow:

```text
Raw trip files
  → Aum: loading, alignment, candidates, features
  → Atharv: candidate scores/probabilities
  → Arya: HMM/Viterbi route sequence
  → Ankit: API, UI, server-side evaluation, deployment
```

| Owner → recipient | Handoff |
| --- | --- |
| Aum → Atharv | Candidate feature rows, labels, trip groups, schema, and validation notes |
| Aum → Arya | Directed road graph, route alignment, candidate-to-arc mapping |
| Atharv → Arya | Loadable classifier and candidate scoring interface |
| Aum → Ankit | Data loading, preprocessing, candidate and feature inference functions |
| Atharv → Ankit | Model-loading and scoring interface plus artifact/configuration requirements |
| Arya → Ankit | Decoder interface, predicted route representation, and evaluation outputs |
| Ankit → team | Integrated application, API contract, run instructions, integration status |

Proposed candidate table (revise and freeze with the team before full integration):

```text
trip_id, observation_id, arc_id, distance_m, projection_position,
road_bearing, gps_heading, heading_difference, gps_speed, time_delta,
is_matching_candidate
```

`is_matching_candidate` is a training/evaluation label, not an inference input. Keep reference-route labels out of the evaluator prediction request and UI. Document any schema changes and update all consumers together.

## Milestones

The handout mentions a first-phase evaluation before the lunch break; the dates and remaining deadlines are not specified here. Set team calendar dates separately and prioritize a real, demonstrable vertical slice early.

1. **Repository, data audit, and interfaces:** All members participate. Establish project structure, loaders, inventory, initial API/UI skeleton, and agreed schemas. **Exit:** permitted trips load and all component interfaces are agreed.
2. **Preprocessing and baseline vertical slice:** Aum leads data processing; Arya reviews alignment; Atharv trains an initial baseline; Ankit connects an actual trip through the API/UI. **Exit:** one real trip completes inference without hardcoded predictions.
3. **Model experimentation:** Atharv leads five-model training and group-aware CV; Aum supports feature/coverage analysis. **Exit:** reproducible logs, comparison table, and evidence-based candidate model selection.
4. **Optimization and decoder:** Atharv runs Bayesian optimization; Arya implements and evaluates HMM/Viterbi. **Exit:** tuning record and classifier-only versus decoded results on permitted partitions.
5. **Application integration:** Ankit leads two-laptop deployment and connects all components. **Exit:** working API/UI with protected ground truth and documented setup.
6. **Evaluation and submission:** All members verify the full pipeline, prepare actual maps/results/report, and test the 10 reserved evaluator trips when supplied. **Exit:** both required deliverables are complete and reproducible.

## Submission requirements

The hackathon requires exactly two final deliverables.

### 1. Public GitHub repository

Commit the complete source code used for data processing, feature engineering, every model experiment, evaluation, Bayesian optimization, HMM/Viterbi, API, UI, visualization, and deployment. Include the README, dependencies, run/deployment instructions, relevant configurations, experiment records, and necessary model artifacts. Do not commit secrets. Keep raw data out of the repository unless the event explicitly permits it; document how authorized data should be placed locally.

Suggested organization (adapt to the implementation):

```text
PathSnap/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # local data instructions; avoid committing restricted raw data
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── decoder/
│   ├── visualization/
│   ├── api/
│   └── ui/
├── artifacts/             # required model/config artifacts, subject to size constraints
├── experiments/           # actual results, logs, and figures
├── maps/                  # generated maps as appropriate
└── report/                # consolidated report PDF
```

### 2. One consolidated technical report PDF

Include actual, verifiable analysis and results. At minimum cover:

- Dataset preprocessing, EDA, missing/invalid values, trajectory and road-network visualizations, Folium map, features and partition used for each analysis.
- Every model actually trained, a consolidated comparison table, configurations, strengths/limitations, measured performance, and appropriate model comparison plots.
- Cross-validation Macro F1; student-test Top-1 accuracy and Macro F1; log loss where applicable; actual training histories only for models that produce them.
- Bayesian search parameters/ranges, objective, trial count/history, best values, computational cost, and tuned-versus-untuned comparison.
- HMM state, emission and transition definitions, gap/disconnection handling, and classifier-only versus HMM results.
- Folium maps generated by the implementation, with GPS, road network, reference (where available), and predicted route layers; include a representative exported view in the PDF and commit the map/code.
- Working UI screenshots and successful API-response screenshots from the receiving laptop.
- API, UI, and deployment explanation; ensure report metrics and claims can be reproduced from the repository.

Suggested report ownership:

| Section | Owner(s) |
| --- | --- |
| Dataset understanding and EDA | Aum |
| Model experimentation/comparison and Bayesian optimization | Atharv |
| HMM/Viterbi and route reconstruction | Arya |
| Geographical maps | Aum + Arya |
| API, UI, deployment and screenshots | Ankit |
| Introduction, conclusion, proofreading and final consistency | All four |

## Experiment and integrity rules

- Train and compare at least five models and retain actual configurations and results.
- Use trip-aware validation on the 80 training trips. Student testing is for development-stage comparisons; evaluator testing is final evaluation only.
- Do not fabricate metrics, charts, training curves, maps, citations, or experiment records. Report only work actually run and explain limitations.
- Do not share code, weights, configurations, derived datasets, or experimental results with other teams. Follow the event's AI disclosure rules; every teammate must understand and defend the submitted work.
- Keep evaluator ground truth server-side. The inference model and UI must not receive it, and predictions must not be hardcoded.

## Team progress checklist

### Aum — Data engineering

- [ ] Validate loaders and inventory all supplied trips/files
- [ ] Produce data-quality, sampling, and graph-connectivity summaries
- [ ] Validate GPS-to-route alignment
- [ ] Implement spatial candidate generation and documented features
- [ ] Generate EDA, alignment, and Folium visualizations
- [ ] Hand off stable feature schema and inference functions

### Atharv — Machine learning

- [ ] Train at least five distinct models
- [ ] Implement trip-grouped cross-validation
- [ ] Record candidate-level and observation/route-level metrics
- [ ] Compare models and justify selection from results
- [ ] Run Bayesian hyperparameter optimization and retain trial history
- [ ] Export model, preprocessing, configuration, and results

### Arya — Route reconstruction

- [ ] Define hidden states and candidate correspondence
- [ ] Implement emissions, transitions, and Viterbi decoding
- [ ] Handle missing candidates, long gaps, invalid transitions, and disconnected components
- [ ] Compare classifier-only and classifier-plus-HMM results
- [ ] Measure route continuity and generate predicted-route maps

### Ankit — Backend and deployment

- [ ] Implement FastAPI health, trip-list, and prediction endpoints
- [ ] Build evaluator UI and map/results display
- [ ] Integrate preprocessing, model, and decoder interfaces
- [ ] Protect server-side reference labels
- [ ] Configure two-laptop networking and visible API logs
- [ ] Capture UI/API evidence and document deployment
- [ ] Run the full evaluator set when it is supplied

## Main risks to manage

1. **GPS-to-route alignment:** Incorrect labels undermine model and decoder evaluation. Validate alignment visually and quantitatively before large experiments.
2. **Trip leakage:** Never randomly split observations in a way that puts one trip in both training and validation.
3. **Irregular timestamps and large gaps:** Make time-aware feature and decoder behavior explicit.
4. **Disconnected road graphs:** Treat unreachable transitions as such; do not assume the full graph is connected.
5. **Candidate misses:** Report when the reference segment is absent from the candidate set; a classifier cannot recover a missing candidate.
6. **Ground-truth leakage:** Keep evaluator labels separate from prediction inputs and UI responses.
7. **Late integration:** Maintain a real-data vertical slice early and use the handoff contracts above.
8. **Unsupported claims:** Re-run data audits and experiments in the repository and report verified outputs only.

## Immediate next steps

1. Confirm this ownership split and agree on the candidate/decoder/API data contracts.
2. Aum builds the validated dataset loader and alignment investigation; Ankit scaffolds the API/UI and project run instructions in parallel.
3. Connect one permitted trip end to end with a baseline model before scaling to all experiments.
4. Track actual progress in the checklists and replace this plan's suggested choices with evidence-backed decisions as the work develops.

# PathSnap

PathSnap matches noisy GPS observations to nearby directed road segments, then uses an HMM/Viterbi decoder to produce a connected route. The FastAPI service provides both the evaluator UI and the prediction API.

## Run the app on Windows

From PowerShell in the project directory:

```powershell
cd "C:\Users\athar\OneDrive\Desktop\ML_SEM5_HACKATHON"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000/](http://localhost:8000/) for the UI or [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation. Keep the terminal running while using the app. The map uses OpenStreetMap tiles and Leaflet from public CDNs, so those map assets require internet access.

## Evaluator UI and data partitions

Use the **Dataset partition** menu to switch between:

- **Student test** — the ten development evaluation trips in `Student folder/student_test/`.
- **Evaluator test** — the ten final evaluation trips in `data/evaluator_test/`.

The evaluator ZIP is intentionally kept out of source control. To set it up on another machine, extract its `evaluator_test/` folder to `data/evaluator_test/`. The evaluator partition is used only for listing trips, inference, and evaluation. Training and preprocessing read from `Student folder/train/` and never include `data/evaluator_test/`.

The UI loads GPS observations, then requests a predicted route. The API does not return reference-route labels to the UI.

## Current model and validation

The current route inference pipeline uses:

1. Candidate generation from GPS points and each trip's road graph.
2. An XGBoost candidate scorer with 20 engineered numeric inputs. It scores each possible road segment.
3. A first-order road-arc HMM and Viterbi decoding. Transition probabilities are learned from labeled training routes; scorer probabilities act as emissions.

The training scripts currently preprocess the first 30 training trips and use five-fold GroupKFold validation, keeping each trip entirely within one fold. The latest recorded 30-trip CV results are in `experiments/xgboost_group_cv_summary.csv` and `experiments/xgboost_group_cv.csv`:

| Measure | Mean CV result | What it measures |
| --- | ---: | --- |
| Candidate accuracy | 90.67% | Candidate classification at a 0.5 threshold |
| Candidate Macro F1 | 0.8248 | Candidate classification across both classes |
| End-to-end Top-1 accuracy | 70.74% | Correct best candidate across all GPS observations; candidate misses count as wrong |
| HMM route continuity | 97.31% | Share of consecutive decoded roads that connect, averaged across held-out trips |
| HMM observation accuracy | 78.14% | Decoded arc matches the aligned arc at each observation |

These are cross-validation results, not scores guaranteed on either test partition. Route continuity measures graph connectivity, not whether every predicted arc matches the reference. The HMM CV transition model is fit on each fold's training trips and evaluated on its held-out trips. The final persisted transition table is in `artifacts/hmm_transition_model.json`.

## Rebuild preprocessing and models

Run from the repository root using the active virtual environment:

```powershell
# Build candidate features and labels from 30 training trips
python -m src.features.build_training_dataset

# Train/evaluate XGBoost with 5-fold trip-grouped CV, then save its final scorer
python -m src.models.train_xgboost

# Fit the HMM transition table on the preprocessed training trips
python -m src.models.train_hmm
```

The preprocessing writes candidate data under `data/processed/` and per-trip training CSVs and audit outputs under `experiments/`. The XGBoost run writes its fold results and summary under `experiments/`, and saves the scorer to `artifacts/xgboost_candidate_model.json`. The HMM training command writes `artifacts/hmm_transition_model.json`. `data/` is git-ignored; keep the authorized raw data locally.

Other model code is in `src/models/`. Logistic Regression can be run with:

```powershell
python -m src.models.train_logistic_regression
```

Its input is the labeled candidate CSV files under `experiments/`; it writes evaluation output to that directory and saves the fitted estimator under `models/`.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | API, data directory, and model readiness |
| `GET` | `/trips?partition=student_test` | List trip IDs from the selected partition |
| `GET` | `/trips/{trip_id}/observations?partition=evaluator_test&limit=2000` | Return GPS observations without reference labels |
| `POST` | `/predict` | Generate candidates, score them, and return an HMM/Viterbi route |
| `GET` | `/model-results` | Return saved model comparison metrics |
| `GET` | `/hmm-status` | Return HMM training details and grouped-CV metrics |

Example prediction request:

```json
{
  "trip_id": "00000015",
  "partition": "evaluator_test"
}
```

The `partition` value must be `student_test` or `evaluator_test`. It defaults to `student_test` when omitted. The response includes map-ready `route` points and predicted arc IDs; it does not include the reference route.

## Docker

With Docker Desktop and Compose installed:

```powershell
docker compose up --build
```

Then open [http://localhost:8000/](http://localhost:8000/). Compose mounts `Student folder/`, `data/`, `experiments/`, and `artifacts/` into the container. Keep the evaluator files in `data/evaluator_test/` on the Docker host; the `data/` mount is read-only in the container.

## Repository map

- `api/` — FastAPI endpoints, partition-aware trip access, candidate scoring, and route prediction.
- `ui/index.html` — Evaluator interface, partition selector, map, metrics, and HMM details.
- `src/data/` — Trip file loading and data checks.
- `src/features/` — Candidate generation, labeling, and training preprocessing.
- `src/graph/` — Road-network helpers.
- `src/models/` — Classifiers, validation, metrics, optimization, HMM training, and Viterbi decoding.
- `artifacts/` — Saved model and HMM parameters needed by inference.
- `experiments/` — Cross-validation results, metrics, and diagnostics.

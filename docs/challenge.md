# LATAM MLE Challenge — Implementation Notes

## Overview
- REST API predicts **departure delay > 15 min** using `OPERA`, `TIPOVUELO`, `MES`.
- **Train offline, serve online**: the model is trained once and **persisted** to `models/delay_model.pkl`; the API **loads** it at boot (no training on startup).

## What’s new / important
- **Persisted model artifact**  
  `make train` saves `models/delay_model.pkl`. API reads `MODEL_PATH` (defaults to `/app/models/delay_model.pkl` in Docker).
- **FastAPI Lifespan (no `on_event`)**  
  Model loads during lifespan; we also **lazy-init** on first request so tests that skip startup still work.
- **Notebook parity for features**  
  Serving uses the **fixed top-10 one-hot features** expected by the challenge/tests.
- **Class weighting like the notebook**  
  Base: `class_weight = {1: n_y0/len(y), 0: n_y1/len(y)}`; we expose `pos_weight_gamma` (`2.0` by default) as a small positive-class boost. Set to `1.0` for exact notebook behavior.
- **Cold-start safety**  
  If an unfitted model is used (e.g., missing artifact), `predict()` returns all zeros instead of crashing.
- **Domain validation**  
  Always validate `TIPOVUELO ∈ {I,N}` and `MES ∈ [1..12]`; enforce `OPERA` membership **only when** a trained model provided known operators.
- **Pydantic v2 schemas**  
  Request/response models live in `challenge/schemas.py`.
- **Legacy pickle compatibility**  
  Can load artifacts saved when the class lived at `model.DelayModel`.

## How to run (local)
```bash
# 1) dependencies
python -m venv .venv && source .venv/bin/activate
make install

# 2) train (creates models/delay_model.pkl)
make train

# 3) quality gates
make lint
make format
make test

# 4) API (http://127.0.0.1:8000/docs)
make run
```

## Docker

Runs as non-root, PYTHONPATH=/app, respects ${PORT:-8000}.

Build requires the trained model (forces “train before build”).

```bash
make train
make docker-build
make docker-run
```

## CI/CD

- **CI** (`.github/workflows/ci.yml`): runs on PRs/pushes — installs dependencies, lints with **Ruff**, and runs tests with **pytest**.

- **CD (GCP)** (`.github/workflows/cd.yml`):
  - Includes a **gate job** so the deploy step **auto-skips** unless all required GCP secrets are present.
  - **Note:** Deployment to GCP is intentionally disabled to avoid enabling billing for this challenge. When you later add the secrets (and enable billing), it will deploy to **Cloud Run** automatically.


import os
from pathlib import Path
from typing import List, Dict, Any

import fastapi
from fastapi.concurrency import asynccontextmanager
import pandas as pd
from fastapi import HTTPException, Body

from .model import DelayModel
from .schemas import PredictRequest, PredictResponse

app = fastapi.FastAPI(title="LATAM Delay API", version="1.5.4")

_MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/delay_model.pkl"))
_ALLOWED_TIPOS = {"I", "N"}
_ALLOWED_MES = set(range(1, 13))
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """
    Lifespan context replaces deprecated @app.on_event("startup"/"shutdown").

    Startup:
      - Try to load persisted model (if present).
      - Fallback to unfitted model (predict -> zeros) if artifact missing or load fails.
      - Populate allowed_opera from the trained artifact (if any).

    Shutdown:
      - No special teardown; clear state defensively.
    """
    model: DelayModel | None = None
    allowed_opera: set[str] = set()
    try:
        if _MODEL_PATH.exists():
            model = DelayModel.load(_MODEL_PATH)
            allowed_opera = set(model.known_operators)
    except Exception as exc:
        print(f"[lifespan/startup] Model load failed: {exc}. Using unfitted model fallback.")
        model = None

    if model is None:
        model = DelayModel()
        allowed_opera = set()

    app.state.model = model
    app.state.allowed_opera = allowed_opera

    yield

    # Shutdown
    app.state.model = None
    app.state.allowed_opera = set()

def _get_model() -> DelayModel:
    model = getattr(app.state, "model", None)
    if model is None:
        print("[lazy-init] Model not found in app state, initializing...")
        try:
            if _MODEL_PATH.exists():
                model = DelayModel.load(_MODEL_PATH)
                app.state.model = model
                app.state.allowed_opera = set(model.known_operators)
                print("[lazy-init] Model loaded successfully.")
            else:
                model = DelayModel()
                app.state.model = model
                app.state.allowed_opera = set()
        except Exception as exc:
            model = DelayModel()
            app.state.model = model
            app.state.allowed_opera = set()
            print(f"[lazy-init] Model load failed: {exc}. Using unfitted model fallback.")
    return model


def _get_allowed_opera() -> set[str]:
    allowed = getattr(app.state, "allowed_opera", None)
    if allowed is None:
        m = getattr(app.state, "model", None)
        if m is not None and getattr(m, "known_operators", None):
            allowed = set(m.known_operators)
            app.state.allowed_opera = allowed
        else:
            allowed = set()
            app.state.allowed_opera = allowed
    return allowed


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.get("/info", status_code=200)
async def get_info() -> dict:
    m = _get_model()
    return {
        "status": "OK",
        "model_version": MODEL_VERSION,
        "operators_count": len(_get_allowed_opera()),
    }


def _validate_rows(req: PredictRequest) -> List[dict]:
    rows: List[dict] = []
    allowed_opera = _get_allowed_opera()

    for idx, f in enumerate(req.flights):
        opera = str(f.OPERA)
        tipo = str(f.TIPOVUELO)
        mes = int(f.MES)

        if tipo not in _ALLOWED_TIPOS:
            raise HTTPException(status_code=400, detail=f"Flight #{idx}: TIPOVUELO must be one of {_ALLOWED_TIPOS}.")
        if mes not in _ALLOWED_MES:
            raise HTTPException(status_code=400, detail=f"Flight #{idx}: MES must be 1..12.")

        if allowed_opera and opera not in allowed_opera:
            raise HTTPException(status_code=400, detail=f"Flight #{idx}: unknown OPERA '{opera}'.")

        rows.append({"OPERA": opera, "TIPOVUELO": tipo, "MES": mes})
    return rows


@app.post("/predict", response_model=PredictResponse, status_code=200)
async def post_predict(payload: Dict[str, Any] = Body(...)) -> PredictResponse:
    if "flights" not in payload or not isinstance(payload["flights"], list):
        raise HTTPException(status_code=400, detail="Body must include a 'flights' list.")

    req = PredictRequest.model_validate(payload)
    rows = _validate_rows(req)

    model = _get_model()
    df = pd.DataFrame(rows)
    feats = model.preprocess(df)
    preds = model.predict(feats)
    return PredictResponse(predict=preds)

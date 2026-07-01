"""FastAPI server exposing inference endpoints for the log classifier."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.config import load_config
from src.inference import LogClassifierInference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_engine: LogClassifierInference | None = None


def _load_engine() -> LogClassifierInference | None:
    config = load_config()
    model_path = Path(config.model_save_dir) / f"{config.model_type}_pipeline.joblib"
    labels_path = Path(config.model_save_dir) / "label_names.json"

    if not model_path.exists():
        logger.warning("Model not found at %s — train first with train.py", model_path)
        return None

    with open(labels_path) as f:
        label_names = json.load(f)

    return LogClassifierInference(config, str(model_path), label_names)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = _load_engine()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Log Classifier API",
    description="Classify system error logs into root cause categories.",
    version="0.1.0",
    lifespan=lifespan,
)


def _require_engine() -> LogClassifierInference:
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py to train the model first.",
        )
    return _engine


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    log_entry: str

    @field_validator("log_entry")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("log_entry must not be empty")
        return v


class BatchClassifyRequest(BaseModel):
    log_entries: List[str]

    @field_validator("log_entries")
    @classmethod
    def not_empty_list(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("log_entries must contain at least one entry")
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
def health():
    """Returns API status and whether the model is loaded."""
    return {"status": "ok", "model_loaded": _engine is not None}


@app.post("/classify", summary="Classify a single log entry")
def classify(req: ClassifyRequest):
    """
    Classify one log entry and return its predicted root cause category,
    confidence scores, and a structured issue summary.
    """
    engine = _require_engine()
    return engine.classify_single(req.log_entry)


@app.post("/classify/batch", summary="Classify multiple log entries")
def classify_batch(req: BatchClassifyRequest):
    """
    Classify a list of log entries in one call.
    Returns a result object for each entry in the same order.
    """
    engine = _require_engine()
    return engine.classify_batch(req.log_entries)

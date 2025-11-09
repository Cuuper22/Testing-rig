"""FastAPI application exposing the Coptic OCR prediction service."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.inference.predict import PredictionPipeline

LOGGER = logging.getLogger(__name__)


@lru_cache()
def _get_pipeline() -> PredictionPipeline:
    pipeline = PredictionPipeline()
    pipeline.warm_up()
    return pipeline


app = FastAPI(title="Coptic OCR Service", version="1.0.0")


@app.on_event("startup")
async def _startup() -> None:
    LOGGER.info("Initialising prediction pipeline for FastAPI service")
    _get_pipeline()


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    """Liveness probe used by the container health check."""

    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> Dict[str, str]:
    """Readiness probe ensuring the models are loaded."""

    try:
        _get_pipeline()
    except Exception as exc:  # pragma: no cover - defensive programming
        LOGGER.exception("Pipeline readiness check failed: %s", exc)
        return {"status": "error", "detail": str(exc)}
    return {"status": "ready"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """Run OCR and translation against an uploaded image."""

    contents = await file.read()
    pipeline = _get_pipeline()
    LOGGER.info("Received prediction request for %s", file.filename)
    result = pipeline.predict(contents).to_dict()
    return JSONResponse(result)


__all__ = ["app"]

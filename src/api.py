from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .data_loader import MarketDataUnavailableError
from .gpt_interpreter import interpret_signal_with_gemini
from .zone_analysis import generate_zone_analysis


app = FastAPI(title="Quant Entry Zone Analyzer", version="3.0")
FRONTEND_DIST_DIR = Path(__file__).resolve().parents[1] / "frontend" / "dist"
LOGGER = logging.getLogger("uvicorn.error")
ANALYSIS_LOCK = Lock()


class AnalyzeRequest(BaseModel):
    ticker: str
    benchmark_primary: str = "QQQ"
    benchmark_secondary: str = "SPY"
    treasury_yield_ticker: str = "^TNX"
    oil_ticker: str = "CL=F"
    start_date: str = "2012-01-01"
    end_date: str | None = None
    use_gpt_interpreter: bool = False
    gemini_model: str | None = None


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    return _generate_analysis(request)


@app.post("/interpret")
@app.post("/api/interpret")
def interpret(signal_json: dict[str, Any]) -> dict[str, Any]:
    try:
        return interpret_signal_with_gemini(signal_json)
    except (ImportError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/analyze-and-interpret")
@app.post("/api/analyze-and-interpret")
def analyze_and_interpret(request: AnalyzeRequest) -> dict[str, Any]:
    analysis = _generate_analysis(request)
    try:
        ai = interpret_signal_with_gemini(analysis, model=request.gemini_model) if request.use_gpt_interpreter else None
    except (ImportError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"zone_analysis": analysis, "ai_interpretation": ai}


def _generate_analysis(request: AnalyzeRequest) -> dict[str, Any]:
    if not ANALYSIS_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another analysis is running. Wait for it to finish and try again.")
    started_at = perf_counter()
    LOGGER.info("Starting zone analysis for %s", request.ticker.upper())
    try:
        analysis = generate_zone_analysis(
            ticker=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date,
            benchmark_primary=request.benchmark_primary,
            benchmark_secondary=request.benchmark_secondary,
            treasury_yield_ticker=request.treasury_yield_ticker,
            oil_ticker=request.oil_ticker,
        )
    except MarketDataUnavailableError as exc:
        LOGGER.warning("Market data unavailable for %s after %.2fs: %s", request.ticker.upper(), perf_counter() - started_at, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        ANALYSIS_LOCK.release()
    LOGGER.info("Completed zone analysis for %s in %.2fs", request.ticker.upper(), perf_counter() - started_at)
    return analysis


# In production FastAPI serves the Vite build; during development Vite runs separately.
if FRONTEND_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")

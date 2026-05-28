from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"


def load_env_file(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env without requiring python-dotenv."""
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def ensure_directories() -> None:
    for path in [
        OUTPUT_DIR,
        MODEL_DIR,
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "cache",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def clean_float(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return round(number, digits)


def pct(value: Any) -> str:
    number = clean_float(value)
    if number is None:
        return "n/a"
    return f"{number:.1%}"


def latest_valid_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("No data available after feature engineering.")
    return df.dropna(how="all").iloc[-1]


def confidence_label(score: float) -> str:
    if score >= 0.80:
        return "VERY_HIGH"
    if score >= 0.65:
        return "HIGH"
    if score >= 0.50:
        return "MEDIUM"
    return "LOW"

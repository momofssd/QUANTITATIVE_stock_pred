"""Probability that price is higher N sessions from now, with walk-forward evidence.

Two forecasters compete: the ticker's historical base rate (how often it was
higher N sessions later) and a regularized logistic regression on four trend
features. Each block of sessions in the walk-forward test is scored by a model
trained only on data whose labels were already known at the start of that
block. The headline probability comes from the model only when it beat the
base rate out of sample; otherwise the base rate is reported.

In testing across broad ETFs, single stocks, bonds, gold and energy at 10- and
60-session horizons, neither this model nor larger feature sets beat the base
rate out of sample, so the base rate is the usual headline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .utils import clean_float


TREND_HORIZONS = {"short_term": 10, "long_term": 60}
MIN_TRAIN_SESSIONS = 756
REFIT_EVERY = 63
MODEL_FEATURES = ["dist_ma200", "ma200_slope_20d", "return_60d", "return_120d"]
BULLISH_THRESHOLD = 0.55
BEARISH_THRESHOLD = 0.45


def trend_features(ohlc: pd.DataFrame, zones: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Causal features known at each close."""
    close = ohlc["close"]
    daily_return = close.pct_change(fill_method=None)
    frame = pd.DataFrame(
        {
            "dist_ma200": close / zones["ma200"] - 1,
            "dist_ma50": close / zones["ma50"] - 1,
            "ma50_vs_ma200": zones["ma50"] / zones["ma200"] - 1,
            "ma200_slope_20d": zones["ma200"] / zones["ma200"].shift(20) - 1,
            "return_5d": close.pct_change(5, fill_method=None),
            "return_20d": close.pct_change(20, fill_method=None),
            "return_60d": close.pct_change(60, fill_method=None),
            "return_120d": close.pct_change(120, fill_method=None),
            "stretch_atr": zones["st_dev"],
            "volatility_20d": daily_return.rolling(20).std() * np.sqrt(252),
            "drawdown_252d": close / close.rolling(252, min_periods=126).max() - 1,
            "rsi_14": features.get("rsi_14"),
            "rs_20d_change": features.get("rs_20d_change"),
            "vix_level": features.get("vix_level"),
            "vix_20d_change": features.get("vix_20d_change"),
        },
        index=ohlc.index,
    )
    return frame.astype(float).replace([np.inf, -np.inf], np.nan)


def _model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=500))


def _direction(probability: float) -> str:
    if probability >= BULLISH_THRESHOLD:
        return "BULLISH"
    if probability <= BEARISH_THRESHOLD:
        return "BEARISH"
    return "NEUTRAL"


def _reliability(skill: float | None, auc: float | None, samples: int) -> tuple[str, str]:
    if skill is None or auc is None or samples < 250:
        return "INSUFFICIENT_HISTORY", "Not enough history to test the model"
    if skill >= 0.02 and auc >= 0.56:
        return "USEFUL", "Beat the base rate out of sample"
    if skill > 0 and auc > 0.52:
        return "WEAK", "Only slightly better than the base rate"
    return "NO_SKILL", "No better than the base rate out of sample"


def trend_outlook(ohlc: pd.DataFrame, zones: pd.DataFrame, features: pd.DataFrame, horizon: int) -> dict:
    X = trend_features(ohlc, zones, features)
    close = ohlc["close"].to_numpy()
    n = len(close)
    labels = np.full(n, np.nan)
    labels[: n - horizon] = (close[horizon:] > close[: n - horizon]).astype(float)
    # Rows need the core trend features; optional macro/relative-strength columns are imputed.
    usable = X[["dist_ma200", "ma200_slope_20d", "return_120d"]].notna().all(axis=1).to_numpy()
    rows = np.flatnonzero(usable)
    if len(rows) < MIN_TRAIN_SESSIONS + horizon + REFIT_EVERY:
        return {
            "available": False,
            "horizon_sessions": horizon,
            "reason": "Needs about four years of daily history after the 200-day average forms.",
        }
    X = X[MODEL_FEATURES]
    values = X.to_numpy()

    # Walk-forward: train only on rows whose label was known when the block starts.
    oos_index: list[int] = []
    oos_probability: list[float] = []
    oos_base_rate: list[float] = []
    first_block = rows[0] + MIN_TRAIN_SESSIONS + horizon
    labeled_end = n - horizon
    for block_start in range(first_block, labeled_end, REFIT_EVERY):
        train = rows[rows <= block_start - horizon]
        block = rows[(rows >= block_start) & (rows < min(block_start + REFIT_EVERY, labeled_end))]
        if len(block) == 0 or len(np.unique(labels[train])) < 2:
            continue
        model = _model().fit(values[train], labels[train])
        oos_index.extend(block.tolist())
        oos_probability.extend(model.predict_proba(values[block])[:, 1].tolist())
        oos_base_rate.extend([float(labels[train].mean())] * len(block))

    train = rows[rows < labeled_end]
    model = _model().fit(values[train], labels[train])
    probability = float(model.predict_proba(values[[n - 1]])[:, 1][0])
    base_rate = float(labels[train].mean())

    actual = labels[oos_index]
    predicted = np.asarray(oos_probability)
    naive = np.asarray(oos_base_rate)
    brier = float(np.mean((predicted - actual) ** 2)) if len(actual) else None
    brier_naive = float(np.mean((naive - actual) ** 2)) if len(actual) else None
    skill = 1 - brier / brier_naive if brier is not None and brier_naive else None
    auc = float(roc_auc_score(actual, predicted)) if len(np.unique(actual)) == 2 else None
    bullish_calls = predicted >= BULLISH_THRESHOLD
    bearish_calls = predicted <= BEARISH_THRESHOLD
    grade, grade_label = _reliability(skill, auc, len(actual))

    calibration = []
    for low, high in ((0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)):
        mask = (predicted >= low) & (predicted < high)
        if mask.sum() >= 20:
            calibration.append(
                {
                    "predicted_range": [low, min(high, 1.0)],
                    "mean_predicted": clean_float(predicted[mask].mean()),
                    "observed_up_rate": clean_float(actual[mask].mean()),
                    "sessions": int(mask.sum()),
                }
            )

    coefficients = model[-1].coef_[0]
    drivers = sorted(
        zip(X.columns, coefficients * model[1].transform(model[0].transform(values[[n - 1]]))[0]),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    use_model = grade == "USEFUL"
    headline = probability if use_model else base_rate
    return {
        "available": True,
        "horizon_sessions": horizon,
        "probability_bullish": clean_float(headline),
        "probability_bearish": clean_float(1 - headline),
        "direction": _direction(headline),
        "probability_source": "model" if use_model else "base_rate",
        "model_probability_bullish": clean_float(probability),
        "model_direction": _direction(probability),
        "base_rate_bullish": clean_float(base_rate),
        "reliability": grade,
        "reliability_label": grade_label,
        "walk_forward": {
            "sessions_scored": len(actual),
            "start": str(ohlc.index[oos_index[0]].date()) if oos_index else None,
            "accuracy": clean_float(np.mean((predicted >= 0.5) == (actual == 1))) if len(actual) else None,
            "auc": clean_float(auc),
            "brier_score": clean_float(brier),
            "brier_score_base_rate": clean_float(brier_naive),
            "brier_skill_score": clean_float(skill),
            "bullish_calls": int(bullish_calls.sum()),
            "bullish_call_hit_rate": clean_float(actual[bullish_calls].mean()) if bullish_calls.any() else None,
            "bearish_calls": int(bearish_calls.sum()),
            "bearish_call_hit_rate": clean_float(1 - actual[bearish_calls].mean()) if bearish_calls.any() else None,
            "calibration": calibration,
        },
        "top_drivers": [
            {"feature": name, "pushes": "bullish" if contribution > 0 else "bearish", "weight": clean_float(contribution, 3)}
            for name, contribution in drivers[:4]
        ],
        "definition": (
            f"Probability that the close {horizon} sessions from now is above today's close. The model is a logistic "
            "regression on distance from the 200-day MA, its slope, and 60/120-session returns, re-fit every "
            f"{REFIT_EVERY} sessions in a walk-forward test. The headline uses the model only if it beat the base rate "
            "in that test. "
            "Consecutive sessions share most of their forward window, so the effective sample is smaller than the session count."
        ),
    }

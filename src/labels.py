from __future__ import annotations

import pandas as pd


BASIC_LABELS = {
    0: "BEARISH",
    1: "SIDEWAYS",
    2: "BULLISH",
}


def create_basic_labels(df: pd.DataFrame, horizon_days: int = 60, threshold: float = 0.08) -> pd.Series:
    future_return = df["Adj Close"].shift(-horizon_days) / df["Adj Close"] - 1
    labels = pd.Series(1, index=df.index, name="label")
    labels[future_return > threshold] = 2
    labels[future_return < -threshold] = 0
    labels[future_return.isna()] = pd.NA
    return labels.astype("Int64")


def create_direction_labels(df: pd.DataFrame, horizon_days: int = 5) -> pd.Series:
    """Return binary direction labels for a fixed forward trading-session horizon."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1.")
    forward_return = create_forward_return_target(df, horizon_days)
    labels = pd.Series(pd.NA, index=df.index, name="direction_label", dtype="Int64")
    labels.loc[forward_return.notna() & (forward_return < 0)] = 0
    labels.loc[forward_return.notna() & (forward_return >= 0)] = 1
    return labels


def create_forward_return_target(df: pd.DataFrame, horizon_days: int = 5) -> pd.Series:
    """Return the realized decimal return at a future trading-session horizon."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1.")
    return (df["Adj Close"].shift(-horizon_days) / df["Adj Close"] - 1).rename("forward_return")


def infer_regime(latest: pd.Series, probabilities: dict[str, float]) -> str:
    close_above_200 = bool(latest.get("close_above_200ma", 0))
    ma50_above_ma200 = bool(latest.get("ma50_above_ma200", 0))
    return_60d = float(latest.get("return_60d", 0) or 0)
    rs_20d = float(latest.get("rs_20d_change", 0) or 0)
    bbw = float(latest.get("bollinger_band_width", 1) or 1)
    atr_pct = float(latest.get("atr_pct", 1) or 1)
    drawdown_20d = abs(float(latest.get("max_drawdown_20d", 0) or 0))
    close_to_50 = float(latest.get("close_to_50ma_pct", 0) or 0)
    rsi = float(latest.get("rsi_14", 50) or 50)
    volume = float(latest.get("volume_vs_20d_avg", 1) or 1)

    if not close_above_200 and not ma50_above_ma200 and return_60d < -0.08:
        return "BEAR_TREND"
    if close_above_200 and ma50_above_ma200 and return_60d > 0.08 and rs_20d > 0:
        if drawdown_20d < 0.10 and close_to_50 < 0.10:
            return "UPTREND_CONSOLIDATION"
        return "BULL_TREND"
    if close_above_200 and bbw < 0.12 and atr_pct < 0.04 and rs_20d > 0:
        return "BREAKOUT_SETUP"
    if close_to_50 < -0.03 and rs_20d < 0 and volume > 1.1:
        return "TREND_DETERIORATION"
    if probabilities.get("bearish", 0) > 0.45 and rsi < 45:
        return "BEAR_MARKET_RALLY" if close_above_200 else "BEAR_TREND"
    return "SIDEWAYS_CONSOLIDATION"

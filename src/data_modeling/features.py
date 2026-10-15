from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import adx, atr, bollinger_band_width, macd, moving_average, rolling_max_drawdown, rolling_volatility, rsi, slope
from .short_term_features import SHORT_TERM_FEATURE_COLUMNS, build_short_term_features


MACRO_FEATURE_COLUMNS = [
    "vix_level",
    "vix_5d_change",
    "vix_20d_change",
    "vix_60d_change",
    "vix_20d_slope_pct",
    "ten_year_yield",
    "ten_year_yield_5d_change",
    "ten_year_yield_20d_change",
    "ten_year_yield_60d_change",
    "ten_year_yield_20d_slope",
    "oil_price",
    "oil_5d_change",
    "oil_20d_change",
    "oil_60d_change",
    "oil_20d_slope_pct",
]


FEATURE_COLUMNS = [
    "close_above_20ma",
    "close_above_50ma",
    "close_above_200ma",
    "ma20_slope",
    "ma50_slope",
    "ma200_slope",
    "close_to_50ma_pct",
    "close_to_200ma_pct",
    "ma50_above_ma200",
    "return_20d",
    "return_60d",
    "return_120d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "adx_14",
    "atr_14",
    "atr_pct",
    "bollinger_band_width",
    "rolling_vol_20d",
    "rolling_vol_60d",
    "max_drawdown_20d",
    "max_drawdown_60d",
    "volume_vs_20d_avg",
    "volume_vs_60d_avg",
    "up_day_volume_ratio",
    "volume_dry_up_score",
    "relative_strength_vs_primary",
    "rs_primary_20d_change",
    "rs_primary_60d_change",
    "rs_primary_above_50ma",
    "relative_strength_vs_spy",
    "rs_20d_change",
    "rs_60d_change",
    "rs_above_50ma",
    *MACRO_FEATURE_COLUMNS,
]


def _align_close(df: pd.DataFrame, index: pd.Index) -> pd.Series:
    return df["Adj Close"].reindex(index).ffill()


def build_features(
    ticker_df: pd.DataFrame,
    benchmark_primary_df: pd.DataFrame,
    benchmark_secondary_df: pd.DataFrame,
    vix_df: pd.DataFrame | None = None,
    treasury_yield_df: pd.DataFrame | None = None,
    oil_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = ticker_df.copy()
    close = df["Adj Close"].astype(float)
    df["close"] = close
    df["ma20"] = moving_average(close, 20)
    df["ma50"] = moving_average(close, 50)
    df["ma200"] = moving_average(close, 200)
    df["close_above_20ma"] = (close > df["ma20"]).astype(int)
    df["close_above_50ma"] = (close > df["ma50"]).astype(int)
    df["close_above_200ma"] = (close > df["ma200"]).astype(int)
    df["ma20_slope"] = slope(df["ma20"], 10) / close
    df["ma50_slope"] = slope(df["ma50"], 20) / close
    df["ma200_slope"] = slope(df["ma200"], 40) / close
    df["close_to_50ma_pct"] = close / df["ma50"] - 1
    df["close_to_200ma_pct"] = close / df["ma200"] - 1
    df["ma50_above_ma200"] = (df["ma50"] > df["ma200"]).astype(int)

    df["return_20d"] = close.pct_change(20)
    df["return_60d"] = close.pct_change(60)
    df["return_120d"] = close.pct_change(120)
    df["rsi_14"] = rsi(close)
    macd_line, signal, hist = macd(close)
    df["macd"] = macd_line / close
    df["macd_signal"] = signal / close
    df["macd_hist"] = hist / close
    df["adx_14"] = adx(df)

    df["atr_14"] = atr(df)
    df["atr_pct"] = df["atr_14"] / close
    df["bollinger_band_width"] = bollinger_band_width(close)
    df["rolling_vol_20d"] = rolling_volatility(close, 20)
    df["rolling_vol_60d"] = rolling_volatility(close, 60)
    df["max_drawdown_20d"] = rolling_max_drawdown(close, 20)
    df["max_drawdown_60d"] = rolling_max_drawdown(close, 60)

    vol = df["Volume"].astype(float)
    df["volume_vs_20d_avg"] = vol / vol.rolling(20, min_periods=20).mean()
    df["volume_vs_60d_avg"] = vol / vol.rolling(60, min_periods=60).mean()
    up_volume = vol.where(close.diff() > 0, 0).rolling(20, min_periods=20).sum()
    total_volume = vol.rolling(20, min_periods=20).sum()
    df["up_day_volume_ratio"] = up_volume / total_volume.replace(0, np.nan)
    df["volume_dry_up_score"] = 1 - (vol.rolling(10, min_periods=10).mean() / vol.rolling(60, min_periods=60).mean())

    primary_close = _align_close(benchmark_primary_df, df.index)
    secondary_close = _align_close(benchmark_secondary_df, df.index)
    rs_primary = close / primary_close
    rs_secondary = close / secondary_close
    df["relative_strength_vs_primary"] = rs_primary
    df["rs_primary_20d_change"] = rs_primary.pct_change(20)
    df["rs_primary_60d_change"] = rs_primary.pct_change(60)
    df["rs_primary_above_50ma"] = (rs_primary > rs_primary.rolling(50, min_periods=50).mean()).astype(int)
    df["relative_strength_vs_spy"] = rs_secondary
    df["rs_20d_change"] = rs_secondary.pct_change(20)
    df["rs_60d_change"] = rs_secondary.pct_change(60)
    df["rs_above_50ma"] = (rs_secondary > rs_secondary.rolling(50, min_periods=50).mean()).astype(int)

    if vix_df is not None and not vix_df.empty:
        vix_close = _align_close(vix_df, df.index)
        df["vix_level"] = vix_close
        df["vix_5d_change"] = vix_close.pct_change(5)
        df["vix_20d_change"] = vix_close.pct_change(20)
        df["vix_60d_change"] = vix_close.pct_change(60)
        df["vix_20d_slope_pct"] = slope(vix_close, 20) / vix_close
    else:
        df["vix_level"] = np.nan
        df["vix_5d_change"] = np.nan
        df["vix_20d_change"] = np.nan
        df["vix_60d_change"] = np.nan
        df["vix_20d_slope_pct"] = np.nan

    if treasury_yield_df is not None and not treasury_yield_df.empty:
        yield_close = _align_close(treasury_yield_df, df.index)
        df["ten_year_yield"] = yield_close
        df["ten_year_yield_5d_change"] = yield_close.diff(5)
        df["ten_year_yield_20d_change"] = yield_close.diff(20)
        df["ten_year_yield_60d_change"] = yield_close.diff(60)
        df["ten_year_yield_20d_slope"] = slope(yield_close, 20)
    else:
        df["ten_year_yield"] = np.nan
        df["ten_year_yield_5d_change"] = np.nan
        df["ten_year_yield_20d_change"] = np.nan
        df["ten_year_yield_60d_change"] = np.nan
        df["ten_year_yield_20d_slope"] = np.nan

    if oil_df is not None and not oil_df.empty:
        oil_close = _align_close(oil_df, df.index)
        df["oil_price"] = oil_close
        df["oil_5d_change"] = oil_close.pct_change(5)
        df["oil_20d_change"] = oil_close.pct_change(20)
        df["oil_60d_change"] = oil_close.pct_change(60)
        df["oil_20d_slope_pct"] = slope(oil_close, 20) / oil_close
    else:
        df["oil_price"] = np.nan
        df["oil_5d_change"] = np.nan
        df["oil_20d_change"] = np.nan
        df["oil_60d_change"] = np.nan
        df["oil_20d_slope_pct"] = np.nan

    df["prior_120d_low"] = close.rolling(120, min_periods=60).min()
    df["prior_120d_high"] = close.rolling(120, min_periods=60).max()
    df["prior_252d_high"] = close.rolling(252, min_periods=120).max()
    short_features = build_short_term_features(df, primary_close, secondary_close)
    return pd.concat([df, short_features], axis=1)


def model_matrix(features: pd.DataFrame) -> pd.DataFrame:
    return features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def short_term_model_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Add short-horizon indicators without changing the long-horizon feature set."""
    return features[FEATURE_COLUMNS + SHORT_TERM_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

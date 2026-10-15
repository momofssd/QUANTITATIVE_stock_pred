"""Causal daily features for short-horizon return and gain prediction."""

import numpy as np
import pandas as pd

from .indicators import money_flow_index, stochastic_oscillator


SHORT_TERM_FEATURE_COLUMNS = [
    "return_1d", "return_3d", "return_5d", "return_10d",
    "ema_5_to_20_pct", "close_to_ema10_pct", "rsi_14_change_3d",
    "macd_hist_change_3d", "stochastic_k_14", "stochastic_d_3",
    "bollinger_percent_b", "price_zscore_20d", "atr_5_pct",
    "realized_vol_5d", "volatility_ratio_5_to_20",
    "overnight_gap_pct", "intraday_return", "daily_range_pct",
    "close_location_value", "distance_to_prior_20d_high", "distance_to_prior_20d_low",
    "volume_ratio_5_to_20", "obv_flow_5d", "money_flow_index_14",
    "chaikin_money_flow_20", "close_to_vwap20_pct",
    "excess_return_primary_5d", "excess_return_secondary_5d",
    "cci_20", "plus_di_14", "minus_di_14", "di_spread_14",
    "efficiency_ratio_10", "downside_vol_10d", "volume_zscore_20",
    "upper_wick_fraction", "lower_wick_fraction", "donchian_width_20_pct",
]


def build_short_term_features(features: pd.DataFrame, primary_close: pd.Series, secondary_close: pd.Series) -> pd.DataFrame:
    """Return additional features using only information available by each close.

    OHLC is put on the adjusted-close basis before mixing price fields. The
    volume-weighted price is a rolling daily typical-price proxy, not intraday VWAP.
    """
    close = features["close"]
    factor = close / features["Close"].replace(0, np.nan)
    high, low, opened = (features[column] * factor for column in ("High", "Low", "Open"))
    volume = features["Volume"].astype(float)
    day_range = (high - low).replace(0, np.nan)
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    mean20, std20 = close.rolling(20).mean(), close.rolling(20).std().replace(0, np.nan)
    daily_return = close.pct_change(fill_method=None)
    vol5, vol20 = daily_return.rolling(5).std(), daily_return.rolling(20).std()
    typical = (high + low + close) / 3
    location = ((2 * close - high - low) / day_range).fillna(0)
    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    stochastic_k, stochastic_d = stochastic_oscillator(high, low, close)
    volume20 = volume.rolling(20).sum().replace(0, np.nan)
    vwap20 = (typical * volume).rolling(20).sum() / volume20
    mean_deviation = typical.rolling(20).apply(lambda window: np.abs(window - window.mean()).mean(), raw=True)
    # Simple rolling smoothing matches this project's existing ADX convention.
    up, down = high.diff(), -low.diff()
    tr14 = true_range.rolling(14).mean()
    plus_di = 100 * up.where((up > down) & (up > 0), 0).rolling(14).mean() / tr14.replace(0, np.nan)
    minus_di = 100 * down.where((down > up) & (down > 0), 0).rolling(14).mean() / tr14.replace(0, np.nan)
    path_length = close.diff().abs().rolling(10).sum()
    volume_std = volume.rolling(20).std()
    values = {f"return_{period}d": close.pct_change(period, fill_method=None) for period in (1, 3, 5, 10)}
    values.update({
        "ema_5_to_20_pct": ema5 / ema20 - 1,
        "close_to_ema10_pct": close / ema10 - 1,
        "rsi_14_change_3d": features["rsi_14"].diff(3),
        "macd_hist_change_3d": features["macd_hist"].diff(3),
        "stochastic_k_14": stochastic_k,
        "stochastic_d_3": stochastic_d,
        "bollinger_percent_b": (close - (mean20 - 2 * std20)) / (4 * std20),
        "price_zscore_20d": (close - mean20) / std20,
        "atr_5_pct": true_range.rolling(5).mean() / close,
        "realized_vol_5d": vol5 * np.sqrt(252),
        "volatility_ratio_5_to_20": vol5 / vol20.replace(0, np.nan),
        "overnight_gap_pct": opened / close.shift() - 1,
        "intraday_return": close / opened.replace(0, np.nan) - 1,
        "daily_range_pct": (high - low) / close,
        "close_location_value": location,
        "distance_to_prior_20d_high": close / high.shift().rolling(20).max() - 1,
        "distance_to_prior_20d_low": close / low.shift().rolling(20).min() - 1,
        "volume_ratio_5_to_20": volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, np.nan),
        "obv_flow_5d": (np.sign(close.diff()) * volume).rolling(5).sum() / volume.rolling(5).sum().replace(0, np.nan),
        "money_flow_index_14": money_flow_index(high, low, close, volume),
        "chaikin_money_flow_20": (location * volume).rolling(20).sum() / volume20,
        "close_to_vwap20_pct": close / vwap20.replace(0, np.nan) - 1,
        "excess_return_primary_5d": values["return_5d"] - primary_close.pct_change(5, fill_method=None),
        "excess_return_secondary_5d": values["return_5d"] - secondary_close.pct_change(5, fill_method=None),
        "cci_20": ((typical - typical.rolling(20).mean()) / (0.015 * mean_deviation.replace(0, np.nan))).mask(mean_deviation == 0, 0),
        "plus_di_14": plus_di.mask(tr14 == 0, 0),
        "minus_di_14": minus_di.mask(tr14 == 0, 0),
        "di_spread_14": (plus_di - minus_di).mask(tr14 == 0, 0),
        "efficiency_ratio_10": (close.diff(10).abs() / path_length.replace(0, np.nan)).mask(path_length == 0, 0),
        "downside_vol_10d": np.sqrt(daily_return.clip(upper=0).pow(2).rolling(10).mean() * 252),
        "volume_zscore_20": ((volume - volume.rolling(20).mean()) / volume_std.replace(0, np.nan)).mask(volume_std == 0, 0),
        "upper_wick_fraction": ((high - pd.concat([opened, close], axis=1).max(axis=1)) / day_range).fillna(0).clip(0, 1),
        "lower_wick_fraction": ((pd.concat([opened, close], axis=1).min(axis=1) - low) / day_range).fillna(0).clip(0, 1),
        "donchian_width_20_pct": (high.rolling(20).max() - low.rolling(20).min()) / close,
    })
    return pd.DataFrame(values, index=features.index)[SHORT_TERM_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, MACRO_FEATURE_COLUMNS, build_features, model_matrix


def _ohlcv(close: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": close,
            "Volume": np.full(len(index), 1_000_000),
        },
        index=index,
    )


def test_macro_momentum_and_trend_features_enter_model_matrix():
    index = pd.date_range("2024-01-02", periods=260, freq="B")
    target = _ohlcv(np.linspace(100, 140, len(index)), index)
    benchmark = _ohlcv(np.linspace(100, 120, len(index)), index)
    vix = _ohlcv(np.linspace(12, 20, len(index)), index)
    treasury = _ohlcv(np.linspace(3.5, 4.5, len(index)), index)
    oil = _ohlcv(np.linspace(95, 70, len(index)), index)

    features = build_features(target, benchmark, benchmark, vix, treasury, oil)
    X = model_matrix(features)
    latest = X.iloc[-1]

    for name in [
        "vix_5d_change",
        "vix_60d_change",
        "vix_20d_slope_pct",
        "ten_year_yield_5d_change",
        "ten_year_yield_20d_slope",
        "oil_5d_change",
        "oil_20d_slope_pct",
    ]:
        assert name in FEATURE_COLUMNS
        assert name in MACRO_FEATURE_COLUMNS
        assert name in X.columns
    assert latest["vix_5d_change"] > 0
    assert latest["vix_20d_slope_pct"] > 0
    assert latest["ten_year_yield_20d_slope"] > 0
    assert latest["oil_5d_change"] < 0
    assert latest["oil_20d_slope_pct"] < 0

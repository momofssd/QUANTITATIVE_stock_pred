import numpy as np
import pandas as pd

from src.data_modeling import FEATURE_COLUMNS, SHORT_TERM_FEATURE_COLUMNS, build_features, model_matrix, short_term_model_matrix
from src.data_modeling.indicators import money_flow_index, stochastic_oscillator
from tests.test_macro_features import _ohlcv


def test_short_term_feature_values_and_no_future_leakage():
    index = pd.date_range("2020-01-01", periods=300, freq="B")
    steps = np.arange(len(index))
    target = _ohlcv(100 + steps * 0.05 + 3 * np.sin(steps / 4), index)
    benchmark = _ohlcv(100 + steps * 0.03, index)
    features = build_features(target, benchmark, benchmark)
    prefix = build_features(target.iloc[:260], benchmark.iloc[:260], benchmark.iloc[:260])
    pd.testing.assert_frame_equal(features.loc[prefix.index], prefix)
    short_X = short_term_model_matrix(features)
    assert short_X.columns.tolist() == FEATURE_COLUMNS + SHORT_TERM_FEATURE_COLUMNS
    assert len(SHORT_TERM_FEATURE_COLUMNS) == 38
    assert model_matrix(features).columns.tolist() == FEATURE_COLUMNS
    assert np.isfinite(short_X[SHORT_TERM_FEATURE_COLUMNS].tail(20).to_numpy()).all()
    latest = features.iloc[-1]
    assert np.isclose(latest.return_5d, target["Adj Close"].iloc[-1] / target["Adj Close"].iloc[-6] - 1)
    assert 0 <= latest.stochastic_k_14 <= 100
    assert 0 <= latest.money_flow_index_14 <= 100
    assert -1 <= latest.chaikin_money_flow_20 <= 1
    assert -1 <= latest.obv_flow_5d <= 1
    typical = (target.High + target.Low + target["Adj Close"]) / 3
    window = typical.tail(20)
    assert np.isclose(latest.cci_20, (window.iloc[-1] - window.mean()) / (0.015 * (window - window.mean()).abs().mean()))
    assert 0 <= latest.efficiency_ratio_10 <= 1
    assert 0 <= latest.upper_wick_fraction <= 1
    assert 0 <= latest.lower_wick_fraction <= 1
    assert latest.di_spread_14 == latest.plus_di_14 - latest.minus_di_14


def test_adjusted_ohlc_features_do_not_create_split_gaps():
    index = pd.date_range("2020-01-01", periods=280, freq="B")
    prices = 100 + np.sin(np.arange(len(index)))
    target = _ohlcv(prices, index)
    split = target.copy()
    split.loc[index[:250], ["Open", "High", "Low", "Close"]] *= 2
    original = build_features(target, target, target)
    adjusted = build_features(split, target, target)
    pd.testing.assert_frame_equal(original[SHORT_TERM_FEATURE_COLUMNS], adjusted[SHORT_TERM_FEATURE_COLUMNS])


def test_flat_price_and_zero_volume_indicators_are_neutral():
    price = pd.Series(np.full(30, 100.0))
    k, d = stochastic_oscillator(price, price, price)
    mfi = money_flow_index(price, price, price, pd.Series(np.zeros(30)))
    assert k.iloc[-1] == d.iloc[-1] == mfi.iloc[-1] == 50

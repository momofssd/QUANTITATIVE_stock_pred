import numpy as np
import pytest

from src.entry_zones import adjusted_ohlcv, compute_zone_frame
from src.trend_probability import BEARISH_THRESHOLD, BULLISH_THRESHOLD, _direction, trend_outlook
from tests.test_entry_zones import _features


@pytest.fixture(scope="module")
def frames():
    features = _features(sessions=2000)
    ohlc = adjusted_ohlcv(features)
    return ohlc, compute_zone_frame(ohlc), features


def test_outlook_probabilities_are_consistent(frames):
    outlook = trend_outlook(*frames, horizon=10)
    assert outlook["available"] is True
    assert outlook["probability_bullish"] + outlook["probability_bearish"] == pytest.approx(1, abs=1e-3)
    assert outlook["direction"] == _direction(outlook["probability_bullish"])
    assert outlook["walk_forward"]["sessions_scored"] > 250


def test_headline_falls_back_to_base_rate_without_skill(frames):
    # A random walk has no forecastable trend, so the model must not be trusted over the base rate.
    outlook = trend_outlook(*frames, horizon=10)
    assert outlook["reliability"] != "USEFUL"
    assert outlook["probability_source"] == "base_rate"
    assert outlook["probability_bullish"] == outlook["base_rate_bullish"]


def test_walk_forward_never_trains_on_unknown_labels(frames, monkeypatch):
    ohlc, zones, features = frames
    seen = []
    import src.trend_probability as module

    original = module._model

    def spy():
        model = original()
        fit = model.fit

        def recording_fit(X, y):
            seen.append(len(y))
            assert not np.isnan(y).any()
            return fit(X, y)

        model.fit = recording_fit
        return model

    monkeypatch.setattr(module, "_model", spy)
    trend_outlook(ohlc, zones, features, horizon=60)
    assert len(seen) > 2 and seen == sorted(seen)


def test_short_history_reports_unavailable():
    features = _features(sessions=600)
    ohlc = adjusted_ohlcv(features)
    outlook = trend_outlook(ohlc, compute_zone_frame(ohlc), features, horizon=10)
    assert outlook["available"] is False


def test_direction_thresholds():
    assert _direction(BULLISH_THRESHOLD) == "BULLISH"
    assert _direction(BEARISH_THRESHOLD) == "BEARISH"
    assert _direction(0.5) == "NEUTRAL"

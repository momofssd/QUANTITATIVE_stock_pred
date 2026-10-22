import json

import numpy as np
import pandas as pd
import pytest

from src.data_loader import MarketDataUnavailableError
from src.entry_zones import (
    _bracket,
    _rsi,
    adjusted_ohlcv,
    breakdown_study,
    compute_zone_frame,
    momentum_frame,
    momentum_pivot,
    momentum_signal_study,
    position_in_zone,
    reversion_study,
    swing_support,
)
from src.features import build_features
from src.zone_analysis import build_zone_report


def _ohlcv(sessions: int, seed: int = 7, drift: float = 0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-02", periods=sessions)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, sessions)))
    opened = close * (1 + rng.normal(0, 0.004, sessions))
    high = np.maximum(close, opened) * (1 + rng.uniform(0, 0.012, sessions))
    low = np.minimum(close, opened) * (1 - rng.uniform(0, 0.012, sessions))
    return pd.DataFrame(
        {
            "Open": opened,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1_000_000, 3_000_000, sessions).astype(float),
        },
        index=index,
    )


ACTIONS = {
    "TOP_SIGNAL", "TOP_FORMING", "TAKE_PROFIT_ZONE", "SUPPORT_BREAKDOWN", "BOTTOM_SIGNAL", "BOTTOM_FORMING",
    "ENTRY_ZONE_STILL_FALLING", "BELOW_ENTRY_ZONE", "APPROACHING_ENTRY", "BOUNCE_UNDERWAY", "WAIT_FOR_PULLBACK",
}


def _features(sessions: int = 1600) -> pd.DataFrame:
    target = _ohlcv(sessions)
    benchmark = _ohlcv(sessions, seed=11)
    return build_features(target, benchmark, benchmark)


def test_zone_levels_are_causal():
    features = _features()
    full = compute_zone_frame(adjusted_ohlcv(features))
    cutoff = 1200
    truncated = compute_zone_frame(adjusted_ohlcv(features.iloc[:cutoff]))
    columns = ["st_buy_low", "st_buy_high", "st_sell_low", "st_sell_high", "next_st_buy_high"]
    pd.testing.assert_frame_equal(full[columns].iloc[:cutoff], truncated[columns])


def test_short_term_zones_are_ordered_around_the_mean():
    zones = compute_zone_frame(adjusted_ohlcv(_features())).iloc[-1]
    assert zones["next_st_buy_low"] < zones["next_st_buy_high"] < zones["ema20"]
    assert zones["ema20"] < zones["next_st_sell_low"] < zones["next_st_sell_high"]


@pytest.mark.parametrize(
    ("price", "kind", "expected"),
    [
        (95, "buy", "BELOW"),
        (101, "buy", "INSIDE"),
        (102.5, "buy", "NEAR"),
        (110, "buy", "AWAY"),
        (99, "sell", "NEAR"),
        (101, "sell", "INSIDE"),
        (104, "sell", "ABOVE"),
        (80, "sell", "AWAY"),
    ],
)
def test_position_in_zone(price, kind, expected):
    position, distance = position_in_zone(price, 100, 102, near_band=1.5, kind=kind)
    assert position == expected
    assert (distance == 0) == (expected == "INSIDE")


def test_bracket_counts_same_session_hit_as_stop():
    highs = np.array([100.0, 106.0, 100.0])
    lows = np.array([100.0, 94.0, 100.0])
    outcome, *_ = _bracket(highs, lows, 0, 2, target=105, stop=95, side="buy", include_entry_day=False)
    assert outcome == "failed"


def test_bracket_success_and_pending():
    highs = np.array([100.0, 103.0, 106.0])
    lows = np.array([99.0, 99.0, 101.0])
    outcome, day, *_ = _bracket(highs, lows, 0, 2, target=105, stop=95, side="buy", include_entry_day=True)
    assert (outcome, day) == ("success", 2)
    outcome, *_ = _bracket(highs, lows, 1, 5, target=200, stop=50, side="buy", include_entry_day=True)
    assert outcome == "pending"


def test_rsi_reads_100_without_losses():
    assert _rsi(pd.Series([1.0, 2.0, 3.0, 4.0]), 2).iloc[-1] == 100.0


def test_reversion_study_reports_baseline_and_edge():
    features = _features()
    ohlc = adjusted_ohlcv(features)
    study = reversion_study(ohlc, compute_zone_frame(ohlc), "buy")
    assert study["episodes"] >= 8
    assert 0 <= study["success_rate"] <= 1
    assert 0 <= study["baseline_success_rate"] <= 1
    assert study["edge_vs_baseline"] == pytest.approx(study["success_rate"] - study["baseline_success_rate"], abs=1e-3)
    assert all(episode["outcome"] in {"success", "failed", "expired", "pending"} for episode in study["recent_episodes"])


def test_zone_report_shape_and_json_serializable():
    report = build_zone_report(_features(), ticker="test")
    json.dumps(report)
    assert report["ticker"] == "TEST"
    assert report["analysis_type"] == "price_zones"
    short_term = report["short_term"]
    assert "long_term" not in report
    assert short_term["action"] in ACTIONS
    assert short_term["buy_zone"]["low"] < short_term["buy_zone"]["high"]
    plan = short_term["trade_plan"]
    assert plan["stop_loss"] < plan["entry_reference"] < plan["target_1"] <= plan["target_2"]
    assert set(short_term["history"]) == {"buy_zone", "momentum_bottom", "sell_zone", "momentum_top", "breakdown"}
    assert "recent_episodes" not in short_term["history"]["momentum_bottom"]
    assert short_term["momentum"]["phase"] in {"BEARISH_BUILDING", "BEARISH_FADING", "BULLISH_BUILDING", "BULLISH_FADING"}
    assert "recent_episodes" not in short_term["history"]["breakdown"]
    assert short_term["breakdown"]["status"] in {"NONE", "ACTIVE", "RECLAIMED"}
    assert "recent_episodes" not in short_term["history"]["buy_zone"]
    assert len(short_term["bounce_readiness"]["checks"]) == short_term["bounce_readiness"]["max_score"]
    points = report["chart_data"]["points"]
    assert len(points) == min(len(_features()), 2520)
    assert set(report["trend_outlook"]) == {"short_term", "long_term"}
    assert {"st_buy_low", "st_buy_high", "st_sell_high", "stretch", "stretch_p05", "momentum"} <= set(points[-1])
    assert not any(key.startswith("lt_") for key in points[-1])


def test_positive_tone_requires_historical_edge():
    report = build_zone_report(_features(), ticker="test")
    short_term = report["short_term"]
    if short_term["tone"] == "positive":
        assert short_term["historical_edge"]["grade"] in {"EDGE", "SLIGHT_EDGE"}


def test_short_history_still_builds_short_term_zones():
    report = build_zone_report(_features(sessions=400), ticker="test")
    assert report["short_term"]["buy_zone"]["low"] is not None


def test_zone_report_rejects_too_little_history():
    with pytest.raises(MarketDataUnavailableError):
        build_zone_report(_features(sessions=200), ticker="test")


def test_crash_through_support_waits_for_reclaim_above_current_price():
    features = _features()
    crash = features.copy()
    for column in ("Open", "High", "Low", "Close", "Adj Close"):
        crash.iloc[-1, crash.columns.get_loc(column)] = crash[column].iloc[-2] * 0.80
    crash["close"] = crash["Adj Close"]
    report = build_zone_report(crash, ticker="test")
    short_term = report["short_term"]
    assert short_term["buy_zone"]["position"] == "BELOW"
    # The crash also closes through swing-low support, which takes precedence over the zone.
    assert short_term["action"] == "SUPPORT_BREAKDOWN"
    plan = short_term["trade_plan"]
    assert plan["entry_reference"] == short_term["breakdown"]["broken_level"]
    assert plan["entry_reference"] > report["current_market"]["current_price"]
    assert plan["stop_loss"] < plan["entry_reference"] < plan["target_1"]


def test_swing_support_is_causal():
    ohlc = adjusted_ohlcv(_features())
    cutoff = 1200
    full = swing_support(ohlc).iloc[:cutoff]
    truncated = swing_support(ohlc.iloc[:cutoff])
    pd.testing.assert_series_equal(full, truncated)


def test_breakdown_study_uses_sell_side_baseline():
    ohlc = adjusted_ohlcv(_features())
    zones = compute_zone_frame(ohlc)
    sell = reversion_study(ohlc, zones, "sell")
    study = breakdown_study(ohlc, zones, sell["baseline_success_rate"])
    assert study["episodes"] >= 8
    assert study["baseline_success_rate"] == sell["baseline_success_rate"]
    assert all(episode["entry"] < episode["support"] for episode in study["recent_episodes"])


def _breakdown_features(last_close: float) -> tuple[pd.DataFrame, float]:
    """Dip to a swing low, rally, then close at `last_close` times the pre-pattern price."""
    features = _features()
    base = float(features["Close"].iloc[-21])
    path = [1.0 - 0.01 * step for step in range(6)] + [0.95 + 0.01 * step for step in range(1, 8)] + [1.02] * 6 + [last_close]
    closes = base * np.array(path)
    rows = features.index[-len(closes):]
    for column, values in (("Open", closes), ("High", closes * 1.002), ("Low", closes * 0.998), ("Close", closes), ("Adj Close", closes)):
        features.loc[rows, column] = values
    features["close"] = features["Adj Close"]
    return features, base * 0.95 * 0.998


def test_close_below_swing_low_reports_support_breakdown():
    features, support = _breakdown_features(0.935)
    short_term = build_zone_report(features, ticker="test")["short_term"]
    breakdown = short_term["breakdown"]
    assert short_term["action"] == "SUPPORT_BREAKDOWN"
    assert short_term["tone"] == "negative" and short_term["is_entry_now"] is False
    assert breakdown["active"] and breakdown["sessions_since_breakdown"] == 0
    assert breakdown["broken_level"] == pytest.approx(support, abs=0.01)
    assert breakdown["follow_through_target"] < float(features["close"].iloc[-1])
    plan = short_term["trade_plan"]
    assert plan["entry_reference"] == breakdown["broken_level"]
    assert plan["stop_loss"] < plan["entry_reference"] < plan["target_1"]


def test_close_above_swing_low_is_not_a_breakdown():
    features, _ = _breakdown_features(0.96)
    short_term = build_zone_report(features, ticker="test")["short_term"]
    assert short_term["action"] != "SUPPORT_BREAKDOWN"
    assert short_term["breakdown"]["active"] is False


def test_hourly_bars_group_into_two_adjusted_4h_candles_per_session():
    from src.zone_analysis import _four_hour_bars

    hours = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"]
    index = pd.to_datetime([f"2026-05-{day} {hour}" for day in (26, 27) for hour in hours])
    prices = np.arange(len(index), dtype=float) + 100
    intraday = pd.DataFrame({"Open": prices, "High": prices + 1, "Low": prices - 1, "Close": prices + 0.5, "Volume": 10.0}, index=index)
    # Day one has a dividend adjustment of 0.98; day two none. Hourly bars after the daily history are dropped.
    features = pd.DataFrame({"Close": [200.0, 210.0], "close": [196.0, 210.0]}, index=pd.to_datetime(["2026-05-26", "2026-05-27"]))
    later = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=pd.to_datetime(["2026-05-28 09:30"]))
    bars = _four_hour_bars(pd.concat([intraday, later]), features)
    assert [(bar["date"], bar["time"]) for bar in bars] == [
        ("2026-05-26", "09:30"), ("2026-05-26", "13:30"), ("2026-05-27", "09:30"), ("2026-05-27", "13:30"),
    ]
    first, second = bars[0], bars[1]
    assert first["open"] == pytest.approx(100 * 0.98) and first["close"] == pytest.approx(103.5 * 0.98)
    assert first["high"] == pytest.approx(104 * 0.98) and first["volume"] == 40
    # The afternoon candle runs to the close, including Yahoo's 16:00 closing print.
    assert second["low"] == pytest.approx(103 * 0.98) and second["close"] == pytest.approx(107.5 * 0.98) and second["volume"] == 40
    assert bars[2]["open"] == pytest.approx(108)


def test_momentum_and_signals_are_causal():
    ohlc = adjusted_ohlcv(_features())
    cutoff = 1200
    full = momentum_frame(ohlc, compute_zone_frame(ohlc)).iloc[:cutoff]
    truncated = momentum_frame(ohlc.iloc[:cutoff], compute_zone_frame(ohlc.iloc[:cutoff]))
    pd.testing.assert_frame_equal(full, truncated)


def test_momentum_pivot_is_the_close_that_leaves_momentum_unchanged():
    ohlc = adjusted_ohlcv(_features())
    zones = compute_zone_frame(ohlc)
    pivot = momentum_pivot(zones)
    nxt = ohlc.index[-1] + pd.offsets.BDay()
    extended = pd.concat([ohlc, pd.DataFrame({"open": pivot, "high": pivot, "low": pivot, "close": pivot, "volume": 1.0}, index=[nxt])])
    after = compute_zone_frame(extended)
    spread = after["ema_fast"] - after["ema_slow"]
    # Momentum is this spread over ATR, so an unchanged spread is unchanged momentum when ATR holds.
    assert spread.iloc[-1] == pytest.approx(spread.iloc[-2], abs=1e-9)


def test_signals_fire_only_after_a_zone_touch_and_a_two_session_momentum_turn():
    ohlc = adjusted_ohlcv(_features(sessions=2500))
    zones = compute_zone_frame(ohlc)
    frame = momentum_frame(ohlc, zones)
    momentum, change = frame["momentum"].to_numpy(), frame["change"].to_numpy()
    lows, highs = ohlc["low"].to_numpy(), ohlc["high"].to_numpy()
    buy_band, sell_band = zones["st_buy_high"].to_numpy(), zones["st_sell_low"].to_numpy()
    buys, sells = np.flatnonzero(frame["buy_signal"]), np.flatnonzero(frame["sell_signal"])
    assert len(buys) >= 20 and len(sells) >= 20
    for i in buys:
        assert momentum[i] < 0 and change[i] > 0 and change[i - 1] > 0 and not change[i - 2] > 0
        assert (lows[i - 4:i + 1] <= buy_band[i - 4:i + 1]).any()
    for i in sells:
        assert momentum[i] > 0 and change[i] < 0 and change[i - 1] < 0 and not change[i - 2] < 0
        assert (highs[i - 4:i + 1] >= sell_band[i - 4:i + 1]).any()


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_momentum_signal_replay_uses_the_interim_extreme_stop_and_a_2r_target(side):
    ohlc = adjusted_ohlcv(_features(sessions=2500))
    zones = compute_zone_frame(ohlc)
    study = momentum_signal_study(ohlc, zones, side)
    assert study["episodes"] >= 8
    assert study["edge_r"] == pytest.approx(study["expectancy_r"] - study["baseline_expectancy_r"], abs=0.011)
    assert 0 <= study["held_rate"] <= 1 and 0 <= study["baseline_held_rate"] <= 1
    sign = 1 if side == "buy" else -1
    for episode in study["recent_episodes"]:
        entry, stop, target = episode["entry"], episode["stop"], episode["target"]
        # The stop sits beyond the interim extreme, and the target is twice the risk.
        assert sign * (episode["interim_extreme"] - stop) > 0
        assert sign * (target - entry) == pytest.approx(2 * sign * (entry - stop))
        if episode["outcome"] == "success":
            assert episode["r_multiple"] == pytest.approx(2.0)
        elif episode["outcome"] == "failed":
            assert episode["r_multiple"] <= -1 + 1e-9


def _dip_features(rebound: list[float]) -> pd.DataFrame:
    """A 12-session slide of 1.5% a day, then the given daily returns."""
    features = _features()
    returns = [-0.015] * 12 + rebound
    closes = float(features["Close"].iloc[-len(returns) - 1]) * np.cumprod(1 + np.array(returns))
    rows = features.index[-len(closes):]
    opens = np.concatenate([[closes[0] / (1 + returns[0])], closes[:-1]])
    for column, values in (
        ("Open", opens), ("High", np.maximum(opens, closes) * 1.002), ("Low", np.minimum(opens, closes) * 0.998),
        ("Close", closes), ("Adj Close", closes),
    ):
        features.loc[rows, column] = values
    features["close"] = features["Adj Close"]
    return features


def test_dip_whose_bearish_momentum_fades_for_two_sessions_is_a_bottom_signal():
    short_term = build_zone_report(_dip_features([0.006, 0.006]), ticker="test")["short_term"]
    momentum = short_term["momentum"]
    assert momentum["phase"] == "BEARISH_FADING" and momentum["rising_sessions"] >= 2
    assert momentum["signal"]["side"] == "buy" and momentum["signal"]["active"]
    assert short_term["action"] in {"BOTTOM_SIGNAL", "SUPPORT_BREAKDOWN"}
    if short_term["action"] == "BOTTOM_SIGNAL":
        plan = short_term["trade_plan"]
        assert plan["stop_loss"] < momentum["interim_low"] < plan["entry_reference"] < plan["target_1"]
        assert short_term["is_entry_now"]


def test_dip_still_falling_is_not_a_buy():
    short_term = build_zone_report(_dip_features([]), ticker="test")["short_term"]
    assert short_term["momentum"]["phase"] == "BEARISH_BUILDING"
    assert short_term["momentum"]["signal"] is None
    assert short_term["action"] in {"ENTRY_ZONE_STILL_FALLING", "BELOW_ENTRY_ZONE", "SUPPORT_BREAKDOWN"}
    assert short_term["tone"] != "positive" and not short_term["is_entry_now"]

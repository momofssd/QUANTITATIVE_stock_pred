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
    forward_return_study,
    high_rr_entry,
    long_term_plan_study,
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


def _features(sessions: int = 1600) -> pd.DataFrame:
    target = _ohlcv(sessions)
    benchmark = _ohlcv(sessions, seed=11)
    return build_features(target, benchmark, benchmark)


def test_zone_levels_are_causal():
    features = _features()
    full = compute_zone_frame(adjusted_ohlcv(features))
    cutoff = 1200
    truncated = compute_zone_frame(adjusted_ohlcv(features.iloc[:cutoff]))
    columns = ["st_buy_low", "st_buy_high", "st_sell_low", "lt_buy_low", "lt_sell_high", "next_st_buy_high", "lt_stop", "next_lt_stop"]
    pd.testing.assert_frame_equal(full[columns].iloc[:cutoff], truncated[columns])


def test_short_term_zones_are_ordered_around_the_mean():
    zones = compute_zone_frame(adjusted_ohlcv(_features())).iloc[-1]
    assert zones["next_st_buy_low"] < zones["next_st_buy_high"] < zones["ema20"]
    assert zones["ema20"] < zones["next_st_sell_low"] < zones["next_st_sell_high"]
    assert zones["next_lt_buy_low"] < zones["next_lt_buy_high"] < zones["next_lt_sell_low"] < zones["next_lt_sell_high"]


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
    short_term, long_term = report["short_term"], report["long_term"]
    assert short_term["action"] in {
        "SUPPORT_BREAKDOWN", "ENTRY_CONFIRMED", "ENTRY_ZONE_BOUNCE_FORMING", "ENTRY_ZONE", "BELOW_ENTRY_ZONE",
        "APPROACHING_ENTRY", "BOUNCE_UNDERWAY", "WAIT_FOR_PULLBACK", "TAKE_PROFIT_ZONE",
    }
    assert short_term["buy_zone"]["low"] < short_term["buy_zone"]["high"]
    plan = short_term["trade_plan"]
    assert plan["stop_loss"] < plan["entry_reference"] < plan["target_1"] <= plan["target_2"]
    assert {"buy_zone", "confirmed_entry", "sell_zone", "breakdown"} <= set(short_term["history"])
    assert "recent_episodes" not in short_term["history"]["breakdown"]
    assert short_term["breakdown"]["status"] in {"NONE", "ACTIVE", "RECLAIMED"}
    assert "recent_episodes" not in short_term["history"]["buy_zone"]
    assert long_term["available"] is True
    assert len(short_term["bounce_readiness"]["checks"]) == short_term["bounce_readiness"]["max_score"]
    points = report["chart_data"]["points"]
    assert len(points) == min(len(_features()), 2520)
    assert set(report["trend_outlook"]) == {"short_term", "long_term"}
    assert {"st_buy_low", "st_buy_high", "lt_sell_high", "stretch", "stretch_p05"} <= set(points[-1])


def test_positive_tone_requires_historical_edge():
    report = build_zone_report(_features(), ticker="test")
    short_term = report["short_term"]
    if short_term["tone"] == "positive":
        assert short_term["historical_edge"]["grade"] in {"EDGE", "SLIGHT_EDGE"}


def test_long_term_zones_unavailable_on_short_history():
    report = build_zone_report(_features(sessions=400), ticker="test")
    assert report["long_term"]["available"] is False
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


def test_high_rr_entry_gives_the_requested_reward_to_risk_capped_at_the_zone():
    entry = high_rr_entry(80.0, 120.0, 200.0, 3.0)
    assert (120 - entry) / (entry - 80) == pytest.approx(3.0)
    assert high_rr_entry(80.0, 120.0, 85.0, 3.0) == 85.0


def test_long_term_zone_replay_reports_any_day_baseline():
    ohlc = adjusted_ohlcv(_features())
    study = forward_return_study(ohlc, compute_zone_frame(ohlc), "buy")
    assert 0 <= study["baseline_success_rate"] <= 1
    assert study["edge_vs_baseline"] == pytest.approx(study["success_rate"] - study["baseline_success_rate"], abs=1e-3)


def test_long_term_plan_study_only_adds_after_a_confirmed_dip():
    ohlc = adjusted_ohlcv(_features(sessions=2500))
    zones = compute_zone_frame(ohlc)
    study = long_term_plan_study(ohlc, zones)
    assert study["episodes"] >= 3
    assert study["edge_r"] == pytest.approx(study["expectancy_r"] - study["baseline_expectancy_r"], abs=0.011)
    assert 0 <= study["probability_no_better_than_baseline"] <= 1
    highs, lows, atr = ohlc["high"].to_numpy(), ohlc["low"].to_numpy(), zones["atr"].to_numpy()
    for episode in study["recent_episodes"]:
        i = episode["index"]
        # The add is a close above the prior 10-session high, after the dip, with the stop under the dip's low.
        assert episode["entry"] > highs[i - 10:i].max()
        assert episode["stop"] < episode["dip_low"] <= lows[i]
        assert episode["stop"] >= episode["dip_low"] - 0.5 * atr[i] - 1e-9
        assert episode["planned_rr"] >= 2.0 - 1e-9
        if episode["outcome"] == "success":
            assert episode["r_multiple"] == pytest.approx(episode["planned_rr"])
        elif episode["outcome"] == "failed":
            assert episode["r_multiple"] <= -1 + 1e-9


def _long_term_report(price_from_plan):
    """Re-run the report with the last close moved to a price chosen from the original plan."""
    features = _features(sessions=2500)
    plan = build_zone_report(features, ticker="test")["long_term"]["trade_plan"]
    moved = features.copy()
    price = price_from_plan(plan)
    for column in ("Open", "High", "Low", "Close", "Adj Close"):
        moved.iloc[-1, moved.columns.get_loc(column)] = price
    moved["close"] = moved["Adj Close"]
    return build_zone_report(moved, ticker="test")["long_term"]


def _dip_report(last_close: float):
    """Fall from 10% above the buy area to 7% below it over 24 sessions, then close at `last_close` x buy area."""
    features = _features(sessions=2500)
    buy_area = build_zone_report(features, ticker="test")["long_term"]["trade_plan"]["buy_area"]
    path = np.concatenate([np.linspace(1.10, 0.95, 15), np.linspace(0.945, 0.93, 9), [last_close]]) * buy_area
    rows = features.index[-len(path):]
    for column, values in (("Open", path), ("High", path * 1.003), ("Low", path * 0.997), ("Close", path), ("Adj Close", path)):
        features.loc[rows, column] = values
    features["close"] = features["Adj Close"]
    return build_zone_report(features, ticker="test")["long_term"]


def test_long_term_plan_fields_are_consistent():
    long_term = build_zone_report(_features(sessions=2500), ticker="test")["long_term"]
    plan = long_term["trade_plan"]
    assert plan["status"] in {"ADD_CONFIRMED", "DIP_STILL_FALLING", "BOUNCED_LOW_RR", "NO_DIP"}
    assert plan["stop"] < plan["target"]
    assert plan["buy_area"] <= long_term["buy_zone"]["high"]
    assert set(long_term["history"]) == {"buy_zone", "sell_zone", "plan"}
    if long_term["tone"] == "positive":
        assert long_term["historical_edge"]["grade"] in {"EDGE", "SLIGHT_EDGE"}


def test_dip_that_keeps_falling_is_not_an_add():
    long_term = _dip_report(0.92)
    assert long_term["action"] == "DIP_STILL_FALLING"
    assert long_term["tone"] != "positive"
    assert long_term["trade_plan"]["confirm_above"] > 0.92 * long_term["trade_plan"]["buy_area"]


def test_dip_that_closes_above_the_prior_high_is_an_add_with_the_stop_under_the_dip():
    long_term = _dip_report(0.97)
    plan = long_term["trade_plan"]
    assert long_term["action"] == "ADD_CONFIRMED"
    assert plan["stop"] < plan["dip_low"] and plan["stop_under_dip"]
    assert plan["reward_risk_now"] >= 2.0


def test_price_below_the_long_term_stop_invalidates_the_plan():
    long_term = _long_term_report(lambda plan: plan["stop"] * 0.95)
    assert long_term["action"] == "BELOW_LONG_TERM_STOP"
    assert long_term["tone"] == "negative"
    assert long_term["trade_plan"]["reward_risk_now"] is None


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

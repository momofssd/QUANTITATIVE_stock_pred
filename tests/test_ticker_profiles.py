import json

import numpy as np
import pytest

from src import zone_analysis
from src.entry_zones import DEFAULT_SIGNALS, SignalParams, adjusted_ohlcv, compute_zone_frame, momentum_frame
from src.ticker_profiles import (
    MIN_RANK_CORRELATION,
    PROFILE_PATH,
    _decide,
    load_profiles,
    profile_for,
    snapshot_ohlc,
    tune_ticker,
)
from src.utils import PROJECT_ROOT
from src.zone_analysis import build_zone_report

from tests.test_entry_zones import _features


def _result(train_score, holdout_edge, signals=40):
    return {"train_score": train_score, "holdout": {"edge_r": holdout_edge, "signals": signals}}


def test_turn_length_sets_how_many_rising_sessions_a_bottom_signal_needs():
    ohlc = adjusted_ohlcv(_features(sessions=2500))
    for turn in (1, 3):
        signals = SignalParams(buy_turn=turn)
        frame = momentum_frame(ohlc, compute_zone_frame(ohlc, signals), signals)
        fired = frame[frame["buy_signal"]]
        assert len(fired) > 0
        assert (fired["rising_sessions"] == turn).all()


def test_ticker_without_a_profile_uses_the_defaults(tmp_path):
    assert profile_for("NOPE", tmp_path / "missing.json") == (DEFAULT_SIGNALS, None)
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"ABC": {"params": {**DEFAULT_SIGNALS.to_dict(), "fast": 8, "slow": 21}}}))
    signals, profile = profile_for("abc", path)
    assert (signals.fast, signals.slow) == (8, 21) and profile is not None


def test_tuned_setting_needs_training_win_holdout_parity_and_predictive_tuning():
    default = _result(0.10, 0.05)
    assert _decide(default, default, 0.9)[0] is False
    assert _decide(_result(0.12, 0.30), default, 0.9)[0] is False  # training win too small
    assert _decide(_result(0.30, 0.00), default, 0.9)[0] is False  # worse on the holdout
    assert _decide(_result(0.30, 0.30, signals=5), default, 0.9)[0] is False  # too few holdout signals
    assert _decide(_result(0.30, 0.30), default, MIN_RANK_CORRELATION - 0.01)[0] is False  # tuning not predictive
    assert _decide(_result(0.30, 0.30), default, MIN_RANK_CORRELATION)[0] is True


def test_tuning_produces_a_complete_profile_that_keeps_defaults_unless_adopted():
    profile = tune_ticker("test", adjusted_ohlcv(_features(sessions=2500)))
    json.dumps(profile)
    params = SignalParams(**profile["params"])
    for side in ("buy", "sell"):
        report = profile[side]
        assert report["settings_tested"] == (36 if side == "buy" else 9)
        if not report["adopted"]:
            assert report["settings"] == {key: getattr(DEFAULT_SIGNALS if side == "buy" else params, key) for key in report["settings"]}
    if not profile["buy"]["adopted"]:
        assert (params.fast, params.slow, params.buy_turn, params.buy_touch) == (5, 13, 2, 5)


def test_report_uses_the_ticker_profile_and_grades_tuned_sides_out_of_sample(monkeypatch):
    signals = SignalParams(fast=8, slow=21, buy_touch=3)
    holdout = {"signals": 40, "edge_r": 0.2, "expectancy_r": 0.3, "baseline_expectancy_r": 0.1, "held_rate": 0.6, "baseline_held_rate": 0.5}
    side = {
        "adopted": True, "reason": "test", "rank_correlation": 0.5, "settings": {}, "best_candidate": {},
        "candidate": {"train": holdout, "holdout": holdout}, "default": {"train": holdout, "holdout": holdout},
    }
    profile = {
        "params": signals.to_dict(), "trained_on": "2026-10-21", "data_start": "2012-01-03", "data_through": "2026-10-14",
        "holdout_start": "2021-01-01", "buy": side, "sell": {**side, "adopted": False},
    }
    monkeypatch.setattr(zone_analysis, "profile_for", lambda ticker: (signals, profile))
    report = build_zone_report(_features(), ticker="test")
    momentum = report["short_term"]["momentum"]
    assert (momentum["fast_span"], momentum["slow_span"], momentum["buy_touch_sessions"]) == (8, 21, 3)
    assert report["signal_profile"]["source"] == "ticker" and report["signal_profile"]["params"] == signals.to_dict()
    edge = report["short_term"]["historical_edge"]
    if report["short_term"]["action"] not in zone_analysis.SELL_ACTIONS | {"SUPPORT_BREAKDOWN"}:
        assert edge["basis"].startswith("out of sample") and edge["edge_r"] == 0.2


def test_report_without_a_profile_says_it_uses_the_defaults():
    profile = build_zone_report(_features(), ticker="test")["signal_profile"]
    assert profile["source"] == "default" and profile["params"] == DEFAULT_SIGNALS.to_dict()


STORED = load_profiles()


@pytest.mark.parametrize("ticker", sorted(STORED))
def test_stored_profile_is_reproduced_exactly_from_its_snapshot(ticker):
    """Regression test: re-train each stored profile from the same price snapshot and compare.

    Any code change that alters a ticker's signals, replay, or tuning shows up here. Snapshots
    live in the gitignored data/cache folder, so the test skips where they have not been downloaded.
    """
    stored = STORED[ticker]
    snapshot = stored["snapshot"]
    path = PROJECT_ROOT / "data" / "cache" / f"{ticker}_{snapshot['start']}_{snapshot['end']}.csv"
    if not path.exists():
        pytest.skip(f"price snapshot {path.name} not cached; run python -m src.ticker_profiles {ticker} --end {snapshot['end']}")
    fresh = tune_ticker(ticker, snapshot_ohlc(ticker, snapshot["end"], snapshot["start"]))
    assert fresh["params"] == stored["params"]
    for side in ("buy", "sell"):
        for key in ("adopted", "settings", "best_candidate", "rank_correlation", "candidate", "default"):
            assert fresh[side][key] == stored[side][key], f"{ticker} {side} {key} changed"


@pytest.mark.parametrize("ticker", sorted(STORED))
def test_stored_profile_obeys_the_adoption_rules(ticker):
    stored = STORED[ticker]
    for side in ("buy", "sell"):
        report = stored[side]
        if report["adopted"]:
            assert report["candidate"]["holdout"]["edge_r"] >= report["default"]["holdout"]["edge_r"]
            assert report["rank_correlation"] >= MIN_RANK_CORRELATION
            assert np.isfinite(report["candidate"]["train_score"])
    assert PROFILE_PATH.exists()

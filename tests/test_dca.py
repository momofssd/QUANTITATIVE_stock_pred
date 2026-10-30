import numpy as np
import pandas as pd
import pytest

from src.dca import WEEKDAYS, weekday_buy_sessions, weekly_dca_study
from src.zone_analysis import build_zone_report
from tests.test_entry_zones import _features

HOLIDAYS = pd.to_datetime(["2024-01-15", "2024-03-29", "2024-07-04"])  # a Monday, a Friday and a Thursday


def _sessions(start="2024-01-01", end="2024-12-31", holidays=HOLIDAYS):
    days = pd.bdate_range(start, end)
    return days[~days.isin(holidays)]


def _ohlc(close, sessions):
    return pd.DataFrame({"close": close}, index=sessions)


def test_a_holiday_moves_the_buy_to_the_next_session_even_into_the_next_week():
    sessions = _sessions()
    weeks, positions, moved = weekday_buy_sessions(sessions)
    bought = sessions.to_numpy()[positions]
    week = dict(zip(weeks, range(len(weeks))))
    mlk = week[pd.Timestamp("2024-01-15")]
    assert bought[mlk, 0] == pd.Timestamp("2024-01-16") and moved[mlk, 0]  # Monday holiday -> Tuesday
    good_friday = week[pd.Timestamp("2024-03-25")]
    assert bought[good_friday, 4] == pd.Timestamp("2024-04-01") and moved[good_friday, 4]  # Friday -> next Monday
    july = week[pd.Timestamp("2024-07-01")]
    assert bought[july, 3] == pd.Timestamp("2024-07-05")
    assert moved.sum() == 3
    assert (~moved[:, 1]).all()


def test_only_weeks_where_every_weekday_has_bought_are_kept():
    sessions = _sessions(start="2024-01-03", end="2024-02-14")  # starts and ends mid-week
    weeks, positions, _ = weekday_buy_sessions(sessions)
    assert weeks[0] == pd.Timestamp("2024-01-08")
    assert weeks[-1] == pd.Timestamp("2024-02-05")
    assert positions.max() < len(sessions)


def test_average_cost_is_total_invested_over_shares_bought():
    sessions = _sessions(start="2023-01-02", end="2024-12-31", holidays=[])
    close = np.where(sessions.weekday == 2, 50.0, 100.0)  # Wednesdays at half price
    study = weekly_dca_study(_ohlc(close, sessions), amount=100)
    rows = {row["weekday"]: row for row in study["weekdays"]}
    assert rows["Wednesday"]["average_cost"] == pytest.approx(50.0)
    assert rows["Monday"]["average_cost"] == pytest.approx(100.0)
    assert rows["Wednesday"]["shares"] == pytest.approx(2 * study["weeks"])
    assert study["best_weekday"] == "Wednesday" and rows["Wednesday"]["best"]
    assert rows["Wednesday"]["years_cheapest"] == study["years"] == 2


def test_a_real_weekday_dip_is_found_once_the_trend_is_removed():
    sessions = _sessions(start="2012-01-02", end="2025-12-31", holidays=[])
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0004, 0.01, len(sessions))
    close = 100 * np.exp(np.cumsum(returns)) * np.where(sessions.weekday == 2, 0.995, 1.0)
    study = weekly_dca_study(_ohlc(close, sessions))
    assert study["trend_neutral_best"] == "Wednesday"
    assert study["reliable"] and study["consistent_across_halves"]


def test_a_trend_alone_favours_the_earliest_weekday_but_is_not_a_reliable_edge():
    sessions = _sessions(start="2012-01-02", end="2025-12-31", holidays=[])
    rng = np.random.default_rng(5)
    close = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.004, len(sessions))))
    study = weekly_dca_study(_ohlc(close, sessions))
    assert study["best_weekday"] == "Monday" and study["worst_weekday"] == "Friday"
    assert not study["reliable"]
    assert "not from the weekday itself" in study["summary"]


def test_recent_year_is_simulated_on_its_own_last_52_weeks():
    sessions = _sessions(start="2020-01-01", end="2024-12-31", holidays=[])
    # Tuesdays are cheap in the last year only; before that Thursdays were.
    recent = sessions >= pd.Timestamp("2024-01-01")
    close = np.full(len(sessions), 100.0)
    close[(sessions.weekday == 3) & ~recent] = 90.0
    close[(sessions.weekday == 1) & recent] = 80.0
    study = weekly_dca_study(_ohlc(close, sessions), amount=100)
    last = study["recent"]
    assert last["weeks"] == 52 and last["invested"] == pytest.approx(5200)
    assert last["start"] == "2024-01-01" and last["end"] == study["end"]
    assert last["best_weekday"] == "Tuesday" and study["best_weekday"] == "Thursday"
    assert not last["matches_full_history"] and "Over the full history Thursday" in last["summary"]
    rows = {row["weekday"]: row for row in last["weekdays"]}
    assert rows["Tuesday"]["average_cost"] == pytest.approx(80.0)
    assert rows["Thursday"]["average_cost"] == pytest.approx(100.0)


def test_short_history_has_no_study():
    sessions = _sessions(start="2024-01-02", end="2024-06-28")
    assert weekly_dca_study(_ohlc(np.linspace(100, 110, len(sessions)), sessions)) is None


def test_report_includes_the_weekly_dca_study():
    dca = build_zone_report(_features(), ticker="test")["weekly_dca"]
    assert [row["weekday"] for row in dca["weekdays"]] == list(WEEKDAYS)
    assert sum(row["best"] for row in dca["weekdays"]) == 1
    assert dca["invested"] == pytest.approx(dca["amount"] * dca["weeks"])

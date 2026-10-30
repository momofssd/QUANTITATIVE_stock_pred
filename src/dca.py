"""Weekly dollar-cost averaging: which weekday has bought most cheaply.

Every strategy invests the same amount once a week over the same calendar weeks, buying at
the close of its weekday. When that weekday is not a trading session (a market holiday), the
buy moves to the next session, which can fall in the following week.

In a rising market the earliest weekday wins a calendar-week race simply by buying a few days
sooner, so a second, trend-neutral measure compares each buy with the sessions around it.
The most recent year is also simulated on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import clean_float

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
WEEKLY_AMOUNT = 100.0
MIN_WEEKS = 52
MIN_YEAR_WEEKS = 26
RECENT_WEEKS = 52
NEARBY_SESSIONS = 2  # sessions either side of a buy in the trend-neutral comparison
RELIABLE_T = 2.0


def weekday_buy_sessions(sessions: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """For each calendar week, the session each weekday strategy buys in.

    Returns the weeks (their Monday), positions into `sessions` shaped (weeks, 5), and whether
    each buy was moved off a holiday. Only weeks where every weekday's buy has happened are kept.
    """
    days = pd.DatetimeIndex(sessions).normalize()
    first = days[0] + pd.Timedelta(days=(7 - days[0].weekday()) % 7)
    mondays = pd.date_range(first, days[-1], freq="W-MON")
    targets = np.stack([(mondays + pd.Timedelta(days=offset)).to_numpy() for offset in range(5)], axis=1)
    positions = np.searchsorted(days.to_numpy(), targets, side="left")
    complete = (positions < len(days)).all(axis=1)
    positions, targets = positions[complete], targets[complete]
    moved = days.to_numpy()[positions] != targets
    return mondays[complete], positions, moved


def _average_cost(prices: np.ndarray, amount: float) -> np.ndarray:
    """Average cost per share of buying `amount` at each price, per column."""
    return len(prices) * amount / (amount / prices).sum(axis=0)


def _vs_nearby(close: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Each buy's price vs the average close of the sessions centred on it (trend-neutral)."""
    window = np.arange(-NEARBY_SESSIONS, NEARBY_SESSIONS + 1)
    rows = (positions.min(axis=1) >= NEARBY_SESSIONS) & (positions.max(axis=1) < len(close) - NEARBY_SESSIONS)
    kept = positions[rows]
    return close[kept] / close[kept[..., None] + window].mean(axis=-1) - 1


def _t_stat(values: np.ndarray) -> np.ndarray:
    """t-statistic for a mean below zero (positive = cheaper than nearby sessions)."""
    se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    return np.where(se > 0, -values.mean(axis=0) / np.where(se > 0, se, 1), 0.0)


def _simulate(close: np.ndarray, positions: np.ndarray, moved: np.ndarray, amount: float) -> dict:
    """Per-weekday results of buying `amount` in each of the given weeks, valued at the latest close."""
    prices = close[positions]
    costs = _average_cost(prices, amount)
    shares = (amount / prices).sum(axis=0)
    values = shares * close[-1]
    invested = amount * len(positions)
    nearby = _vs_nearby(close, positions)
    nearby_mean, nearby_t = nearby.mean(axis=0), _t_stat(nearby)
    best = int(np.argmin(costs))
    rows = [
        {
            "weekday": WEEKDAYS[day],
            "buys": len(positions),
            "holiday_moves": int(moved[:, day].sum()),
            "average_cost": clean_float(costs[day], 4),
            "vs_average": clean_float(costs[day] / costs.mean() - 1, 5),
            "shares": clean_float(shares[day], 4),
            "final_value": clean_float(values[day], 2),
            "total_return": clean_float(values[day] / invested - 1, 4),
            "vs_nearby": clean_float(nearby_mean[day], 5),
            "vs_nearby_t": clean_float(nearby_t[day], 2),
            "best": day == best,
        }
        for day in range(5)
    ]
    return {
        "prices": prices,
        "costs": costs,
        "best": best,
        "neutral_best": int(np.argmin(nearby_mean)),
        "nearby_mean": nearby_mean,
        "nearby_t": nearby_t,
        "gap": float(values[best] - values.mean()),
        "invested": invested,
        "rows": rows,
    }


def _recent_study(
    close_series: pd.Series, weeks: pd.DatetimeIndex, positions: np.ndarray, moved: np.ndarray, amount: float, full_best: int,
) -> dict:
    """The same simulation over only the most recent RECENT_WEEKS weeks."""
    block = slice(-RECENT_WEEKS, None)
    result = _simulate(close_series.to_numpy(), positions[block], moved[block], amount)
    best, count = result["best"], len(positions[block])
    summary = (
        f"Over the last {count} weeks, buying every {WEEKDAYS[best]} had the lowest average cost: "
        f"{abs(result['rows'][best]['vs_average']) * 100:.2f}% below the average weekday, "
        f"worth ${result['gap']:,.0f} more on ${amount:,.0f} a week."
    )
    summary += (
        " That matches the full history."
        if best == full_best
        else f" Over the full history {WEEKDAYS[full_best]} was cheapest."
    )
    summary += f" One year is only {count} buys per weekday, so treat this as the recent pattern, not a dependable edge."
    return {
        "weeks": count,
        "start": str(weeks[block][0].date()),
        "end": str(close_series.index[positions[-1].max()].date()),
        "invested": clean_float(result["invested"], 2),
        "best_weekday": WEEKDAYS[best],
        "worst_weekday": WEEKDAYS[int(np.argmax(result["costs"]))],
        "best_value_gain": clean_float(result["gap"], 2),
        "matches_full_history": best == full_best,
        "summary": summary,
        "weekdays": result["rows"],
    }


def weekly_dca_study(ohlc: pd.DataFrame, amount: float = WEEKLY_AMOUNT) -> dict | None:
    """Simulate buying `amount` every week on each weekday and compare the average cost.

    Covers the full history, plus the most recent year on its own under `recent`.
    """
    close_series = ohlc["close"].astype(float)
    if len(close_series) < 10:
        return None
    weeks, positions, moved = weekday_buy_sessions(close_series.index)
    if len(weeks) < MIN_WEEKS:
        return None
    close = close_series.to_numpy()
    full = _simulate(close, positions, moved, amount)
    prices, rows = full["prices"], full["rows"]
    best, neutral_best = full["best"], full["neutral_best"]
    nearby_mean, nearby_t = full["nearby_mean"], full["nearby_t"]

    years = pd.DatetimeIndex(weeks).year.to_numpy()
    year_rows = []
    for year in np.unique(years):
        block = years == year
        if block.sum() < MIN_YEAR_WEEKS:
            continue
        winner = int(np.argmin(_average_cost(prices[block], amount)))
        year_rows.append({"year": int(year), "weeks": int(block.sum()), "cheapest": WEEKDAYS[winner]})
    for row in rows:
        row["years_cheapest"] = sum(year["cheapest"] == row["weekday"] for year in year_rows)

    half = len(weeks) // 2
    halves = []
    for label, block in (("first", slice(0, half)), ("second", slice(half, None))):
        block_nearby = _vs_nearby(close, positions[block]).mean(axis=0)
        halves.append({
            "half": label,
            "start": str(weeks[block][0].date()),
            "end": str(weeks[block][-1].date()),
            "cheapest": WEEKDAYS[int(np.argmin(_average_cost(prices[block], amount)))],
            "cheapest_vs_nearby": WEEKDAYS[int(np.argmin(block_nearby))],
            "vs_nearby": [clean_float(value, 5) for value in block_nearby],
        })

    consistent = all(row["cheapest_vs_nearby"] == WEEKDAYS[neutral_best] for row in halves)
    reliable = bool(nearby_t[neutral_best] >= RELIABLE_T and consistent)
    gap = full["gap"]

    summary = (
        f"Buying every {WEEKDAYS[best]} had the lowest average cost: {abs(rows[best]['vs_average']) * 100:.2f}% below the "
        f"average weekday, worth ${gap:,.0f} more on ${amount:,.0f} a week over {len(weeks)} weeks."
    )
    if reliable:
        summary += (
            f" With the trend removed, {WEEKDAYS[neutral_best]} tended to close {abs(nearby_mean[neutral_best]) * 1e4:.0f} bp below "
            f"the sessions around it, in both halves of the history (t = {nearby_t[neutral_best]:.1f}): a real weekday edge, "
            "though small next to the gains from buying every week."
        )
    else:
        summary += (
            " Much of that comes from buying earlier in the calendar week while prices trend up, not from the weekday itself: "
            "with the trend removed, no weekday was reliably cheaper than the sessions around it, so buying every week "
            "matters far more than which day."
        )
    return {
        "amount": amount,
        "price_basis": "close, adjusted for splits and dividends",
        "holiday_rule": "If the weekday is a market holiday, the buy moves to the next trading session.",
        "nearby_definition": (
            f"Each buy's close vs the average close of the {2 * NEARBY_SESSIONS + 1} sessions centred on it; "
            "negative means that weekday tends to close below its neighbours, which removes the trend."
        ),
        "start": str(weeks[0].date()),
        "end": str(close_series.index[positions[-1].max()].date()),
        "weeks": len(weeks),
        "years": len(year_rows),
        "invested": clean_float(full["invested"], 2),
        "best_weekday": WEEKDAYS[best],
        "worst_weekday": WEEKDAYS[int(np.argmax(full["costs"]))],
        "best_value_gain": clean_float(gap, 2),
        "trend_neutral_best": WEEKDAYS[neutral_best],
        "trend_neutral_t": clean_float(nearby_t[neutral_best], 2),
        "consistent_across_halves": consistent,
        "reliable": reliable,
        "summary": summary,
        "weekdays": rows,
        "halves": halves,
        "by_year": year_rows,
        "recent": _recent_study(close_series, weeks, positions, moved, amount, best),
    }

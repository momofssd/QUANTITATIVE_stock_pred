"""Price-zone analysis: where to buy and sell short-term.

Zones are not forecasts. Each zone is the price range that would place the
ticker in an extreme percentile of its own historical deviation from a trend
line, and each zone ships with a replay of what price did after earlier visits.

Short-term deviation: (close - prior EMA20) / prior ATR14, so a zone is a
stretch measured in the ticker's own volatility units.

Momentum: (EMA5 - EMA13) / ATR14, the speed of the recent move in ATR units.
Negative is bearish, positive bullish. Whether it is building or fading is read
from the EMA spread itself, so a quiet session (shrinking ATR) cannot make a
stalled decline look like it is accelerating. A zone says where price is stretched; the
momentum turn says when the move into it has lost force.

Every level is causal: the band that applies to a session is computed from
data through the previous close, so the historical replay never sees the
session it is grading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import clean_float


SHORT_TERM_LOOKBACK = 756
MIN_HISTORY = 252
SHORT_TERM_BUY_PCTL = (0.05, 0.20)
SHORT_TERM_SELL_PCTL = (0.80, 0.95)
SHORT_TERM_HORIZON = 10
MIN_EPISODES = 8
BOUNCE_ATR = 1.5
RECENT_TOUCH_SESSIONS = 3
SWING_WINDOW = 5
BREAKDOWN_ACTIVE_SESSIONS = 5
MOMENTUM_FAST, MOMENTUM_SLOW = 5, 13
# A bottom (top) signal: within SIGNAL_TOUCH_SESSIONS of a buy (take-profit) zone touch, momentum is still
# bearish (bullish) but has moved against the trend for MOMENTUM_TURN_SESSIONS sessions in a row.
MOMENTUM_TURN_SESSIONS = 2
SIGNAL_TOUCH_SESSIONS = 5
SIGNAL_ACTIVE_SESSIONS = 3
MOMENTUM_EXTREME_SESSIONS = 10
# The signal's stop sits STOP_BUFFER_ATR beyond the interim low (high) of the last INTERIM_SESSIONS,
# at least MIN_RISK_ATR from the entry; the target is TARGET_R times that risk.
INTERIM_SESSIONS = 6
STOP_BUFFER_ATR = 0.25
MIN_RISK_ATR = 0.5
TARGET_R = 2.0

ZONE_KEYS = ("st_buy", "st_sell")


def adjusted_ohlcv(features: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV on the adjusted-close basis so ranges and closes are comparable."""
    close = features["close"].astype(float)
    factor = (close / features["Close"].replace(0, np.nan)).fillna(1.0)
    frame = pd.DataFrame(
        {
            "open": (features["Open"] * factor).fillna(close),
            "high": (features["High"] * factor).fillna(close),
            "low": (features["Low"] * factor).fillna(close),
            "close": close,
            "volume": features["Volume"].astype(float),
        },
        index=features.index,
    )
    frame["high"] = frame[["high", "open", "close"]].max(axis=1)
    frame["low"] = frame[["low", "open", "close"]].min(axis=1)
    return frame


def compute_zone_frame(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Anchors, deviations, and zone levels for every session.

    `<zone>_low/high` columns are the band that applied *during* that session.
    `next_<zone>_low/high` columns are the band known after that session's close,
    so the last row holds the zones for the next trading session.
    """
    close, high, low = ohlc["close"], ohlc["high"], ohlc["low"]
    prev_close = close.shift()
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    frame = pd.DataFrame(index=ohlc.index)
    frame["atr"] = true_range.rolling(14, min_periods=14).mean()
    frame["ema20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    frame["ma50"] = close.rolling(50, min_periods=50).mean()
    frame["ma200"] = close.rolling(200, min_periods=200).mean()
    frame["st_dev"] = (close - frame["ema20"].shift()) / frame["atr"].shift()
    frame["ema_fast"] = close.ewm(span=MOMENTUM_FAST, adjust=False, min_periods=MOMENTUM_FAST).mean()
    frame["ema_slow"] = close.ewm(span=MOMENTUM_SLOW, adjust=False, min_periods=MOMENTUM_SLOW).mean()
    spread = frame["ema_fast"] - frame["ema_slow"]
    frame["momentum"] = spread / frame["atr"]
    # Daily change of the spread, expressed in today's ATRs for display; its sign is price-only.
    frame["momentum_change"] = spread.diff() / frame["atr"]

    for key, pctls in (("st_buy", SHORT_TERM_BUY_PCTL), ("st_sell", SHORT_TERM_SELL_PCTL)):
        for edge, level in zip(("low", "high"), pctls):
            q = frame["st_dev"].rolling(SHORT_TERM_LOOKBACK, min_periods=MIN_HISTORY).quantile(level)
            frame[f"st_q{int(level * 100):02d}"] = q
            frame[f"next_{key}_{edge}"] = frame["ema20"] + frame["atr"] * q
    for key in ZONE_KEYS:
        for edge in ("low", "high"):
            frame[f"{key}_{edge}"] = frame[f"next_{key}_{edge}"].shift()
    # Thresholds that applied to each session, for the stretch study pane.
    for column in [name for name in frame.columns if name.startswith("st_q")]:
        frame[f"{column}_applied"] = frame[column].shift()
    return frame


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Simple-average RSI that reads 100 (not NaN) when a window has no losses."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    value = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return value.mask((loss == 0) & (gain > 0), 100.0).mask((loss == 0) & (gain == 0), 50.0)


def percentile_rank(series: pd.Series, lookback: int) -> float | None:
    window = series.dropna().tail(lookback)
    if len(window) < MIN_HISTORY:
        return None
    return float((window <= window.iloc[-1]).mean())


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _rate(flags: list[bool]) -> float | None:
    return float(np.mean(flags)) if flags else None


def _summarize_context(episodes: list[dict], context_up: bool, success_key: str = "success") -> dict:
    graded = [episode for episode in episodes if episode["complete"]]
    def subset(flag: bool) -> dict:
        rows = [episode for episode in graded if episode["trend_up"] == flag]
        return {"episodes": len(rows), "success_rate": clean_float(_rate([row[success_key] for row in rows]))}
    up, down = subset(True), subset(False)
    current = up if context_up else down
    return {
        "uptrend": up,
        "downtrend": down,
        "current_context": "uptrend" if context_up else "downtrend",
        "current_context_episodes": current["episodes"],
        "current_context_success_rate": current["success_rate"],
    }


def _bracket(
    highs: np.ndarray,
    lows: np.ndarray,
    i: int,
    horizon: int,
    target: float,
    stop: float,
    side: str,
    include_entry_day: bool,
) -> tuple[str, int | None, np.ndarray, np.ndarray]:
    """Grade a target/stop bracket over the next `horizon` sessions.

    Targets count from the session after entry. The entry session's own range
    counts toward the stop only when the fill happened intraday. A session that
    hits both levels is graded as a stop, which is the conservative reading.
    """
    end = min(i + horizon, len(highs) - 1)
    first = i if include_entry_day else i + 1
    favorable = highs[i + 1:end + 1] if side == "buy" else lows[i + 1:end + 1]
    adverse = lows[first:end + 1] if side == "buy" else highs[first:end + 1]
    offset = 0 if include_entry_day else 1
    if side == "buy":
        target_hits, stop_hits = np.flatnonzero(favorable >= target) + 1, np.flatnonzero(adverse <= stop) + offset
    else:
        target_hits, stop_hits = np.flatnonzero(favorable <= target) + 1, np.flatnonzero(adverse >= stop) + offset
    target_day = int(target_hits[0]) if target_hits.size else None
    stop_day = int(stop_hits[0]) if stop_hits.size else None
    if target_day is not None and (stop_day is None or target_day < stop_day):
        outcome = "success"
    elif stop_day is not None:
        outcome = "failed"
    else:
        outcome = "expired" if i + horizon <= len(highs) - 1 else "pending"
    return outcome, target_day, favorable, adverse


def reversion_study(
    ohlc: pd.DataFrame,
    zones: pd.DataFrame,
    side: str,
    horizon: int = SHORT_TERM_HORIZON,
) -> dict:
    """Replay every earlier visit to a short-term zone with a symmetric ATR bracket.

    Buy side: the first touch of a visit fills at the top of the zone (or the
    open if price gaps through it). Success = a move of BOUNCE_ATR above the
    fill before a move of BOUNCE_ATR below it, within `horizon` sessions. The
    sell side mirrors this downward. The baseline applies the same bracket from
    every session's close, so the edge shows whether the zone beats entering on
    an arbitrary day.
    """
    prefix = "st_buy" if side == "buy" else "st_sell"
    band_low = zones[f"{prefix}_low"].to_numpy()
    band_high = zones[f"{prefix}_high"].to_numpy()
    atr_prior = zones["atr"].shift().to_numpy()
    atr_now = zones["atr"].to_numpy()
    trend_up = (ohlc["close"].shift() > zones["ma200"].shift()).to_numpy()
    opens, highs, lows, closes = (ohlc[column].to_numpy() for column in ("open", "high", "low", "close"))
    valid = np.isfinite(band_low) & np.isfinite(band_high) & np.isfinite(atr_prior)
    touched = valid & ((lows <= band_high) if side == "buy" else (highs >= band_low))
    starts = np.flatnonzero(touched & ~np.concatenate([[False], touched[:-1]]))
    sign = 1 if side == "buy" else -1
    episodes: list[dict] = []
    for i in starts:
        entry = min(opens[i], band_high[i]) if side == "buy" else max(opens[i], band_low[i])
        target = entry + sign * BOUNCE_ATR * atr_prior[i]
        stop = entry - sign * BOUNCE_ATR * atr_prior[i]
        outcome, target_day, favorable, adverse = _bracket(highs, lows, int(i), horizon, target, stop, side, True)
        best = (favorable.max() if side == "buy" else favorable.min()) if favorable.size else entry
        worst = adverse.min() if side == "buy" else adverse.max()
        episodes.append(
            {
                "index": int(i),
                "date": str(ohlc.index[i].date()),
                "entry": float(entry),
                "target": float(target),
                "stop": float(stop),
                "outcome": outcome,
                "success": outcome == "success",
                "complete": outcome != "pending",
                "trend_up": bool(trend_up[i]),
                "favorable_move": float(sign * (best / entry - 1)),
                "adverse_move": float(sign * (worst / entry - 1)),
                "days_to_target": target_day if outcome == "success" else None,
                "days_to_extreme": int(np.argmin(adverse) if side == "buy" else np.argmax(adverse)),
            }
        )
    baseline: list[bool] = []
    for i in np.flatnonzero(valid & np.isfinite(atr_now)):
        if i + horizon > len(closes) - 1:
            break
        entry = closes[i]
        outcome, *_ = _bracket(
            highs, lows, int(i), horizon,
            entry + sign * BOUNCE_ATR * atr_now[i], entry - sign * BOUNCE_ATR * atr_now[i], side, False,
        )
        baseline.append(outcome == "success")
    graded = [episode for episode in episodes if episode["complete"]]
    success_rate = _rate([episode["success"] for episode in graded])
    baseline_rate = _rate(baseline)
    current_trend_up = bool(ohlc["close"].iloc[-1] > zones["ma200"].iloc[-1])
    move, against = ("rises", "falling") if side == "buy" else ("falls", "rising")
    return {
        "side": side,
        "horizon_sessions": horizon,
        "bracket_atr": BOUNCE_ATR,
        "episodes": len(graded),
        "sufficient_sample": len(graded) >= MIN_EPISODES,
        "success_rate": clean_float(success_rate),
        "failure_rate": clean_float(_rate([episode["outcome"] == "failed" for episode in graded])),
        "expired_rate": clean_float(_rate([episode["outcome"] == "expired" for episode in graded])),
        "baseline_success_rate": clean_float(baseline_rate),
        "edge_vs_baseline": (
            clean_float(success_rate - baseline_rate) if success_rate is not None and baseline_rate is not None else None
        ),
        "median_favorable_move": clean_float(_median([episode["favorable_move"] for episode in graded])),
        "median_adverse_move": clean_float(_median([episode["adverse_move"] for episode in graded])),
        "median_days_to_target": clean_float(
            _median([episode["days_to_target"] for episode in graded if episode["days_to_target"] is not None]), 1
        ),
        "median_days_to_extreme": clean_float(_median([episode["days_to_extreme"] for episode in graded]), 1),
        "by_trend": _summarize_context(episodes, current_trend_up),
        "definition": (
            f"First touch of the zone fills at the zone edge. Success = price {move} {BOUNCE_ATR:g} ATR from the fill "
            f"before {against} {BOUNCE_ATR:g} ATR, within {horizon} sessions. "
            "Baseline = the same bracket from every session's close."
        ),
        "recent_episodes": episodes,
    }


def _run_length(flags: pd.Series) -> pd.Series:
    """Consecutive True values ending at each session."""
    return flags.astype(int).groupby((~flags).cumsum()).cumsum()


def momentum_frame(ohlc: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    """Per-session momentum-exhaustion state, all known at that session's close.

    buy_signal: the session on which, within SIGNAL_TOUCH_SESSIONS of a buy-zone touch, momentum is
    still negative but has risen for exactly MOMENTUM_TURN_SESSIONS sessions: the selling is losing force.
    sell_signal mirrors it at the take-profit zone. buy/sell stop and target are the levels an entry at
    that session's close would use: beyond the interim low (high) of the last INTERIM_SESSIONS, and
    TARGET_R times that risk.
    """
    momentum = zones["momentum"]
    change = zones["momentum_change"]
    rising, falling = _run_length(change > 0), _run_length(change < 0)
    touched_buy = (ohlc["low"] <= zones["st_buy_high"]).astype(float).rolling(SIGNAL_TOUCH_SESSIONS, min_periods=1).max()
    touched_sell = (ohlc["high"] >= zones["st_sell_low"]).astype(float).rolling(SIGNAL_TOUCH_SESSIONS, min_periods=1).max()
    close, atr = ohlc["close"], zones["atr"]
    interim_low = ohlc["low"].rolling(INTERIM_SESSIONS, min_periods=1).min()
    interim_high = ohlc["high"].rolling(INTERIM_SESSIONS, min_periods=1).max()
    buy_stop = np.minimum(interim_low - STOP_BUFFER_ATR * atr, close - MIN_RISK_ATR * atr)
    sell_stop = np.maximum(interim_high + STOP_BUFFER_ATR * atr, close + MIN_RISK_ATR * atr)
    return pd.DataFrame(
        {
            "momentum": momentum,
            "change": change,
            "rising_sessions": rising,
            "falling_sessions": falling,
            "touched_buy": touched_buy > 0,
            "touched_sell": touched_sell > 0,
            "buy_signal": (touched_buy > 0) & (momentum < 0) & (rising == MOMENTUM_TURN_SESSIONS),
            "sell_signal": (touched_sell > 0) & (momentum > 0) & (falling == MOMENTUM_TURN_SESSIONS),
            "interim_low": interim_low,
            "interim_high": interim_high,
            "buy_stop": buy_stop,
            "buy_target": close + TARGET_R * (close - buy_stop),
            "sell_stop": sell_stop,
            "sell_target": close - TARGET_R * (sell_stop - close),
        },
        index=ohlc.index,
    )


def momentum_pivot(zones: pd.DataFrame) -> float | None:
    """Next-session close at which the EMA spread, and so the momentum direction, is unchanged.

    Both EMAs move linearly with the close, so a close above this price makes momentum
    rise and a close below makes it fall.
    """
    last = zones.iloc[-1]
    fast, slow = 2 / (MOMENTUM_FAST + 1), 2 / (MOMENTUM_SLOW + 1)
    value = (fast * last["ema_fast"] - slow * last["ema_slow"]) / (fast - slow)
    return float(value) if np.isfinite(value) else None


def momentum_state(ohlc: pd.DataFrame, zones: pd.DataFrame) -> dict:
    """Where momentum stands after the latest close, and the most recent bottom or top signal."""
    frame = momentum_frame(ohlc, zones)
    last = frame.iloc[-1]
    value, change = float(last["momentum"]), float(last["change"])
    if value < 0:
        phase = "BEARISH_FADING" if change > 0 else "BEARISH_BUILDING"
    else:
        phase = "BULLISH_FADING" if change < 0 else "BULLISH_BUILDING"
    # Share of the recent extreme (same sign as today) that has already worn off, measured on the EMA
    # spread like the turn itself, then shown in today's ATRs so it sits on the same scale as the value.
    spread = (zones["ema_fast"] - zones["ema_slow"]).tail(MOMENTUM_EXTREME_SESSIONS)
    extreme_spread = float(spread.min() if value < 0 else spread.max())
    today_spread = float(spread.iloc[-1])
    same_side = extreme_spread != 0 and np.sign(extreme_spread) == np.sign(today_spread)
    faded = min(max(1 - today_spread / extreme_spread, 0.0), 1.0) if same_side else 0.0
    extreme = extreme_spread / float(zones["atr"].iloc[-1])
    close = float(ohlc["close"].iloc[-1])
    signal, signal_index = None, -1
    for side in ("buy", "sell"):
        hits = np.flatnonzero(frame[f"{side}_signal"].to_numpy()[-SIGNAL_ACTIVE_SESSIONS:])
        if not hits.size:
            continue
        i = len(frame) - SIGNAL_ACTIVE_SESSIONS + int(hits[-1])
        if i <= signal_index:
            continue
        stop, target = float(frame[f"{side}_stop"].iloc[i]), float(frame[f"{side}_target"].iloc[i])
        favorable = (target - close) if side == "buy" else (close - target)
        adverse = (close - stop) if side == "buy" else (stop - close)
        reward_risk = favorable / adverse if adverse > 0 and favorable > 0 else None
        signal_index = i
        signal = {
            "side": side,
            "date": str(ohlc.index[i].date()),
            "sessions_ago": len(frame) - 1 - i,
            "entry": clean_float(ohlc["close"].iloc[i], 2),
            "stop": clean_float(stop, 2),
            "target": clean_float(target, 2),
            "reward_risk_now": clean_float(reward_risk, 2),
            # Still actionable while at least 1:1 remains from today's close.
            "active": reward_risk is not None and reward_risk >= 1.0,
        }
    return {
        "value_atr": clean_float(value, 2),
        "change_atr": clean_float(change, 3),
        "percentile": clean_float(percentile_rank(frame["momentum"], SHORT_TERM_LOOKBACK)),
        "phase": phase,
        "rising_sessions": int(last["rising_sessions"]),
        "falling_sessions": int(last["falling_sessions"]),
        "recent_extreme_atr": clean_float(extreme, 2),
        "faded_from_extreme": clean_float(faded),
        "pivot_price": clean_float(momentum_pivot(zones), 2),
        "interim_low": clean_float(last["interim_low"], 2),
        "interim_high": clean_float(last["interim_high"], 2),
        "touched_buy_zone": bool(last["touched_buy"]),
        "touched_sell_zone": bool(last["touched_sell"]),
        "touch_window_sessions": SIGNAL_TOUCH_SESSIONS,
        "turn_sessions_required": MOMENTUM_TURN_SESSIONS,
        "signal": signal,
        "definition": (
            f"Momentum = (EMA{MOMENTUM_FAST} - EMA{MOMENTUM_SLOW}) / ATR14: how fast price has been moving, in ATRs. "
            "Below 0 is bearish, above 0 bullish. Fading = moving back toward 0. "
            f"A bottom signal needs a buy-zone touch within {SIGNAL_TOUCH_SESSIONS} sessions and bearish momentum that "
            f"has faded for {MOMENTUM_TURN_SESSIONS} sessions in a row; a top signal mirrors it at the take-profit zone."
        ),
    }


def _trade_r(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
    i: int, horizon: int, stop: float, target: float, side: str,
) -> tuple[str, float | None, int | None]:
    """Grade an entry at session i's close in R (multiples of the planned risk). A gap through the stop exits at the open."""
    outcome, target_day, *_ = _bracket(highs, lows, i, horizon, target, stop, side, False)
    entry, sign = closes[i], 1 if side == "buy" else -1
    risk = abs(entry - stop)
    end = min(i + horizon, len(closes) - 1)
    if outcome == "success":
        return outcome, abs(target - entry) / risk, target_day
    if outcome == "failed":
        hit = lows[i + 1:end + 1] <= stop if side == "buy" else highs[i + 1:end + 1] >= stop
        day = i + 1 + int(np.flatnonzero(hit)[0])
        fill = min(opens[day], stop) if side == "buy" else max(opens[day], stop)
        return outcome, sign * (fill - entry) / risk, None
    if outcome == "expired":
        return outcome, sign * (closes[end] - entry) / risk, None
    return outcome, None, None


def momentum_signal_study(ohlc: pd.DataFrame, zones: pd.DataFrame, side: str, horizon: int = SHORT_TERM_HORIZON) -> dict:
    """Replay every earlier bottom (buy) or top (sell) signal as the trade the dashboard proposes.

    Entry is the signal session's close, the stop sits beyond the interim low (high), and the
    target is TARGET_R times the risk; each trade ends at the target, the stop, or after `horizon`
    sessions. The baseline makes the same trade from every session's close, so the edge shows
    whether waiting for momentum to fade beat entering on an arbitrary day. "Held" = the interim
    low (high) was not broken within `horizon` sessions, i.e. the signal caught a real turn.
    """
    frame = momentum_frame(ohlc, zones)
    opens, highs, lows, closes = (ohlc[column].to_numpy() for column in ("open", "high", "low", "close"))
    stops, targets = frame[f"{side}_stop"].to_numpy(), frame[f"{side}_target"].to_numpy()
    interim = frame["interim_low" if side == "buy" else "interim_high"].to_numpy()
    atr = zones["atr"].to_numpy()
    trend_up = (ohlc["close"] > zones["ma200"]).to_numpy()
    band = zones["st_buy_high" if side == "buy" else "st_sell_low"].to_numpy()
    n = len(closes)

    def held(i: int) -> bool:
        window = slice(i + 1, i + 1 + horizon)
        return bool(lows[window].min() > interim[i]) if side == "buy" else bool(highs[window].max() < interim[i])

    episodes: list[dict] = []
    last = -SIGNAL_TOUCH_SESSIONS - 1
    for i in np.flatnonzero(frame[f"{side}_signal"].to_numpy()):
        if i - last <= SIGNAL_TOUCH_SESSIONS:
            continue
        last = i
        outcome, r, target_day = _trade_r(opens, highs, lows, closes, int(i), horizon, stops[i], targets[i], side)
        complete = outcome != "pending"
        # How much further price went against the entry before the window closed; graded trades only.
        window = slice(i + 1, i + 1 + horizon)
        further = None
        if complete:
            further = (closes[i] - lows[window].min()) if side == "buy" else (highs[window].max() - closes[i])
        episodes.append(
            {
                "index": int(i),
                "date": str(ohlc.index[i].date()),
                "entry": float(closes[i]),
                "stop": float(stops[i]),
                "target": float(targets[i]),
                "interim_extreme": float(interim[i]),
                "momentum": float(frame["momentum"].iloc[i]),
                "outcome": outcome,
                "success": outcome == "success",
                "complete": complete,
                "trend_up": bool(trend_up[i]),
                "r_multiple": r,
                "held": held(int(i)) if complete else None,
                "further_move_atr": float(further / atr[i]) if further is not None else None,
                "days_to_target": target_day,
            }
        )
    baseline_r, baseline_held = [], []
    for i in np.flatnonzero(np.isfinite(band) & np.isfinite(stops) & np.isfinite(atr)):
        if i + horizon > n - 1:
            break
        baseline_r.append(_trade_r(opens, highs, lows, closes, int(i), horizon, stops[i], targets[i], side)[1])
        baseline_held.append(held(int(i)))
    graded = [episode for episode in episodes if episode["complete"]]
    results = [episode["r_multiple"] for episode in graded]
    expectancy = float(np.mean(results)) if results else None
    baseline_expectancy = float(np.mean(baseline_r)) if baseline_r else None
    current_trend_up = bool(ohlc["close"].iloc[-1] > zones["ma200"].iloc[-1])
    extreme = "low" if side == "buy" else "high"
    return {
        "side": side,
        "horizon_sessions": horizon,
        "target_r": TARGET_R,
        "episodes": len(graded),
        "sufficient_sample": len(graded) >= MIN_EPISODES,
        "success_rate": clean_float(_rate([episode["success"] for episode in graded])),
        "stop_rate": clean_float(_rate([episode["outcome"] == "failed" for episode in graded])),
        "expired_rate": clean_float(_rate([episode["outcome"] == "expired" for episode in graded])),
        "expectancy_r": clean_float(expectancy, 2),
        "median_r": clean_float(_median(results), 2),
        "baseline_expectancy_r": clean_float(baseline_expectancy, 2),
        "edge_r": (
            clean_float(expectancy - baseline_expectancy, 2) if expectancy is not None and baseline_expectancy is not None else None
        ),
        "held_rate": clean_float(_rate([episode["held"] for episode in graded])),
        "baseline_held_rate": clean_float(_rate(baseline_held)),
        "median_further_move_atr": clean_float(_median([episode["further_move_atr"] for episode in graded]), 2),
        "median_days_to_target": clean_float(
            _median([episode["days_to_target"] for episode in graded if episode["days_to_target"] is not None]), 1
        ),
        "by_trend": _summarize_context(episodes, current_trend_up),
        "definition": (
            f"Signal = within {SIGNAL_TOUCH_SESSIONS} sessions of a {'buy' if side == 'buy' else 'take-profit'}-zone touch, "
            f"momentum {'rose' if side == 'buy' else 'fell'} {MOMENTUM_TURN_SESSIONS} sessions in a row while still "
            f"{'bearish' if side == 'buy' else 'bullish'}. Entry at that close; stop {STOP_BUFFER_ATR:g} ATR beyond the "
            f"{INTERIM_SESSIONS}-session {extreme}; target {TARGET_R:g}x the risk; otherwise exit after {horizon} sessions. "
            f"Results are in R (multiples of the risk). Held = the interim {extreme} was not broken within {horizon} sessions. "
            "Baseline = the same trade from every session's close."
        ),
        "recent_episodes": episodes,
    }


def swing_support(ohlc: pd.DataFrame, window: int = SWING_WINDOW) -> pd.Series:
    """Most recent confirmed swing low known after each close.

    A swing low is the lowest low of a centered (2*window+1)-session span, so it
    is only confirmed `window` sessions later. Shift by one for the level that
    applied during a session.
    """
    low = ohlc["low"]
    pivot = low == low.rolling(2 * window + 1, center=True).min()
    return low.where(pivot).shift(window).ffill()


def _breakdowns(ohlc: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sessions whose close was the first close below the swing-low support that applied to them."""
    support = swing_support(ohlc).shift().to_numpy()
    closes = ohlc["close"].to_numpy()
    prev_close = np.concatenate([[np.nan], closes[:-1]])
    with np.errstate(invalid="ignore"):
        breaks = np.isfinite(support) & (closes < support) & (prev_close >= support)
    return breaks, support


def breakdown_study(ohlc: pd.DataFrame, zones: pd.DataFrame, baseline_rate: float | None, horizon: int = SHORT_TERM_HORIZON) -> dict:
    """Replay every close below swing-low support as a continuation trade.

    Entry is the breakdown close. Success = price falls BOUNCE_ATR below it before
    rising BOUNCE_ATR above it within `horizon` sessions. The baseline is the
    sell-side bracket from every session's close, so the edge shows whether a
    breakdown kept falling more often than an arbitrary day did.
    """
    highs, lows, closes = (ohlc[column].to_numpy() for column in ("high", "low", "close"))
    atr = zones["atr"].to_numpy()
    trend_up = (ohlc["close"].shift() > zones["ma200"].shift()).to_numpy()
    breaks, support = _breakdowns(ohlc)
    episodes: list[dict] = []
    for i in np.flatnonzero(breaks & np.isfinite(atr)):
        entry = closes[i]
        target, stop = entry - BOUNCE_ATR * atr[i], entry + BOUNCE_ATR * atr[i]
        outcome, target_day, favorable, adverse = _bracket(highs, lows, int(i), horizon, target, stop, "sell", False)
        end = min(i + horizon, len(closes) - 1)
        episodes.append(
            {
                "index": int(i),
                "date": str(ohlc.index[i].date()),
                "entry": float(entry),
                "support": float(support[i]),
                "outcome": outcome,
                "success": outcome == "success",
                "complete": outcome != "pending",
                "trend_up": bool(trend_up[i]),
                "favorable_move": float(1 - favorable.min() / entry) if favorable.size else 0.0,
                "adverse_move": float(1 - adverse.max() / entry) if adverse.size else 0.0,
                "reclaimed": bool((closes[i + 1:end + 1] >= support[i]).any()),
                "days_to_target": target_day if outcome == "success" else None,
            }
        )
    graded = [episode for episode in episodes if episode["complete"]]
    success_rate = _rate([episode["success"] for episode in graded])
    current_trend_up = bool(ohlc["close"].iloc[-1] > zones["ma200"].iloc[-1])
    return {
        "side": "breakdown",
        "horizon_sessions": horizon,
        "bracket_atr": BOUNCE_ATR,
        "episodes": len(graded),
        "sufficient_sample": len(graded) >= MIN_EPISODES,
        "success_rate": clean_float(success_rate),
        "failure_rate": clean_float(_rate([episode["outcome"] == "failed" for episode in graded])),
        "expired_rate": clean_float(_rate([episode["outcome"] == "expired" for episode in graded])),
        "reclaim_rate": clean_float(_rate([episode["reclaimed"] for episode in graded])),
        "baseline_success_rate": clean_float(baseline_rate),
        "edge_vs_baseline": (
            clean_float(success_rate - baseline_rate) if success_rate is not None and baseline_rate is not None else None
        ),
        "median_favorable_move": clean_float(_median([episode["favorable_move"] for episode in graded])),
        "median_adverse_move": clean_float(_median([episode["adverse_move"] for episode in graded])),
        "median_days_to_target": clean_float(
            _median([episode["days_to_target"] for episode in graded if episode["days_to_target"] is not None]), 1
        ),
        "by_trend": _summarize_context(episodes, current_trend_up),
        "definition": (
            f"First close below the most recent confirmed swing low (a {2 * SWING_WINDOW + 1}-session pivot), entered at that close. "
            f"Worked = price fell another {BOUNCE_ATR:g} ATR before rising {BOUNCE_ATR:g} ATR, within {horizon} sessions. "
            "Baseline = the same downside bracket from every session's close. "
            f"Reclaim rate = share that closed back above the broken level within {horizon} sessions."
        ),
        "recent_episodes": episodes,
    }


def breakdown_state(ohlc: pd.DataFrame, zones: pd.DataFrame, swings: list[dict]) -> dict:
    """Whether price recently closed below swing-low support and is still below it."""
    close = float(ohlc["close"].iloc[-1])
    atr = float(zones["atr"].iloc[-1])
    breaks, support = _breakdowns(ohlc)
    start = max(len(breaks) - BREAKDOWN_ACTIVE_SESSIONS, 0)
    recent = np.flatnonzero(breaks[start:])
    supports_below = [swing["price"] for swing in swings if swing["type"] == "swing_low" and swing["price"] < close]
    state = {
        "status": "NONE",
        "active": False,
        "broken_level": None,
        "breakdown_date": None,
        "sessions_since_breakdown": None,
        "close_vs_broken_level_pct": None,
        "follow_through_target": None,
        "next_support": clean_float(max(supports_below), 2) if supports_below else None,
        "definition": (
            f"Active when a close fell below the most recent confirmed swing low within the last {BREAKDOWN_ACTIVE_SESSIONS} "
            "sessions and price has not closed back above it."
        ),
    }
    if recent.size:
        i = start + int(recent[-1])
        level = float(support[i])
        active = close < level
        state.update(
            {
                "status": "ACTIVE" if active else "RECLAIMED",
                "active": active,
                "broken_level": clean_float(level, 2),
                "breakdown_date": str(ohlc.index[i].date()),
                "sessions_since_breakdown": len(breaks) - 1 - i,
                "close_vs_broken_level_pct": clean_float(close / level - 1),
                "follow_through_target": clean_float(close - BOUNCE_ATR * atr, 2) if active else None,
            }
        )
    return state


def swing_levels(ohlc: pd.DataFrame, window: int = SWING_WINDOW, lookback: int = 252) -> list[dict]:
    """Confirmed swing highs and lows: extremes of a centered (2*window+1)-session span."""
    recent = ohlc.tail(lookback)
    span = 2 * window + 1
    pivot_low = recent["low"] == recent["low"].rolling(span, center=True).min()
    pivot_high = recent["high"] == recent["high"].rolling(span, center=True).max()
    levels = [
        {"date": str(index.date()), "price": float(recent.at[index, "low"]), "type": "swing_low"}
        for index in recent.index[pivot_low.to_numpy()]
    ]
    levels += [
        {"date": str(index.date()), "price": float(recent.at[index, "high"]), "type": "swing_high"}
        for index in recent.index[pivot_high.to_numpy()]
    ]
    return sorted(levels, key=lambda level: level["date"], reverse=True)


def zone_confluence(low: float, high: float, atr: float, anchors: dict[str, float], swings: list[dict], kind: str) -> list[str]:
    """Plain-language list of known levels that sit inside or just around a zone."""
    pad = 0.5 * atr
    notes = [
        f"{label} at {value:.2f}"
        for label, value in anchors.items()
        if value is not None and np.isfinite(value) and low - pad <= value <= high + pad
    ]
    wanted = "swing_low" if kind == "buy" else "swing_high"
    matches = [swing for swing in swings if swing["type"] == wanted and low - pad <= swing["price"] <= high + pad]
    notes += [f"{'Swing low' if kind == 'buy' else 'Swing high'} {swing['price']:.2f} on {swing['date']}" for swing in matches[:3]]
    return notes


def bounce_readiness(ohlc: pd.DataFrame, features: pd.DataFrame, zones: pd.DataFrame, market_close: pd.Series | None) -> dict:
    """Deterministic checklist of the conditions that usually precede a V-shaped bounce."""
    latest = ohlc.iloc[-1]
    previous = ohlc.iloc[-2]
    close = float(latest["close"])
    day_range = max(float(latest["high"] - latest["low"]), 1e-9)
    lower_wick = (min(latest["open"], close) - latest["low"]) / day_range
    close_location = (close - latest["low"]) / day_range
    rsi_14_series = _rsi(ohlc["close"], 14)
    rsi_14 = float(rsi_14_series.iloc[-1]) if np.isfinite(rsi_14_series.iloc[-1]) else 50.0
    rsi_2_series = _rsi(ohlc["close"], 2)
    rsi_2 = float(rsi_2_series.iloc[-1]) if np.isfinite(rsi_2_series.iloc[-1]) else 50.0
    momentum = float(zones["momentum"].iloc[-1])
    momentum_change = float(zones["momentum_change"].iloc[-1])
    volume_ratio = (ohlc["volume"] / ohlc["volume"].rolling(20, min_periods=20).mean()).tail(RECENT_TOUCH_SESSIONS)
    peak_volume_ratio = float(volume_ratio.max()) if volume_ratio.notna().any() else 0.0
    stretch_pctl = percentile_rank(zones["st_dev"], SHORT_TERM_LOOKBACK)
    recent = ohlc.tail(RECENT_TOUCH_SESSIONS)
    recent_bands = zones["st_buy_high"].tail(RECENT_TOUCH_SESSIONS)
    touched_recently = bool((recent["low"] <= recent_bands).any())
    engulfing = close > previous["open"] and latest["open"] <= previous["close"] and previous["close"] < previous["open"]
    reversal_candle = bool((lower_wick >= 0.5 and close_location >= 0.5) or engulfing)
    reclaimed = bool(close > float(previous["high"]))

    market_ok = None
    if market_close is not None and market_close.notna().sum() >= 50:
        aligned = market_close.reindex(ohlc.index).ffill()
        market_ok = bool(aligned.iloc[-1] > aligned.rolling(50, min_periods=50).mean().iloc[-1])

    checks = [
        {
            "key": "stretched",
            "label": "Stretched below its 20-day mean",
            "passed": stretch_pctl is not None and stretch_pctl <= 0.20,
            "detail": "Stretch percentile unavailable" if stretch_pctl is None else f"Deviation is at the {stretch_pctl:.0%} percentile of the last 3 years",
        },
        {
            "key": "oversold",
            "label": "Momentum oversold",
            "passed": rsi_14 <= 35 or rsi_2 <= 10,
            "detail": f"RSI(14) {rsi_14:.1f}, RSI(2) {rsi_2:.1f}",
        },
        {
            "key": "momentum_fading",
            "label": "Bearish momentum fading",
            "passed": momentum < 0 and momentum_change > 0,
            "detail": (
                f"Momentum {momentum:+.2f} ATR, "
                + ("bullish" if momentum >= 0 else "rising" if momentum_change > 0 else "still falling")
                + f" ({momentum_change:+.2f} today)"
            ),
        },
        {
            "key": "capitulation_volume",
            "label": "Capitulation volume",
            "passed": peak_volume_ratio >= 1.5,
            "detail": f"Peak volume over the last {RECENT_TOUCH_SESSIONS} sessions is {peak_volume_ratio:.2f}x the 20-day average",
        },
        {
            "key": "reversal_candle",
            "label": "Reversal candle",
            "passed": reversal_candle,
            "detail": "Hammer or bullish engulfing on the last session" if reversal_candle else "No hammer or engulfing candle on the last session",
        },
        {
            "key": "reclaim",
            "label": "Reclaimed prior session high",
            "passed": reclaimed,
            "detail": f"Close {close:.2f} vs prior high {float(previous['high']):.2f}",
        },
        {
            "key": "market_backdrop",
            "label": "Market above its 50-day average",
            "passed": bool(market_ok),
            "detail": "Benchmark data unavailable" if market_ok is None else ("Broad market supportive" if market_ok else "Broad market below its 50-day average"),
        },
    ]
    passed = {check["key"] for check in checks if check["passed"]}
    selling_signals = len(passed & {"stretched", "oversold", "capitulation_volume"})
    if touched_recently and "reclaim" in passed:
        stage = "CONFIRMED"
    elif touched_recently and "reversal_candle" in passed:
        stage = "FORMING"
    elif selling_signals >= 2:
        stage = "WASHOUT_NO_REVERSAL_YET"
    else:
        stage = "NOT_SET_UP"
    return {
        "stage": stage,
        "score": len(passed),
        "max_score": len(checks),
        "touched_buy_zone_recently": touched_recently,
        "rsi_14": clean_float(rsi_14, 2),
        "rsi_2": clean_float(rsi_2, 2),
        "stretch_percentile": clean_float(stretch_pctl),
        "checks": checks,
    }


def position_in_zone(price: float, low: float, high: float, near_band: float, kind: str) -> tuple[str, float]:
    """Classify price against a zone and return the signed % distance to its nearest edge.

    Distance is positive when price must move toward the zone (down for buy, up for sell)
    and zero inside the zone.
    """
    if kind == "buy":
        if price < low:
            return "BELOW", (price / low - 1)
        if price <= high:
            return "INSIDE", 0.0
        return ("NEAR" if price - high <= near_band else "AWAY"), (price / high - 1)
    if price > high:
        return "ABOVE", (price / high - 1)
    if price >= low:
        return "INSIDE", 0.0
    return ("NEAR" if low - price <= near_band else "AWAY"), (low / price - 1)

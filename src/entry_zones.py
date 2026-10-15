"""Price-zone analysis: where to buy short-term, buy long-term, and sell.

Zones are not forecasts. Each zone is the price range that would place the
ticker in an extreme percentile of its own historical deviation from a trend
line, and each zone ships with a replay of what price did after earlier visits.

Short-term deviation: (close - prior EMA20) / prior ATR14, so a zone is a
stretch measured in the ticker's own volatility units.
Long-term deviation: close / prior MA200 - 1.

Every level is causal: the band that applies to a session is computed from
data through the previous close, so the historical replay never sees the
session it is grading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import clean_float


SHORT_TERM_LOOKBACK = 756
LONG_TERM_LOOKBACK = 1260
MIN_HISTORY = 252
SHORT_TERM_BUY_PCTL = (0.05, 0.20)
SHORT_TERM_SELL_PCTL = (0.80, 0.95)
LONG_TERM_BUY_PCTL = (0.05, 0.25)
LONG_TERM_SELL_PCTL = (0.85, 0.97)
SHORT_TERM_HORIZON = 10
LONG_TERM_HORIZON = 60
LONG_TERM_COOLDOWN = 20
MIN_EPISODES = 8
BOUNCE_ATR = 1.5
RECENT_TOUCH_SESSIONS = 3

ZONE_KEYS = ("st_buy", "st_sell", "lt_buy", "lt_sell")


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
    frame["lt_dev"] = close / frame["ma200"].shift() - 1

    def quantile(series: pd.Series, lookback: int, level: float) -> pd.Series:
        return series.rolling(lookback, min_periods=MIN_HISTORY).quantile(level)

    for key, pctls in (("st_buy", SHORT_TERM_BUY_PCTL), ("st_sell", SHORT_TERM_SELL_PCTL)):
        for edge, level in zip(("low", "high"), pctls):
            q = quantile(frame["st_dev"], SHORT_TERM_LOOKBACK, level)
            frame[f"st_q{int(level * 100):02d}"] = q
            frame[f"next_{key}_{edge}"] = frame["ema20"] + frame["atr"] * q
    for key, pctls in (("lt_buy", LONG_TERM_BUY_PCTL), ("lt_sell", LONG_TERM_SELL_PCTL)):
        for edge, level in zip(("low", "high"), pctls):
            q = quantile(frame["lt_dev"], LONG_TERM_LOOKBACK, level)
            frame[f"next_{key}_{edge}"] = frame["ma200"] * (1 + q)
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


def confirmation_study(ohlc: pd.DataFrame, zones: pd.DataFrame, baseline_rate: float | None, horizon: int = SHORT_TERM_HORIZON) -> dict:
    """Replay the 'wait for the turn' entry: buy at the close of the first session
    that closes above the prior session's high within RECENT_TOUCH_SESSIONS of a
    buy-zone touch, then grade the same symmetric ATR bracket.
    """
    highs, lows, closes = (ohlc[column].to_numpy() for column in ("high", "low", "close"))
    atr = zones["atr"].to_numpy()
    band_high = zones["st_buy_high"].to_numpy()
    touched = np.isfinite(band_high) & (lows <= band_high)
    recent_touch = pd.Series(touched).rolling(RECENT_TOUCH_SESSIONS, min_periods=1).max().astype(bool).to_numpy()
    reclaim = np.concatenate([[False], closes[1:] > highs[:-1]])
    results: list[dict] = []
    last = -RECENT_TOUCH_SESSIONS - 1
    for i in np.flatnonzero(recent_touch & reclaim & np.isfinite(atr)):
        if i - last <= RECENT_TOUCH_SESSIONS:
            continue
        last = i
        entry = closes[i]
        outcome, _, favorable, _ = _bracket(
            highs, lows, int(i), horizon, entry + BOUNCE_ATR * atr[i], entry - BOUNCE_ATR * atr[i], "buy", False,
        )
        if outcome != "pending":
            results.append({"success": outcome == "success", "favorable_move": float(favorable.max() / entry - 1) if favorable.size else 0.0})
    success_rate = _rate([row["success"] for row in results])
    return {
        "episodes": len(results),
        "sufficient_sample": len(results) >= MIN_EPISODES,
        "success_rate": clean_float(success_rate),
        "baseline_success_rate": clean_float(baseline_rate),
        "edge_vs_baseline": (
            clean_float(success_rate - baseline_rate) if success_rate is not None and baseline_rate is not None else None
        ),
        "median_favorable_move": clean_float(_median([row["favorable_move"] for row in results])),
        "definition": (
            f"Buy at the close of the first session that closes above the prior high within {RECENT_TOUCH_SESSIONS} sessions "
            f"of a buy-zone touch; same {BOUNCE_ATR:g} ATR bracket over {horizon} sessions."
        ),
    }


def forward_return_study(
    ohlc: pd.DataFrame,
    zones: pd.DataFrame,
    side: str,
    horizon: int = LONG_TERM_HORIZON,
    cooldown: int = LONG_TERM_COOLDOWN,
) -> dict:
    """Replay earlier visits to a long-term zone and measure the return `horizon` sessions later."""
    prefix = "lt_buy" if side == "buy" else "lt_sell"
    band_low = zones[f"{prefix}_low"].to_numpy()
    band_high = zones[f"{prefix}_high"].to_numpy()
    ma200 = zones["ma200"]
    trend_up = (ma200.shift() > ma200.shift(21)).to_numpy()
    opens, highs, lows, closes = (ohlc[column].to_numpy() for column in ("open", "high", "low", "close"))
    valid = np.isfinite(band_low) & np.isfinite(band_high)
    touched = valid & ((lows <= band_high) if side == "buy" else (highs >= band_low))
    n = len(ohlc)
    episodes: list[dict] = []
    last_start = -cooldown - 1
    for i in np.flatnonzero(touched):
        if i - last_start <= cooldown:
            continue
        last_start = i
        entry = min(opens[i], band_high[i]) if side == "buy" else max(opens[i], band_low[i])
        complete = i + horizon <= n - 1
        end = min(i + horizon, n - 1)
        forward = closes[end] / entry - 1
        drawdown = lows[i:end + 1].min() / entry - 1
        run_up = highs[i:end + 1].max() / entry - 1
        success = forward > 0 if side == "buy" else forward < 0
        episodes.append(
            {
                "index": int(i),
                "date": str(ohlc.index[i].date()),
                "entry": float(entry),
                "outcome": ("success" if success else "failed") if complete else "pending",
                "success": bool(success),
                "complete": complete,
                "trend_up": bool(trend_up[i]),
                "forward_return": float(forward),
                "max_drawdown": float(drawdown),
                "max_run_up": float(run_up),
            }
        )
    graded = [episode for episode in episodes if episode["complete"]]
    current_trend_up = bool(ma200.iloc[-1] > ma200.iloc[-21]) if len(ma200.dropna()) > 21 else False
    return {
        "side": side,
        "horizon_sessions": horizon,
        "episodes": len(graded),
        "sufficient_sample": len(graded) >= MIN_EPISODES,
        "success_rate": clean_float(_rate([episode["success"] for episode in graded])),
        "median_forward_return": clean_float(_median([episode["forward_return"] for episode in graded])),
        "median_max_drawdown": clean_float(_median([episode["max_drawdown"] for episode in graded])),
        "median_max_run_up": clean_float(_median([episode["max_run_up"] for episode in graded])),
        "by_trend": _summarize_context(episodes, current_trend_up),
        "definition": (
            f"Visits at least {cooldown} sessions apart, entered at the zone edge. "
            + (
                f"Success = higher close {horizon} sessions later."
                if side == "buy"
                else f"Success = lower close {horizon} sessions later (the trim avoided a drawdown)."
            )
        ),
        "recent_episodes": episodes,
    }


def swing_levels(ohlc: pd.DataFrame, window: int = 5, lookback: int = 252) -> list[dict]:
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

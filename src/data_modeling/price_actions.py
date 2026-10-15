"""Deterministic, conditional price guidance for the next five sessions."""

import numpy as np
import pandas as pd

from ..utils import clean_float


def build_price_action_plan(features: pd.DataFrame, forecast: dict) -> dict:
    close = float(features.iloc[-1]["close"])
    if not np.isfinite(close) or close <= 0:
        raise ValueError("Price action guidance requires a positive reference close.")
    latest = features.iloc[-1]
    atr_pct = float(latest.get("atr_5_pct", np.nan))
    atr = close * atr_pct if np.isfinite(atr_pct) and atr_pct > 0 else close * 0.03
    atr = max(atr, close * 0.0001)
    history = features["close"].iloc[:-1].tail(20)
    prior_low = float(history.min()) if len(history) else close - atr
    prior_high = float(history.max()) if len(history) else close + atr
    support = max(min(prior_low, close - atr * 0.5), close * 0.01)
    resistance = max(prior_high, close + atr * 0.5)
    breakout = resistance + atr * 0.25
    stop = max(support - atr * 0.5, close * 0.001)
    target1 = max(breakout + atr, close + 2 * atr)
    p90 = forecast.get("price_distribution", {}).get("price_quantiles", {}).get("p90")
    target2 = max(target1 + atr, float(p90) if p90 is not None else target1 + atr)
    # Retain enough precision for small prices; UI may format cents for display.
    level = lambda value: clean_float(value, 6)
    levels = {
        "support": level(support), "bounce_watch_ceiling": level(support + atr * 0.25),
        "resistance": level(resistance), "breakout_confirmation": level(breakout),
        "invalidation": level(stop), "take_profit_1": level(target1), "take_profit_2": level(target2),
    }
    supported = bool(forecast.get("purchase_evidence", {}).get("supports_purchase"))
    rules = [
        ("at_or_below", "invalidation", "REDUCE_OR_EXIT", "A daily close at or below invalidation breaks this tactical setup. For an existing long, consider reducing or exiting; do not assume a rebound."),
        ("at_or_below", "bounce_watch_ceiling", "WATCH_FOR_BOUNCE", "Support test: wait for a later close reclaiming support, improving RSI above 45, and nonnegative money flow. A price touch alone cannot confirm a bounce or justify a new buy."),
        ("at_or_above", "take_profit_2", "TAKE_PROFIT_OR_TRAIL", "For an existing long with a gain after costs, consider taking further profit or trailing the remaining position. If the position is at a loss, this is a resistance/risk review, not profit taking."),
        ("at_or_above", "take_profit_1", "CONSIDER_PARTIAL_PROFIT", "For an existing long with a gain after costs, consider partial profit taking. Keep any remainder only while trend and support hold."),
        ("at_or_above", "breakout_confirmation", "CONFIRM_BREAKOUT", "Require a daily close above the trigger, volume above its 20-day average, and improving relative strength. Re-run the forecast before adding; touching the level alone is insufficient."),
        ("at_or_above", "resistance", "WATCH_FOR_REJECTION", "At resistance, watch for a failed close or upper-wick rejection. Existing profitable longs may consider trimming if rejection confirms; continued strength can instead lead to a breakout."),
    ]
    return {
        "reference_price": level(close),
        "as_of_date": forecast.get("as_of_date"),
        "horizon_trading_days": forecast.get("horizon_trading_days", 5),
        "price_basis": "adjusted_daily_close",
        "position_assumption": "conditional_management_of_an_existing_long; cost_basis_unknown",
        "levels": levels,
        "basis": "Prior 20-session closing support/resistance with adjusted 5-session ATR spacing; second profit level also considers the empirical 90th percentile. Fallback ATR is 3% if missing.",
        "new_entry_supported_by_validation": supported,
        "rules": [{"operator": operator, "level": key, "trigger_price": levels[key], "action": action, "guidance": guidance} for operator, key, action, guidance in rules],
        "default_action": "HOLD_OR_WAIT",
        "default_guidance": "For an existing long, monitor support and momentum. For a new entry, wait for a confirmed setup and refreshed validation.",
        "note": "These are conditional management levels, not predicted destinations or automatic orders. Profit depends on your entry price and costs. Bounce and intraday touch probabilities are not estimated. Recompute after each new daily close; gaps can bypass levels.",
    }

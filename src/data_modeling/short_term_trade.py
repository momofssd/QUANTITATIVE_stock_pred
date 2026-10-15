from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import clean_float, confidence_label


SHORT_TERM_ACTIONS = {
    "BUY_NOW_AT_SUPPORT": "Current price is in the tactical entry zone with a supportive five-session outlook.",
    "BUY_BREAKOUT_CONFIRMED": "Price has confirmed a short-term breakout with supportive forecast and participation.",
    "BUY_ON_PULLBACK": "Short-term outlook is constructive, but entry is preferred on a pullback into support.",
    "BUY_ON_BREAKOUT_CONFIRMATION": "Short-term outlook is constructive, but entry requires a confirmed breakout.",
    "WAIT": "The short-term trade does not yet have enough confirmation for a new buy.",
    "AVOID_SHORT_TERM_BUY": "The five-session outlook or technical condition does not support a short-term buy.",
}


def build_short_term_trade_plan(
    features: pd.DataFrame,
    regime: str,
    five_day_forecast: dict,
    price_zones: dict,
) -> dict:
    latest = features.iloc[-1]
    close = float(latest["close"])
    atr = float(latest.get("atr_14", close * 0.03) or close * 0.03)
    if not np.isfinite(atr) or atr <= 0:
        atr = close * 0.03
    ma20 = float(latest.get("ma20", np.nan))
    rsi = float(latest.get("rsi_14", 50) or 50)
    rs_20d = float(latest.get("rs_20d_change", 0) or 0)
    volume_ratio = float(latest.get("volume_vs_20d_avg", 0) or 0)

    tactical = price_zones["short_term_tactical_buy_zone"]
    pullback_low = float(tactical["low"])
    pullback_high = float(tactical["high"])
    prior_close_window = features["close"].iloc[:-1].tail(20)
    prior_20d_high = float(prior_close_window.max()) if not prior_close_window.empty else close + atr
    breakout_trigger = max(prior_20d_high * 1.002, close) if prior_close_window.size else close + (0.35 * atr)

    predicted_direction = five_day_forecast.get("predicted_direction")
    forecast_confidence = float(five_day_forecast.get("confidence_score", 0) or 0)
    bullish_probability = float(
        (five_day_forecast.get("direction_probabilities") or {}).get("bullish", 0) or 0
    )
    possible_close = float(five_day_forecast.get("possible_price_in_5_trading_days", close) or close)
    possible_range = five_day_forecast.get("possible_price_range") or {}
    direction_agrees = bool(five_day_forecast.get("direction_price_models_agree"))
    projected_positive = possible_close > close

    in_pullback_zone = pullback_low <= close <= pullback_high
    above_ma20 = not np.isfinite(ma20) or close >= ma20 * 0.99
    momentum_usable = 42 <= rsi <= 70
    not_bear_regime = regime not in {"BEAR_TREND", "TREND_DETERIORATION"}
    bullish_forecast = predicted_direction == "BULLISH" and bullish_probability >= 0.55 and projected_positive
    breakout_has_modeled_upside = possible_close > breakout_trigger
    breakout_confirmed = (
        close >= breakout_trigger
        and volume_ratio >= 1.0
        and rs_20d > 0
        and bullish_forecast
        and breakout_has_modeled_upside
    )
    purchase_evidence = five_day_forecast.get("purchase_evidence")
    validation_supports_buy = purchase_evidence is None or purchase_evidence.get("supports_purchase", False)

    if predicted_direction == "BEARISH" and forecast_confidence >= 0.55:
        action = "AVOID_SHORT_TERM_BUY"
    elif not bullish_forecast or not direction_agrees or not not_bear_regime or not validation_supports_buy:
        action = "WAIT"
    elif in_pullback_zone and above_ma20 and momentum_usable:
        action = "BUY_NOW_AT_SUPPORT"
    elif breakout_confirmed and momentum_usable:
        action = "BUY_BREAKOUT_CONFIRMED"
    elif close > pullback_high and rs_20d > 0 and breakout_has_modeled_upside:
        action = "BUY_ON_BREAKOUT_CONFIRMATION"
    elif possible_close > ((pullback_low + pullback_high) / 2):
        action = "BUY_ON_PULLBACK"
    else:
        action = "WAIT"

    is_good_buy_now = action in {"BUY_NOW_AT_SUPPORT", "BUY_BREAKOUT_CONFIRMED"}
    setup_available = action in {
        "BUY_NOW_AT_SUPPORT",
        "BUY_BREAKOUT_CONFIRMED",
        "BUY_ON_PULLBACK",
        "BUY_ON_BREAKOUT_CONFIRMATION",
    }
    primary_entry_type = "BREAKOUT_ENTRY" if action in {"BUY_BREAKOUT_CONFIRMED", "BUY_ON_BREAKOUT_CONFIRMATION"} else "PULLBACK_ENTRY"
    primary_entry_price = breakout_trigger if primary_entry_type == "BREAKOUT_ENTRY" else (pullback_low + pullback_high) / 2
    if primary_entry_type == "BREAKOUT_ENTRY":
        stop_price = breakout_trigger - (1.0 * atr)
    else:
        stop_price = pullback_low - (0.75 * atr)
    stop_price = min(stop_price, primary_entry_price * 0.995)
    risk_per_share = max(primary_entry_price - stop_price, 0.01)
    expected_reward = max(possible_close - primary_entry_price, 0)
    trade_risk_reward = expected_reward / risk_per_share

    technical_confirmation = sum(
        [
            bullish_forecast,
            direction_agrees,
            above_ma20,
            momentum_usable,
            rs_20d > 0,
            not_bear_regime,
        ]
    ) / 6
    confidence_score = (forecast_confidence * 0.65) + (technical_confirmation * 0.35)
    if action in {"WAIT", "AVOID_SHORT_TERM_BUY"}:
        confidence_score = forecast_confidence

    reasons: list[str] = []
    risks: list[str] = []
    if not validation_supports_buy:
        risks.append("Held-out validation does not demonstrate sufficient evidence for a current-price purchase.")
    if predicted_direction == "BULLISH":
        reasons.append(f"Five-session model favors a bullish move with {forecast_confidence:.1%} confidence.")
    else:
        risks.append(f"Five-session model favors a bearish move with {forecast_confidence:.1%} confidence.")
    if direction_agrees:
        reasons.append("The direction and possible-close models agree.")
    else:
        risks.append("The direction and possible-close models disagree.")
    if in_pullback_zone:
        reasons.append("Current price is inside the tactical pullback entry zone.")
    if rs_20d > 0:
        reasons.append("Relative strength is improving.")
    else:
        risks.append("Relative strength is not confirming a short-term buy.")
    if rsi > 70:
        risks.append("RSI is elevated, increasing chase risk.")
    if not not_bear_regime:
        risks.append("The broader regime is unfavorable for a long trade.")

    return {
        "horizon_trading_days": int(five_day_forecast.get("horizon_trading_days", 5) or 5),
        "action": action,
        "action_definition": SHORT_TERM_ACTIONS[action],
        "is_good_short_term_buy_now": is_good_buy_now,
        "has_conditional_buy_setup": setup_available,
        "validation_supports_buy": validation_supports_buy if purchase_evidence is not None else None,
        "probability_of_gain_from_current_close": five_day_forecast.get("probability_of_gain"),
        "confidence_score": clean_float(min(max(confidence_score, 0), 1)),
        "confidence_label": confidence_label(confidence_score),
        "summary": _short_term_summary(action, pullback_low, pullback_high, breakout_trigger),
        "primary_entry_type": primary_entry_type,
        "entry_points": {
            "pullback_entry": {
                "low": clean_float(pullback_low, 2),
                "high": clean_float(pullback_high, 2),
                "status": "CURRENT_PRICE_IN_ZONE" if in_pullback_zone else "WAIT_FOR_PULLBACK",
                "confirmation_rules": tactical.get("confirmation_rules", []),
            },
            "breakout_entry": {
                "trigger_price": clean_float(breakout_trigger, 2),
                "status": "CONFIRMED" if breakout_confirmed else "WAIT_FOR_CONFIRMED_CLOSE",
                "confirmation_rules": [
                    "daily close at or above trigger",
                    "volume at or above 20-day average",
                    "relative strength improving",
                ],
            },
        },
        "trade_management": {
            "active_for_recommended_setup": setup_available,
            "primary_entry_reference": clean_float(primary_entry_price, 2),
            "stop_loss_price": clean_float(stop_price, 2) if setup_available else None,
            "possible_close_in_5_trading_days": clean_float(possible_close, 2),
            "possible_close_range": {
                "low": clean_float(possible_range.get("low"), 2),
                "high": clean_float(possible_range.get("high"), 2),
            },
            "estimated_reward_to_risk": clean_float(trade_risk_reward, 2) if setup_available else None,
            "basis": "five-session possible close versus tactical entry and ATR-based stop",
        },
        "reasons": reasons[:4],
        "risks": risks[:4],
        "not_financial_advice": True,
    }


def _short_term_summary(action: str, low: float, high: float, breakout: float) -> str:
    if action == "BUY_NOW_AT_SUPPORT":
        return "Current price is in the tactical support zone and the five-session outlook supports a short-term buy."
    if action == "BUY_BREAKOUT_CONFIRMED":
        return "A confirmed short-term breakout is present with supportive forecast evidence."
    if action == "BUY_ON_PULLBACK":
        return f"Do not chase current price; consider entry only on a supported pullback near {low:.2f}-{high:.2f}."
    if action == "BUY_ON_BREAKOUT_CONFIRMATION":
        return f"Wait for short-term breakout confirmation at or above {breakout:.2f} before considering entry."
    if action == "AVOID_SHORT_TERM_BUY":
        return "The five-session forecast does not support opening a short-term long position now."
    return "Wait for bullish model agreement and technical confirmation before a short-term buy."

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils import clean_float, confidence_label


SIGNAL_DEFINITIONS = {
    "STRONG_LONG_TERM_BUY": "Strong long-term setup near a favorable entry area; normal long-term exposure may be appropriate.",
    "LONG_TERM_BUY": "Constructive long-term setup at an acceptable entry area; consider initiating or adding exposure.",
    "ACCUMULATE_ON_PULLBACK": "Trend is constructive, but price is above the preferred buy area; add only on a pullback toward support.",
    "HOLD": "Setup remains constructive, but evidence is not strong enough for a new buy signal; maintain existing exposure.",
    "WAIT": "Evidence or confidence is insufficient for entry; wait for better price or stronger confirmation.",
    "TRIM": "Price is extended or near resistance; consider reducing part of an existing position and avoid adding.",
    "SELL": "Trend is broken with meaningful bearish model support; consider exiting or materially reducing exposure.",
    "AVOID": "Risk is elevated or technical support is weak; avoid new exposure.",
}

ALLOWED_SIGNALS = set(SIGNAL_DEFINITIONS)


@dataclass
class DecisionResult:
    decision: dict
    price_zones: dict
    near_term_forecast: dict
    risk_management: dict
    technical_score: int
    reasons: list[str]
    risks: list[str]


def _zone(low: float, high: float) -> tuple[float, float]:
    low, high = sorted([float(low), float(high)])
    if low == high:
        high = low * 1.01
    return low, high


def calculate_decision(
    features: pd.DataFrame,
    regime: str,
    probabilities: dict[str, float],
) -> DecisionResult:
    latest = features.iloc[-1]
    close = float(latest["close"])
    ma50 = float(latest.get("ma50", np.nan))
    ma200 = float(latest.get("ma200", np.nan))
    atr = float(latest.get("atr_14", close * 0.03) or close * 0.03)
    atr_pct = float(latest.get("atr_pct", 0.03) or 0.03)
    rsi = float(latest.get("rsi_14", 50) or 50)
    prior_low = float(latest.get("prior_120d_low", close * 0.9) or close * 0.9)
    prior_high = float(latest.get("prior_120d_high", close * 1.1) or close * 1.1)
    prior_252_high = float(latest.get("prior_252d_high", prior_high) or prior_high)
    close_to_50 = float(latest.get("close_to_50ma_pct", 0) or 0)
    close_to_200 = float(latest.get("close_to_200ma_pct", 0) or 0)
    rs_20d = float(latest.get("rs_20d_change", 0) or 0)
    vix_change = float(latest.get("vix_20d_change", 0) or 0)

    buy_candidates = [
        ma50 if np.isfinite(ma50) else close * 0.95,
        ma200 if np.isfinite(ma200) else close * 0.88,
        prior_low,
        close - (1.5 * atr),
    ]
    buy_center = float(np.nanmedian(buy_candidates))
    buy_low, buy_high = _zone(buy_center - atr, buy_center + 0.5 * atr)
    buy_high = min(buy_high, close * 1.02)

    recent_20d_low = float(features["close"].tail(20).min())
    ma20 = float(latest.get("ma20", np.nan))
    short_term_candidates = [
        ma20 if np.isfinite(ma20) else close - atr,
        ma50 if np.isfinite(ma50) else close - 2 * atr,
        recent_20d_low,
        close - atr,
    ]
    short_term_center = float(np.nanmedian(short_term_candidates))
    short_buy_low, short_buy_high = _zone(short_term_center - 0.5 * atr, short_term_center + 0.35 * atr)
    short_buy_high = min(short_buy_high, close * 1.01)

    breakout_trigger = max(prior_high, close + atr)
    trim_low, trim_high = _zone(max(prior_high, close + 2 * atr), max(prior_252_high, close + 4 * atr))
    recent_20d_high = float(features["close"].tail(20).max())
    recent_60d_high = float(features["close"].tail(60).max())
    base_low, base_high = _zone(max(short_buy_high, close - 0.5 * atr), min(max(close + atr, recent_20d_high), breakout_trigger))
    bullish_low, bullish_high = _zone(max(breakout_trigger, close + atr), max(trim_low, recent_60d_high, close + 2.5 * atr))
    bearish_low, bearish_high = _zone(max(invalidation if "invalidation" in locals() else close - 3 * atr, short_buy_low - atr), short_buy_low)
    stop_padding = max(0.03, min(0.08, atr_pct * 1.5))
    invalidation = min(ma200 * (1 - stop_padding) if np.isfinite(ma200) else close - 3 * atr, prior_low - atr, close - 2 * atr)
    invalidation = min(invalidation, close * 0.98)
    bearish_low, bearish_high = _zone(max(invalidation, short_buy_low - atr), short_buy_low)

    downside = max(close - invalidation, 0.01)
    upside = max(((trim_low + trim_high) / 2) - close, 0.01)
    risk_reward = upside / downside

    score = 0
    reasons: list[str] = []
    risks: list[str] = []
    if latest.get("close_above_200ma", 0):
        score += 15
        reasons.append("Price is above the 200-day moving average.")
    else:
        risks.append("Price is below the 200-day moving average.")
    if latest.get("ma50_above_ma200", 0):
        score += 10
        reasons.append("The 50-day average is above the 200-day average.")
    if float(latest.get("ma200_slope", 0) or 0) >= -0.0002:
        score += 10
    near_support = close <= buy_high * 1.03 or abs(close_to_50) <= 0.05 or abs(close_to_200) <= 0.06
    if near_support:
        score += 15
        reasons.append("Current price is close to a long-term support area.")
    if rs_20d > 0:
        score += 10
        reasons.append("Relative strength is improving.")
    else:
        risks.append("Relative strength is not confirming the setup.")
    if not regime.startswith("BEAR"):
        score += 15
    if vix_change < 0.20:
        score += 5
    else:
        risks.append("VIX is rising quickly.")
    if risk_reward >= 2:
        score += 10
        reasons.append("Risk/reward is favorable versus the trim zone.")
    if probabilities.get("bullish", 0) >= 0.55:
        score += 10
        reasons.append("The classifier favors a bullish forward regime.")

    extended = close_to_50 > 0.12 or close_to_200 > 0.25 or rsi >= 72
    trend_broken = regime in {"BEAR_TREND", "TREND_DETERIORATION"} or close_to_200 < -0.04
    far_above_buy_zone = close > buy_high * 1.08
    if trend_broken and probabilities.get("bearish", 0) > 0.45:
        final_signal = "SELL"
    elif trend_broken:
        final_signal = "AVOID"
    elif extended and close >= trim_low * 0.97:
        final_signal = "TRIM"
        risks.append("Price is extended near a trim zone.")
    elif score >= 80 and not far_above_buy_zone:
        final_signal = "STRONG_LONG_TERM_BUY"
    elif score >= 65 and not far_above_buy_zone:
        final_signal = "LONG_TERM_BUY"
    elif score >= 50:
        final_signal = "ACCUMULATE_ON_PULLBACK" if close > buy_high else "HOLD"
    elif score >= 40:
        final_signal = "WAIT"
    else:
        final_signal = "AVOID"

    model_strength = max(probabilities.values()) if probabilities else 0.34
    technical_strength = min(max(score / 100, 0), 1)
    regime_alignment = 0.9 if final_signal in {"LONG_TERM_BUY", "STRONG_LONG_TERM_BUY", "ACCUMULATE_ON_PULLBACK"} and not regime.startswith("BEAR") else 0.55
    rs_confirmation = 1.0 if rs_20d > 0 else 0.3
    rr_quality = min(risk_reward / 3, 1)
    agreement_count = sum(
        [
            score >= 65,
            probabilities.get("bullish", 0) >= probabilities.get("bearish", 0),
            rs_20d > 0,
            risk_reward >= 1.5,
            not trend_broken,
        ]
    ) / 5
    confidence_score = (
        model_strength * 0.30
        + technical_strength * 0.25
        + regime_alignment * 0.15
        + rs_confirmation * 0.10
        + rr_quality * 0.10
        + agreement_count * 0.10
    )
    confidence_score = float(min(max(confidence_score, 0), 1))
    if confidence_score < 0.50 and final_signal in {"LONG_TERM_BUY", "STRONG_LONG_TERM_BUY"}:
        final_signal = "WAIT"

    is_buy = final_signal in {"STRONG_LONG_TERM_BUY", "LONG_TERM_BUY"}
    is_sell = final_signal in {"TRIM", "SELL"}
    summary = _summary(final_signal, regime, close, buy_low, buy_high)

    decision = {
        "final_signal": final_signal,
        "is_current_price_long_term_buy": is_buy,
        "is_current_price_long_term_sell": is_sell,
        "confidence_score": round(confidence_score, 4),
        "confidence_label": confidence_label(confidence_score),
        "summary": summary,
    }
    price_zones = {
        "preferred_long_term_buy_zone": {
            "low": clean_float(buy_low, 2),
            "high": clean_float(buy_high, 2),
            "basis": ["50MA/200MA support", "120-day range", "ATR pullback zone"],
            "confidence_score": round(min(confidence_score + 0.03, 1), 4),
        },
        "breakout_buy_zone": {
            "trigger_price": clean_float(breakout_trigger, 2),
            "confirmation_rules": ["close above resistance", "volume above 20d average", "relative strength rising"],
            "confidence_score": round(max(confidence_score - 0.05, 0), 4),
        },
        "short_term_tactical_buy_zone": {
            "low": clean_float(short_buy_low, 2),
            "high": clean_float(short_buy_high, 2),
            "basis": ["20MA support", "recent 20-day low", "ATR pullback", "50MA reference"],
            "confirmation_rules": [
                "price holds above the short-term zone",
                "RSI remains above 45-50",
                "volume does not expand sharply on down days",
            ],
            "confidence_score": round(max(confidence_score - 0.08, 0), 4),
        },
        "trim_or_sell_zone": {
            "low": clean_float(trim_low, 2),
            "high": clean_float(trim_high, 2),
            "basis": ["prior 120/252-day high", "ATR extension", "overextension check"],
            "confidence_score": round(max(confidence_score - 0.02, 0), 4),
        },
        "invalidation_zone": {
            "price": clean_float(invalidation, 2),
            "basis": ["below 200MA", "below 120-day low", "2 ATR risk band"],
            "confidence_score": round(min(confidence_score + 0.02, 1), 4),
        },
    }
    near_term_forecast = {
        "purpose": "AI should choose a likely near-term scenario and timeframe from these quant-computed levels.",
        "timeframe_scope": "near_term_swing",
        "horizon_candidates": [
            "2-4 weeks if breakout or momentum continuation confirms",
            "4-8 weeks if price consolidates before moving",
            "1-3 months if volatility remains elevated or macro risk delays follow-through",
        ],
        "bullish_scenario": {
            "target_zone_low": clean_float(bullish_low, 2),
            "target_zone_high": clean_float(bullish_high, 2),
            "basis": ["breakout trigger", "recent 60-day high", "ATR extension", "trim zone lower bound"],
            "probability_hint": clean_float(probabilities.get("bullish", 0)),
            "confirmation_rules": ["close above breakout trigger", "relative strength remains positive", "volume confirms move"],
        },
        "base_case_scenario": {
            "target_zone_low": clean_float(base_low, 2),
            "target_zone_high": clean_float(base_high, 2),
            "basis": ["current price", "20-day high", "ATR drift range", "breakout resistance"],
            "probability_hint": clean_float(probabilities.get("sideways", 0)),
            "confirmation_rules": ["price holds above short-term tactical zone", "ATR does not expand sharply", "RSI remains constructive"],
        },
        "bearish_scenario": {
            "pullback_zone_low": clean_float(bearish_low, 2),
            "pullback_zone_high": clean_float(bearish_high, 2),
            "basis": ["short-term tactical zone failure", "ATR pullback", "invalidation reference"],
            "probability_hint": clean_float(probabilities.get("bearish", 0)),
            "confirmation_rules": ["close below short-term tactical zone", "relative strength weakens", "down-volume expands"],
        },
        "ai_rules": [
            "Select one likely near-term scenario from bullish_scenario, base_case_scenario, or bearish_scenario.",
            "Choose an estimated timeframe from horizon_candidates or a narrower range justified by the data.",
            "Do not invent target prices outside these scenario zones.",
            "Express the forecast as probabilistic, not certain.",
        ],
    }
    risk_management = {
        "risk_reward_ratio": clean_float(risk_reward, 2),
        "suggested_position_size": _position_size(final_signal, confidence_score),
        "max_loss_to_invalidation_pct": clean_float(invalidation / close - 1),
        "upside_to_trim_zone_pct": clean_float(((trim_low + trim_high) / 2) / close - 1),
        "notes": [
            "Do not use as a day-trading signal.",
            "Signal becomes invalid if price closes below invalidation zone.",
        ],
    }
    if not risks:
        risks.append("Main risk is a failed breakout or broad market regime deterioration.")
    return DecisionResult(decision, price_zones, near_term_forecast, risk_management, score, reasons[:6], risks[:6])


def _summary(signal: str, regime: str, close: float, buy_low: float, buy_high: float) -> str:
    if signal in {"STRONG_LONG_TERM_BUY", "LONG_TERM_BUY"}:
        return "Current price is inside or close to a favorable long-term buy area with supportive trend evidence."
    if signal == "ACCUMULATE_ON_PULLBACK":
        return f"Trend is constructive, but the preferred long-term buy zone is around {buy_low:.2f}-{buy_high:.2f}."
    if signal == "TRIM":
        return "Price is extended or near resistance; trimming risk is favored over adding new exposure."
    if signal in {"SELL", "AVOID"}:
        return f"Risk is elevated under the current {regime.lower().replace('_', ' ')} regime."
    return "Model cannot confirm a favorable long-term risk/reward entry at the current price."


def _position_size(signal: str, confidence_score: float) -> str:
    if signal in {"STRONG_LONG_TERM_BUY", "LONG_TERM_BUY"} and confidence_score >= 0.65:
        return "NORMAL"
    if signal in {"ACCUMULATE_ON_PULLBACK", "HOLD"}:
        return "PARTIAL_OR_EXISTING_ONLY"
    if signal in {"TRIM", "SELL", "AVOID"}:
        return "REDUCE_OR_NONE"
    return "NONE_OR_SMALL"

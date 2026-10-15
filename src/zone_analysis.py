"""Assemble the price-zone report served to the dashboard.

No model is trained here. The report states where price is relative to
historically calibrated buy and sell zones, and what happened after earlier
visits to those zones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import MarketDataUnavailableError, load_market_data
from .entry_zones import (
    BOUNCE_ATR,
    LONG_TERM_BUY_PCTL,
    LONG_TERM_HORIZON,
    LONG_TERM_LOOKBACK,
    LONG_TERM_SELL_PCTL,
    SHORT_TERM_BUY_PCTL,
    SHORT_TERM_HORIZON,
    SHORT_TERM_LOOKBACK,
    SHORT_TERM_SELL_PCTL,
    ZONE_KEYS,
    adjusted_ohlcv,
    bounce_readiness,
    breakdown_state,
    breakdown_study,
    compute_zone_frame,
    confirmation_study,
    forward_return_study,
    position_in_zone,
    reversion_study,
    swing_levels,
    zone_confluence,
    percentile_rank,
)
from .features import build_features
from .json_exporter import build_macro_context
from .trend_probability import TREND_HORIZONS, trend_outlook
from .utils import clean_float


CHART_PERIODS = 1260

SHORT_TERM_ACTIONS = {
    "SUPPORT_BREAKDOWN": ("Support breakdown", "negative", "Price closed below its most recent swing-low support and has not reclaimed it; breakdowns can keep falling. Do not buy until price closes back above the broken level."),
    "ENTRY_CONFIRMED": ("Entry confirmed", "positive", "Price tagged the buy zone and has reclaimed the prior session's high; the bounce has started."),
    "ENTRY_ZONE_BOUNCE_FORMING": ("In entry zone, bounce forming", "positive", "Price is in the buy zone and printed a reversal candle; confirmation is a close above today's high."),
    "ENTRY_ZONE": ("In entry zone", "caution", "Price is in the buy zone but has not printed a reversal signal; scale in or wait for a reversal candle."),
    "BELOW_ENTRY_ZONE": ("Below entry zone", "negative", "Price has fallen through the buy zone; do not catch the knife until it closes back inside with a reversal."),
    "APPROACHING_ENTRY": ("Approaching entry", "caution", "Price is within one ATR of the buy zone; set an alert at the top of the zone."),
    "BOUNCE_UNDERWAY": ("Bounce underway", "neutral", "Price recently bounced from the buy zone and is now above it; chasing worsens reward/risk."),
    "WAIT_FOR_PULLBACK": ("Wait for pullback", "neutral", "Price is between zones; wait for a pullback into the buy zone."),
    "TAKE_PROFIT_ZONE": ("Take-profit zone", "negative", "Price is stretched into the short-term sell zone; take profits on existing longs and avoid new entries."),
}

LONG_TERM_ACTIONS = {
    "ACCUMULATE": ("Accumulate", "positive", "Price is inside the long-term value zone relative to its 200-day average."),
    "DEEP_VALUE_TREND_RISK": ("Below value zone", "caution", "Price is cheaper than the value zone; historically rare, but the long-term trend may be breaking. Scale in slowly."),
    "APPROACHING_VALUE": ("Approaching value zone", "caution", "Price is close to the long-term value zone."),
    "HOLD_WAIT": ("Hold / wait", "neutral", "Price is between the long-term value and trim zones."),
    "APPROACHING_TRIM": ("Approaching trim zone", "caution", "Price is close to the long-term trim zone; avoid adding."),
    "TRIM_ZONE": ("Trim zone", "negative", "Price is extended far above its 200-day average; consider trimming long-term holdings."),
}


def _zone_payload(frame: pd.DataFrame, key: str, close: float, near_band: float, kind: str, percentiles: tuple[float, float], basis: str, confluence: list[str]) -> dict:
    last = frame.iloc[-1]
    low, high = float(last[f"next_{key}_low"]), float(last[f"next_{key}_high"])
    position, distance = position_in_zone(close, low, high, near_band, kind)
    atr = float(last["atr"])
    edge = high if kind == "buy" else low
    return {
        "low": clean_float(low, 2),
        "high": clean_float(high, 2),
        "position": position,
        "distance_pct": clean_float(distance),
        "distance_atr": clean_float(abs(close - edge) / atr if position in {"NEAR", "AWAY"} else 0.0, 2),
        "percentile_band": [int(percentiles[0] * 100), int(percentiles[1] * 100)],
        "basis": basis,
        "confluence": confluence,
    }


def _edge_grade(study: dict) -> dict:
    """Grade how much better a replayed entry did than entering on an arbitrary day."""
    edge = study.get("edge_vs_baseline")
    if not study.get("sufficient_sample") or edge is None:
        grade, label = "THIN_SAMPLE", "Too few past setups to judge"
    elif edge >= 0.05:
        grade, label = "EDGE", "Historically better than an average day"
    elif edge > 0:
        grade, label = "SLIGHT_EDGE", "Marginally better than an average day"
    else:
        grade, label = "NO_EDGE", "No better than an average day"
    return {"grade": grade, "label": label, "edge_vs_baseline": edge, "success_rate": study.get("success_rate"), "baseline_success_rate": study.get("baseline_success_rate"), "episodes": study.get("episodes")}


def _trim_study(study: dict) -> dict:
    return {key: value for key, value in study.items() if key != "recent_episodes"}


def _short_term_view(ohlc: pd.DataFrame, features: pd.DataFrame, zones: pd.DataFrame, market_close: pd.Series | None, swings: list[dict]) -> tuple[dict, dict, dict]:
    last = zones.iloc[-1]
    close = float(ohlc["close"].iloc[-1])
    atr = float(last["atr"])
    anchors = {"20-day EMA": float(last["ema20"]), "50-day MA": float(last["ma50"]), "200-day MA": float(last["ma200"])}
    buy_low, buy_high = float(last["next_st_buy_low"]), float(last["next_st_buy_high"])
    sell_low, sell_high = float(last["next_st_sell_low"]), float(last["next_st_sell_high"])
    buy_zone = _zone_payload(
        zones, "st_buy", close, atr, "buy", SHORT_TERM_BUY_PCTL,
        "20-day EMA minus the ticker's 5th-20th percentile stretch (in ATRs) over the last 3 years",
        zone_confluence(buy_low, buy_high, atr, anchors, swings, "buy"),
    )
    sell_zone = _zone_payload(
        zones, "st_sell", close, atr, "sell", SHORT_TERM_SELL_PCTL,
        "20-day EMA plus the ticker's 80th-95th percentile stretch (in ATRs) over the last 3 years",
        zone_confluence(sell_low, sell_high, atr, anchors, swings, "sell"),
    )
    readiness = bounce_readiness(ohlc, features, zones, market_close)
    buy_study = reversion_study(ohlc, zones, "buy")
    sell_study = reversion_study(ohlc, zones, "sell")
    confirmed_study = confirmation_study(ohlc, zones, buy_study["baseline_success_rate"])
    breakdown = breakdown_state(ohlc, zones, swings)
    broken_study = breakdown_study(ohlc, zones, sell_study["baseline_success_rate"])

    buy_position, sell_position = buy_zone["position"], sell_zone["position"]
    touched = readiness["touched_buy_zone_recently"]
    if sell_position in {"INSIDE", "ABOVE"}:
        action = "TAKE_PROFIT_ZONE"
    elif breakdown["active"]:
        # A close through support overrides the bounce reading until the level is reclaimed.
        action = "SUPPORT_BREAKDOWN"
    elif touched and readiness["stage"] == "CONFIRMED":
        action = "BOUNCE_UNDERWAY" if buy_position == "AWAY" else "ENTRY_CONFIRMED"
    elif buy_position == "INSIDE":
        action = "ENTRY_ZONE_BOUNCE_FORMING" if readiness["stage"] == "FORMING" else "ENTRY_ZONE"
    elif buy_position == "BELOW":
        action = "BELOW_ENTRY_ZONE"
    elif buy_position == "NEAR":
        action = "APPROACHING_ENTRY"
    elif touched:
        action = "BOUNCE_UNDERWAY"
    else:
        action = "WAIT_FOR_PULLBACK"
    label, tone, definition = SHORT_TERM_ACTIONS[action]
    evidence = {"ENTRY_CONFIRMED": confirmed_study, "SUPPORT_BREAKDOWN": broken_study}.get(action, buy_study)
    edge = _edge_grade(evidence)
    # A setup this ticker's own history does not reward is never shown as a green light.
    if tone == "positive" and edge["grade"] not in {"EDGE", "SLIGHT_EDGE"}:
        tone = "caution"

    if action in {"ENTRY_CONFIRMED", "ENTRY_ZONE_BOUNCE_FORMING", "ENTRY_ZONE"}:
        reference, reference_basis = close, "current close"
    elif action == "SUPPORT_BREAKDOWN":
        reference, reference_basis = breakdown["broken_level"], "close back above the broken support (failed breakdown)"
    elif buy_position == "BELOW":
        # Below the zone a limit at its top would sit above the stop; wait for a close back inside instead.
        reference, reference_basis = buy_low, "close back above the bottom of the buy zone (reclaim)"
    else:
        reference, reference_basis = buy_high, "top of buy zone (limit entry)"
    stop = reference - BOUNCE_ATR * atr
    target_1 = reference + BOUNCE_ATR * atr
    target_2 = max(sell_low, target_1)
    risk = max(reference - stop, 0.01)
    trade_plan = {
        "entry_reference": clean_float(reference, 2),
        "entry_reference_basis": reference_basis,
        "stop_loss": clean_float(stop, 2),
        "stop_basis": f"{BOUNCE_ATR:g} ATR below entry, the same bracket the history replay uses",
        "target_1": clean_float(target_1, 2),
        "target_1_basis": f"{BOUNCE_ATR:g} ATR bounce, the replay's success level",
        "target_2": clean_float(target_2, 2),
        "target_2_basis": "bottom of the short-term sell zone (full swing)",
        "mean_reversion_level": clean_float(last["ema20"], 2),
        "reward_risk_target_1": clean_float((target_1 - reference) / risk, 2),
        "reward_risk_target_2": clean_float((target_2 - reference) / risk, 2),
    }

    context = buy_study["by_trend"]
    reasons, risks = [], []
    if buy_study["sufficient_sample"]:
        reasons.append(
            f"{buy_study['success_rate']:.0%} of {buy_study['episodes']} earlier buy-zone touches bounced {BOUNCE_ATR:g} ATR "
            f"before dropping {BOUNCE_ATR:g} ATR (any-day baseline {buy_study['baseline_success_rate']:.0%}); "
            f"median best bounce {buy_study['median_favorable_move']:+.1%} within {SHORT_TERM_HORIZON} sessions."
        )
        if (buy_study["edge_vs_baseline"] or 0) <= 0:
            risks.append("Buying at this zone has not beaten buying on an arbitrary day for this ticker.")
        if confirmed_study["sufficient_sample"] and (confirmed_study["edge_vs_baseline"] or 0) <= 0:
            risks.append("Waiting for a reversal close has not beaten buying on an arbitrary day for this ticker.")
        if context["current_context_success_rate"] is not None and context["current_context_episodes"] >= 5:
            reasons.append(
                f"In the current {context['current_context']} context (vs. 200-day MA) the bounce rate was "
                f"{context['current_context_success_rate']:.0%} over {context['current_context_episodes']} touches."
            )
    else:
        risks.append(f"Only {buy_study['episodes']} completed buy-zone touches in the history; statistics are thin.")
    if action == "TAKE_PROFIT_ZONE" and sell_study["sufficient_sample"]:
        reasons.insert(
            0,
            f"{sell_study['success_rate']:.0%} of {sell_study['episodes']} earlier sell-zone touches pulled back {BOUNCE_ATR:g} ATR "
            f"before extending {BOUNCE_ATR:g} ATR higher (baseline {sell_study['baseline_success_rate']:.0%}).",
        )
    if close < float(last["ma200"]):
        risks.append("Price is below its 200-day average; bounces in downtrends fail more often.")
    if action == "SUPPORT_BREAKDOWN":
        # Bounce statistics argue for the opposite trade, so the breakdown evidence replaces them.
        reasons = [
            f"Closed below swing-low support at {breakdown['broken_level']:.2f} on {breakdown['breakdown_date']}; "
            f"still {abs(breakdown['close_vs_broken_level_pct']):.1%} below it."
        ]
        if broken_study["sufficient_sample"]:
            reasons.append(
                f"{broken_study['success_rate']:.0%} of {broken_study['episodes']} earlier breakdowns fell another {BOUNCE_ATR:g} ATR "
                f"before rebounding {BOUNCE_ATR:g} ATR (any-day baseline {broken_study['baseline_success_rate']:.0%}); "
                f"median further drop {broken_study['median_favorable_move']:.1%}.",
            )
            if (broken_study["edge_vs_baseline"] or 0) <= 0:
                risks.insert(0, "Breakdowns below support have not kept falling more often than an arbitrary day for this ticker.")
            if broken_study["reclaim_rate"] is not None:
                risks.append(f"{broken_study['reclaim_rate']:.0%} of earlier breakdowns closed back above the broken level within {SHORT_TERM_HORIZON} sessions.")
        else:
            risks.insert(0, f"Only {broken_study['episodes']} completed support breakdowns in the history; statistics are thin.")
    elif buy_position in {"INSIDE", "BELOW", "NEAR"} or touched:
        reasons += [f"{check['label']}: {check['detail']}." for check in readiness["checks"] if check["passed"]]
        if not any(check["passed"] for check in readiness["checks"] if check["key"] in {"reversal_candle", "reclaim"}):
            risks.append("No reversal candle or reclaim yet; selling pressure may not be finished.")
    market_check = next(check for check in readiness["checks"] if check["key"] == "market_backdrop")
    if not market_check["passed"]:
        risks.append(market_check["detail"] + ".")

    view = {
        "horizon_sessions": SHORT_TERM_HORIZON,
        "action": action,
        "action_label": label,
        "tone": tone,
        "action_definition": definition,
        "is_entry_now": action in {"ENTRY_CONFIRMED", "ENTRY_ZONE_BOUNCE_FORMING", "ENTRY_ZONE"},
        "historical_edge": edge,
        "buy_zone": buy_zone,
        "sell_zone": sell_zone,
        "stretch": {
            "current_atr": clean_float(last["st_dev"], 2),
            "percentile": clean_float(percentile_rank(zones["st_dev"], SHORT_TERM_LOOKBACK)),
            "thresholds_atr": {
                f"p{int(level * 100):02d}": clean_float(last[f"st_q{int(level * 100):02d}"], 2)
                for level in (*SHORT_TERM_BUY_PCTL, *SHORT_TERM_SELL_PCTL)
            },
        },
        "bounce_readiness": readiness,
        "breakdown": breakdown,
        "trade_plan": trade_plan,
        "history": {
            "buy_zone": _trim_study(buy_study),
            "confirmed_entry": confirmed_study,
            "sell_zone": _trim_study(sell_study),
            "breakdown": _trim_study(broken_study),
        },
        "reasons": reasons[:6],
        "risks": risks[:5],
    }
    return view, buy_study, sell_study


def _long_term_view(ohlc: pd.DataFrame, zones: pd.DataFrame, swings: list[dict]) -> tuple[dict, dict | None, dict | None]:
    last = zones.iloc[-1]
    if not np.isfinite(last[["next_lt_buy_low", "next_lt_buy_high", "next_lt_sell_low", "next_lt_sell_high"]].astype(float)).all():
        return {
            "available": False,
            "action": "INSUFFICIENT_HISTORY",
            "action_label": "Insufficient history",
            "tone": "neutral",
            "action_definition": "Long-term zones need at least about two years of daily history. Choose an earlier start date.",
        }, None, None
    close = float(ohlc["close"].iloc[-1])
    atr = float(last["atr"])
    near_band = max(2 * atr, 0.03 * close)
    anchors = {"50-day MA": float(last["ma50"]), "200-day MA": float(last["ma200"])}
    buy_low, buy_high = float(last["next_lt_buy_low"]), float(last["next_lt_buy_high"])
    sell_low, sell_high = float(last["next_lt_sell_low"]), float(last["next_lt_sell_high"])
    buy_zone = _zone_payload(
        zones, "lt_buy", close, near_band, "buy", LONG_TERM_BUY_PCTL,
        "200-day MA adjusted by the ticker's 5th-25th percentile distance from it over the last 5 years",
        zone_confluence(buy_low, buy_high, atr, anchors, swings, "buy"),
    )
    sell_zone = _zone_payload(
        zones, "lt_sell", close, near_band, "sell", LONG_TERM_SELL_PCTL,
        "200-day MA adjusted by the ticker's 85th-97th percentile distance from it over the last 5 years",
        zone_confluence(sell_low, sell_high, atr, anchors, swings, "sell"),
    )
    buy_study = forward_return_study(ohlc, zones, "buy")
    sell_study = forward_return_study(ohlc, zones, "sell")
    if sell_zone["position"] in {"INSIDE", "ABOVE"}:
        action = "TRIM_ZONE"
    elif buy_zone["position"] == "INSIDE":
        action = "ACCUMULATE"
    elif buy_zone["position"] == "BELOW":
        action = "DEEP_VALUE_TREND_RISK"
    elif buy_zone["position"] == "NEAR":
        action = "APPROACHING_VALUE"
    elif sell_zone["position"] == "NEAR":
        action = "APPROACHING_TRIM"
    else:
        action = "HOLD_WAIT"
    label, tone, definition = LONG_TERM_ACTIONS[action]
    ma200 = zones["ma200"]
    view = {
        "available": True,
        "horizon_sessions": LONG_TERM_HORIZON,
        "action": action,
        "action_label": label,
        "tone": tone,
        "action_definition": definition,
        "buy_zone": buy_zone,
        "sell_zone": sell_zone,
        "distance_from_ma200": {
            "current_pct": clean_float(close / float(last["ma200"]) - 1),
            "percentile": clean_float(percentile_rank(zones["lt_dev"], LONG_TERM_LOOKBACK)),
            "ma200_rising": bool(ma200.iloc[-1] > ma200.iloc[-21]),
        },
        "history": {"buy_zone": _trim_study(buy_study), "sell_zone": _trim_study(sell_study)},
    }
    return view, buy_study, sell_study


def _chart_data(ohlc: pd.DataFrame, features: pd.DataFrame, zones: pd.DataFrame, studies: dict[str, dict | None]) -> dict:
    window = ohlc.tail(CHART_PERIODS)
    first_index = len(ohlc) - len(window)
    points = []
    for position, index in enumerate(window.index):
        row, zone, feature = window.loc[index], zones.loc[index], features.loc[index]
        point = {
            "date": str(index.date()),
            "open": clean_float(row["open"], 2),
            "high": clean_float(row["high"], 2),
            "low": clean_float(row["low"], 2),
            "close": clean_float(row["close"], 2),
            "volume": clean_float(row["volume"], 0),
            "ema20": clean_float(zone["ema20"], 2),
            "ma50": clean_float(zone["ma50"], 2),
            "ma200": clean_float(zone["ma200"], 2),
            "rsi_14": clean_float(feature.get("rsi_14"), 2),
            "macd": clean_float(feature.get("macd")),
            "macd_signal": clean_float(feature.get("macd_signal")),
            "macd_hist": clean_float(feature.get("macd_hist")),
            "stretch": clean_float(zone["st_dev"], 3),
        }
        for key in ZONE_KEYS:
            for edge in ("low", "high"):
                point[f"{key}_{edge}"] = clean_float(zone[f"{key}_{edge}"], 2)
        for level in (*SHORT_TERM_BUY_PCTL, *SHORT_TERM_SELL_PCTL):
            name = f"st_q{int(level * 100):02d}"
            point[f"stretch_p{int(level * 100):02d}"] = clean_float(zone[f"{name}_applied"], 3)
        points.append(point)
    events = []
    for key, study in studies.items():
        if not study:
            continue
        for episode in study["recent_episodes"]:
            if episode["index"] >= first_index:
                events.append(
                    {
                        "date": episode["date"],
                        "zone": key,
                        "entry": clean_float(episode["entry"], 2),
                        "outcome": episode["outcome"],
                    }
                )
    return {
        "interval": "1d",
        "price_basis": "adjusted_ohlc",
        "display_periods": len(points),
        "points": points,
        "zone_touches": sorted(events, key=lambda event: event["date"]),
    }


def generate_zone_analysis(
    ticker: str,
    start_date: str = "2012-01-01",
    end_date: str | None = None,
    benchmark_primary: str = "QQQ",
    benchmark_secondary: str = "SPY",
    treasury_yield_ticker: str = "^TNX",
    oil_ticker: str = "CL=F",
) -> dict:
    bundle = load_market_data(
        ticker, start_date, end_date, benchmark_primary, benchmark_secondary, treasury_yield_ticker, oil_ticker,
    )
    features = build_features(
        bundle.ticker_data,
        bundle.benchmark_primary_data,
        bundle.benchmark_secondary_data,
        bundle.vix_data,
        bundle.treasury_yield_data,
        bundle.oil_data,
    ).dropna(subset=["close"])
    return build_zone_report(
        features,
        ticker=bundle.ticker,
        market_close=bundle.benchmark_secondary_data["Adj Close"] if not bundle.benchmark_secondary_data.empty else None,
        benchmarks={
            "equity_primary": bundle.benchmark_primary.upper(),
            "equity_secondary": bundle.benchmark_secondary.upper(),
            "treasury_yield": bundle.treasury_yield_ticker.upper(),
            "oil": bundle.oil_ticker.upper(),
        },
        data_warnings=bundle.warnings,
    )


def build_zone_report(
    features: pd.DataFrame,
    ticker: str,
    market_close: pd.Series | None = None,
    benchmarks: dict | None = None,
    data_warnings: list[str] | None = None,
) -> dict:
    benchmarks = benchmarks or {"equity_primary": "QQQ", "equity_secondary": "SPY", "treasury_yield": "^TNX", "oil": "CL=F"}
    ohlc = adjusted_ohlcv(features)
    zones = compute_zone_frame(ohlc)
    last = zones.iloc[-1]
    if not np.isfinite(last[["next_st_buy_low", "next_st_buy_high", "atr", "ema20"]].astype(float)).all():
        raise MarketDataUnavailableError(
            f"{ticker.upper()} needs at least about one year of daily history to calibrate its zones."
        )
    swings = swing_levels(ohlc)
    short_term, st_buy_study, st_sell_study = _short_term_view(ohlc, features, zones, market_close, swings)
    long_term, lt_buy_study, lt_sell_study = _long_term_view(ohlc, zones, swings)
    latest = features.iloc[-1]
    close = float(ohlc["close"].iloc[-1])
    previous_close = float(ohlc["close"].iloc[-2])
    ma200 = float(last["ma200"]) if np.isfinite(last["ma200"]) else None
    return {
        "schema_version": "3.0",
        "analysis_type": "price_zones",
        "ticker": ticker.upper(),
        "as_of_date": str(features.index[-1].date()),
        "benchmarks": benchmarks,
        "data_quality": {"warnings": data_warnings or [], "history_sessions": len(features), "history_start": str(features.index[0].date())},
        "current_market": {
            "current_price": clean_float(close, 2),
            "previous_close": clean_float(previous_close, 2),
            "change_pct": clean_float(close / previous_close - 1),
            "trend": (
                "UNKNOWN" if ma200 is None
                else "UPTREND" if close > ma200 and last["ma50"] > ma200
                else "DOWNTREND" if close < ma200 and last["ma50"] < ma200
                else "MIXED"
            ),
        },
        "short_term": short_term,
        "long_term": long_term,
        "trend_outlook": {key: trend_outlook(ohlc, zones, features, horizon) for key, horizon in TREND_HORIZONS.items()},
        "technical_indicators": {
            "ema20": clean_float(last["ema20"], 2),
            "ma50": clean_float(last["ma50"], 2),
            "ma200": clean_float(ma200, 2),
            "rsi_14": short_term["bounce_readiness"]["rsi_14"],
            "rsi_2": short_term["bounce_readiness"]["rsi_2"],
            "atr_14": clean_float(last["atr"], 2),
            "atr_pct": clean_float(last["atr"] / close),
            "return_20d": clean_float(latest.get("return_20d")),
            "volume_vs_20d_avg": clean_float(latest.get("volume_vs_20d_avg"), 2),
            "rs_vs_secondary_20d_change": clean_float(latest.get("rs_20d_change")),
        },
        "macro_context": build_macro_context(latest, benchmarks["treasury_yield"], benchmarks["oil"]),
        "chart_data": _chart_data(
            ohlc, features, zones,
            {"st_buy": st_buy_study, "st_sell": st_sell_study, "lt_buy": lt_buy_study, "lt_sell": lt_sell_study},
        ),
        "methodology": {
            "summary": (
                "Zones are calibrated from this ticker's own history, not predicted. Short-term zones measure how far "
                "price is stretched from its 20-day EMA in ATR units; long-term zones measure distance from the 200-day MA. "
                "Each zone is replayed against every earlier visit to report how price behaved afterwards."
            ),
            "short_term_lookback_sessions": SHORT_TERM_LOOKBACK,
            "long_term_lookback_sessions": LONG_TERM_LOOKBACK,
            "levels_apply_to": "next trading session; recompute after every daily close",
        },
        "not_financial_advice": True,
    }

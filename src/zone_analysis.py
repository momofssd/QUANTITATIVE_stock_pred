"""Assemble the price-zone report served to the dashboard.

No model is trained here. The report states where price is relative to
historically calibrated buy and sell zones, and what happened after earlier
visits to those zones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import MarketDataUnavailableError, download_intraday, load_market_data
from .entry_zones import (
    BOUNCE_ATR,
    DEFAULT_SIGNALS,
    MIN_EPISODES,
    MIN_RISK_ATR,
    MOMENTUM_EXTREME_SESSIONS,
    SHORT_TERM_BUY_PCTL,
    SHORT_TERM_HORIZON,
    SHORT_TERM_LOOKBACK,
    SHORT_TERM_SELL_PCTL,
    STOP_BUFFER_ATR,
    TARGET_R,
    ZONE_KEYS,
    SignalParams,
    adjusted_ohlcv,
    bounce_readiness,
    breakdown_state,
    breakdown_study,
    compute_zone_frame,
    momentum_signal_study,
    momentum_state,
    position_in_zone,
    reversion_study,
    swing_levels,
    zone_confluence,
    percentile_rank,
)
from .features import build_features
from .json_exporter import build_macro_context
from .ticker_profiles import profile_for
from .trend_probability import TREND_HORIZONS, trend_outlook
from .utils import clean_float


CHART_PERIODS = 2520
# Hourly bars per 4-hour candle, counted from each session's first bar (09:30 and 13:30 for US stocks).
HOURS_PER_CANDLE = 4

# Definitions are templates: {turn} is the ticker's turn length for that side.
SHORT_TERM_ACTIONS = {
    "TOP_SIGNAL": ("Sell signal: bullish momentum faded", "negative", "Price reached the take-profit zone and momentum has fallen {turn} session(s) in a row, so buying is losing force. Take profits on longs; the top fails if price closes above the interim high."),
    "TOP_FORMING": ("Bullish momentum easing near the top", "caution", "Price touched the take-profit zone and bullish momentum is easing, but there is no actionable top signal: the {turn}-session turn is not complete, or most of the move after it is already done. A close below the momentum pivot keeps the turn going."),
    "TAKE_PROFIT_ZONE": ("Take-profit zone, momentum still rising", "negative", "Price is stretched into the take-profit zone but bullish momentum is still building. Tops usually come after momentum fades: avoid new entries and trail a stop rather than guessing the high."),
    "SUPPORT_BREAKDOWN": ("Support breakdown", "negative", "Price closed below its most recent swing-low support and has not reclaimed it; breakdowns can keep falling. Do not buy until price closes back above the broken level."),
    "BOTTOM_SIGNAL": ("Buy signal: bearish momentum faded", "positive", "Price touched the buy zone and momentum has risen {turn} session(s) in a row, so the selling is losing force. The stop goes under the interim low: if that low breaks, the bottom failed."),
    "BOTTOM_FORMING": ("Bearish momentum easing", "caution", "Price is at or near the buy zone and bearish momentum is fading, but there is no actionable buy signal: the {turn}-session turn is not complete, or most of the bounce after it is already done. A close above the momentum pivot keeps the turn going."),
    "ENTRY_ZONE_STILL_FALLING": ("In buy zone, momentum still falling", "caution", "Price is in the buy zone but momentum is still falling, with no bearish turn yet. Buying now is catching a falling knife; wait for momentum to turn up."),
    "BELOW_ENTRY_ZONE": ("Below buy zone, momentum still falling", "negative", "Price has fallen through the buy zone and momentum is still falling. Wait for momentum to turn up before buying."),
    "APPROACHING_ENTRY": ("Approaching buy zone", "caution", "Price is within one ATR of the buy zone; set an alert at the top of the zone, then wait for momentum to turn."),
    "BOUNCE_UNDERWAY": ("Bounce underway", "neutral", "Price already turned up from the buy zone and is away from the interim low; chasing worsens reward/risk."),
    "WAIT_FOR_PULLBACK": ("Wait for pullback", "neutral", "Price is between zones; wait for a pullback into the buy zone and a momentum turn."),
}
SELL_ACTIONS = {"TOP_SIGNAL", "TOP_FORMING", "TAKE_PROFIT_ZONE"}

PHASE_TEXT = {
    "BEARISH_BUILDING": "bearish and still building",
    "BEARISH_FADING": "bearish but fading",
    "BULLISH_BUILDING": "bullish and still building",
    "BULLISH_FADING": "bullish but fading",
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


def _holdout_evidence(profile: dict | None) -> dict[str, dict]:
    """For sides whose settings were tuned, the untouched-holdout results in study form.

    The full-history replay includes the period the settings were chosen on, so it flatters
    them; the holdout is the honest record and is what the edge grade uses for those sides.
    """
    evidence = {}
    for side in ("buy", "sell"):
        if not profile or not profile[side]["adopted"]:
            continue
        holdout = profile[side]["candidate"]["holdout"]
        evidence[side] = {
            "edge_r": holdout["edge_r"],
            "expectancy_r": holdout["expectancy_r"],
            "baseline_expectancy_r": holdout["baseline_expectancy_r"],
            "held_rate": holdout["held_rate"],
            "baseline_held_rate": holdout["baseline_held_rate"],
            "episodes": holdout["signals"],
            "sufficient_sample": holdout["signals"] >= MIN_EPISODES,
            "basis": f"out of sample: {profile['holdout_start'][:4]} to {profile['data_through'][:4]}, after tuning on earlier years",
        }
    return evidence


def _r_edge_grade(study: dict) -> dict:
    """Grade momentum signals against the same trade taken on any day, in R per trade."""
    edge = study.get("edge_r")
    if not study.get("sufficient_sample") or edge is None:
        grade, label = "THIN_SAMPLE", "Too few past signals to judge"
    elif edge >= 0.10:
        grade, label = "EDGE", "Signals beat an average day"
    elif edge > 0:
        grade, label = "SLIGHT_EDGE", "Signals marginally beat an average day"
    else:
        grade, label = "NO_EDGE", "Signals no better than an average day"
    return {
        "grade": grade, "label": label, "edge_r": edge, "expectancy_r": study.get("expectancy_r"),
        "baseline_expectancy_r": study.get("baseline_expectancy_r"), "held_rate": study.get("held_rate"),
        "baseline_held_rate": study.get("baseline_held_rate"), "episodes": study.get("episodes"),
        "basis": study.get("basis", "full history"),
    }


def _trim_study(study: dict) -> dict:
    return {key: value for key, value in study.items() if key != "recent_episodes"}


def _signal_reasons(study: dict, kind: str) -> tuple[list[str], list[str]]:
    """Plain-language evidence from a momentum-signal replay."""
    extreme = "low" if kind == "bottom" else "high"
    if not study["sufficient_sample"]:
        return [], [f"Only {study['episodes']} completed {kind} signals in the history; statistics are thin."]
    reasons = [
        f"Past {kind} signals averaged {study['expectancy_r']:+.2f}R per trade vs {study['baseline_expectancy_r']:+.2f}R for the "
        f"same trade on any day ({study['episodes']} signals); the interim {extreme} held {study['held_rate']:.0%} of the time "
        f"vs {study['baseline_held_rate']:.0%} on any day."
    ]
    risks = [] if (study["edge_r"] or 0) > 0 else [f"{kind.capitalize()} signals have not beaten trading on an arbitrary day for this ticker."]
    return reasons, risks


def _short_term_view(
    ohlc: pd.DataFrame, features: pd.DataFrame, zones: pd.DataFrame, market_close: pd.Series | None, swings: list[dict],
    signals: SignalParams = DEFAULT_SIGNALS,
    holdout: dict[str, dict] | None = None,
) -> tuple[dict, dict[str, dict]]:
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
    momentum = momentum_state(ohlc, zones, signals)
    buy_study = reversion_study(ohlc, zones, "buy")
    sell_study = reversion_study(ohlc, zones, "sell")
    bottom_study = momentum_signal_study(ohlc, zones, "buy", signals)
    top_study = momentum_signal_study(ohlc, zones, "sell", signals)
    breakdown = breakdown_state(ohlc, zones, swings)
    broken_study = breakdown_study(ohlc, zones, sell_study["baseline_success_rate"])

    buy_position, sell_position = buy_zone["position"], sell_zone["position"]
    signal = momentum["signal"] if momentum["signal"] and momentum["signal"]["active"] else None
    phase = momentum["phase"]
    near_buy = momentum["touched_buy_zone"] or buy_position in {"INSIDE", "BELOW"}
    if signal and signal["side"] == "sell":
        action = "TOP_SIGNAL"
    elif momentum["touched_sell_zone"] and phase == "BULLISH_FADING":
        action = "TOP_FORMING"
    elif sell_position in {"INSIDE", "ABOVE"}:
        action = "TAKE_PROFIT_ZONE"
    elif breakdown["active"]:
        # A close through support overrides the bottom reading until the level is reclaimed.
        action = "SUPPORT_BREAKDOWN"
    elif signal:
        action = "BOTTOM_SIGNAL"
    elif near_buy and phase == "BEARISH_FADING":
        action = "BOTTOM_FORMING"
    elif buy_position == "INSIDE":
        action = "ENTRY_ZONE_STILL_FALLING"
    elif buy_position == "BELOW":
        action = "BELOW_ENTRY_ZONE"
    elif buy_position == "NEAR":
        action = "APPROACHING_ENTRY"
    elif momentum["touched_buy_zone"]:
        action = "BOUNCE_UNDERWAY"
    else:
        action = "WAIT_FOR_PULLBACK"
    label, tone, definition = SHORT_TERM_ACTIONS[action]
    definition = definition.format(turn=signals.turn("sell" if action in SELL_ACTIONS else "buy"))
    if action == "SUPPORT_BREAKDOWN":
        edge = _edge_grade(broken_study)
    else:
        side = "sell" if action in SELL_ACTIONS else "buy"
        # Tuned settings are graded on their holdout record, not on the history they were tuned on.
        edge = _r_edge_grade((holdout or {}).get(side) or (top_study if side == "sell" else bottom_study))
    # A setup this ticker's own history does not reward is never shown as a green light.
    if tone == "positive" and edge["grade"] not in {"EDGE", "SLIGHT_EDGE"}:
        tone = "caution"

    pivot = momentum["pivot_price"]
    if action == "BOTTOM_SIGNAL":
        reference, reference_basis = close, f"current close; the bottom signal fired on {signal['date']} at {signal['entry']:.2f}"
        stop, stop_basis = signal["stop"], f"{STOP_BUFFER_ATR:g} ATR under the interim low (set at the signal): below it the bottom failed"
        target_1, target_1_basis = signal["target"], f"{TARGET_R:g}x the risk from the signal close, the replay's target"
    elif action == "SUPPORT_BREAKDOWN":
        reference, reference_basis = breakdown["broken_level"], "close back above the broken support (failed breakdown)"
        stop, stop_basis = reference - BOUNCE_ATR * atr, f"{BOUNCE_ATR:g} ATR below the reclaim level"
        target_1, target_1_basis = None, None
    elif near_buy and phase.startswith("BEARISH") and pivot is not None:
        remaining = max(signals.buy_turn - momentum["rising_sessions"], 1)
        turns = f"{remaining} more rising session(s) complete the {signals.buy_turn}-session turn"
        reference = max(pivot, close) if phase == "BEARISH_FADING" else pivot
        reference_basis = f"estimated close that turns momentum up next session; {turns}"
        stop = min(momentum["interim_low"] - STOP_BUFFER_ATR * atr, reference - MIN_RISK_ATR * atr)
        stop_basis = f"{STOP_BUFFER_ATR:g} ATR under the interim low ({momentum['interim_low']:.2f}); moves lower if price makes a new low first"
        target_1, target_1_basis = None, None
    else:
        reference, reference_basis = buy_high, "top of the buy zone, then wait for momentum to turn up before buying"
        stop = buy_low - STOP_BUFFER_ATR * atr
        stop_basis = "estimate under the buy zone; once the signal fires the stop goes under the interim low"
        target_1, target_1_basis = None, None
    risk = max(reference - stop, 0.01)
    if target_1 is None:
        target_1 = reference + (BOUNCE_ATR * atr if action == "SUPPORT_BREAKDOWN" else TARGET_R * risk)
        target_1_basis = f"{BOUNCE_ATR:g} ATR rebound" if action == "SUPPORT_BREAKDOWN" else f"{TARGET_R:g}x the risk, the replay's target"
    target_2 = max(sell_low, target_1)
    trade_plan = {
        "entry_reference": clean_float(reference, 2),
        "entry_reference_basis": reference_basis,
        "stop_loss": clean_float(stop, 2),
        "stop_basis": stop_basis,
        "target_1": clean_float(target_1, 2),
        "target_1_basis": target_1_basis,
        "target_2": clean_float(target_2, 2),
        "target_2_basis": "bottom of the short-term take-profit zone (full swing)",
        "mean_reversion_level": clean_float(last["ema20"], 2),
        "reward_risk_target_1": clean_float((target_1 - reference) / risk, 2),
        "reward_risk_target_2": clean_float((target_2 - reference) / risk, 2),
    }

    reasons, risks = [], []
    faded = momentum["faded_from_extreme"] or 0
    reasons.append(
        f"Momentum is {momentum['value_atr']:+.2f} ATR, {PHASE_TEXT[phase]}"
        + (f"; {faded:.0%} of its {MOMENTUM_EXTREME_SESSIONS}-session extreme has worn off." if faded > 0 else ".")
    )
    if action in SELL_ACTIONS:
        signal_reasons, signal_risks = _signal_reasons(top_study, "top")
        reasons += signal_reasons
        risks += signal_risks
        if sell_study["sufficient_sample"]:
            reasons.append(
                f"{sell_study['success_rate']:.0%} of {sell_study['episodes']} earlier take-profit-zone touches pulled back "
                f"{BOUNCE_ATR:g} ATR before extending {BOUNCE_ATR:g} ATR higher (baseline {sell_study['baseline_success_rate']:.0%})."
            )
        if phase == "BULLISH_BUILDING":
            risks.append("Bullish momentum is still building; selling into it often leaves the last leg on the table.")
    elif action == "SUPPORT_BREAKDOWN":
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
    else:
        signal_reasons, signal_risks = _signal_reasons(bottom_study, "bottom")
        reasons += signal_reasons
        risks += signal_risks
        if buy_study["sufficient_sample"]:
            reasons.append(
                f"Buying at the first touch instead: {buy_study['success_rate']:.0%} of {buy_study['episodes']} touches bounced "
                f"{BOUNCE_ATR:g} ATR before dropping {BOUNCE_ATR:g} ATR (any-day baseline {buy_study['baseline_success_rate']:.0%})."
            )
        if near_buy:
            reasons += [f"{check['label']}: {check['detail']}." for check in readiness["checks"] if check["passed"] and check["key"] != "momentum_fading"]
            if phase == "BEARISH_BUILDING":
                risks.insert(0, "Bearish momentum is still building; the low is probably not in yet.")
    if close < float(last["ma200"]):
        risks.append("Price is below its 200-day average; bounces in downtrends fail more often.")
    market_check = next(check for check in readiness["checks"] if check["key"] == "market_backdrop")
    if not market_check["passed"]:
        risks.append(market_check["detail"] + ".")

    view = {
        "horizon_sessions": SHORT_TERM_HORIZON,
        "action": action,
        "action_label": label,
        "tone": tone,
        "action_definition": definition,
        "is_entry_now": action == "BOTTOM_SIGNAL",
        "is_exit_now": action == "TOP_SIGNAL",
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
        "momentum": momentum,
        "bounce_readiness": readiness,
        "breakdown": breakdown,
        "trade_plan": trade_plan,
        "history": {
            "buy_zone": _trim_study(buy_study),
            "momentum_bottom": _trim_study(bottom_study),
            "sell_zone": _trim_study(sell_study),
            "momentum_top": _trim_study(top_study),
            "breakdown": _trim_study(broken_study),
        },
        "reasons": reasons[:6],
        "risks": risks[:5],
    }
    return view, {"st_buy": buy_study, "st_sell": sell_study, "mom_buy": bottom_study, "mom_sell": top_study}


def _four_hour_bars(intraday: pd.DataFrame | None, features: pd.DataFrame) -> list[dict]:
    """Group hourly bars into 4-hour candles per session, on the daily chart's dividend-adjusted basis.

    Yahoo's intraday prices are not dividend-adjusted, so each candle is scaled by its
    day's adjusted/raw close ratio; otherwise the daily zones would sit off the candles.
    """
    if intraday is None or intraday.empty:
        return []
    day = intraday.index.normalize()
    bars = intraday[day <= features.index[-1]].assign(start=lambda frame: frame.index, day=lambda frame: frame.index.normalize())
    bars["slot"] = bars.groupby("day").cumcount() // HOURS_PER_CANDLE
    candles = bars.groupby(["day", "slot"]).agg(
        start=("start", "first"), open=("Open", "first"), high=("High", "max"), low=("Low", "min"),
        close=("Close", "last"), volume=("Volume", "sum"),
    )
    factor = (features["close"] / features["Close"].replace(0, np.nan)).reindex(candles.index.get_level_values("day"))
    factor = factor.ffill().fillna(1.0).to_numpy()
    return [
        {
            "date": str(row.start.date()),
            "time": row.start.strftime("%H:%M"),
            "open": clean_float(row.open * scale, 2),
            "high": clean_float(row.high * scale, 2),
            "low": clean_float(row.low * scale, 2),
            "close": clean_float(row.close * scale, 2),
            "volume": clean_float(row.volume, 0),
        }
        for row, scale in zip(candles.itertuples(), factor)
    ]


def _chart_data(
    ohlc: pd.DataFrame,
    features: pd.DataFrame,
    zones: pd.DataFrame,
    studies: dict[str, dict],
    intraday: pd.DataFrame | None = None,
) -> dict:
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
            "momentum": clean_float(zone["momentum"], 3),
            "momentum_change": clean_float(zone["momentum_change"], 3),
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
        "bars_4h": _four_hour_bars(intraday, features),
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
    warnings = list(bundle.warnings or [])
    intraday = None
    # Yahoo only serves recent hourly bars, so a historical snapshot has no 4-hour chart.
    if end_date is None:
        try:
            intraday = download_intraday(bundle.ticker)
        except MarketDataUnavailableError as exc:
            warnings.append(f"4-hour candles unavailable: {exc}")
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
        data_warnings=warnings,
        intraday=intraday,
    )


def _signal_profile(ticker: str, signals: SignalParams, profile: dict | None) -> dict:
    """Which signal settings this report used, and the validation behind a per-ticker profile."""
    if profile is None:
        return {
            "source": "default",
            "params": signals.to_dict(),
            "summary": f"No tuned profile for {ticker.upper()}; using the default settings chosen across 29 tickers.",
        }
    sides = {}
    for side in ("buy", "sell"):
        report = profile[side]
        used = report["candidate"] if report["adopted"] else report["default"]
        sides[side] = {
            "adopted": report["adopted"],
            "reason": report["reason"],
            "rank_correlation": report["rank_correlation"],
            "settings": report["settings"],
            "best_candidate": report["best_candidate"],
            "used": {"train": used["train"], "holdout": used["holdout"]},
            "default": {"train": report["default"]["train"], "holdout": report["default"]["holdout"]},
            "candidate": {"train": report["candidate"]["train"], "holdout": report["candidate"]["holdout"]},
        }
    tuned = [label for side, label in (("buy", "bottom"), ("sell", "top")) if sides[side]["adopted"]]
    return {
        "source": "ticker",
        "params": signals.to_dict(),
        "tuned": bool(tuned),
        "trained_on": profile["trained_on"],
        "data_start": profile["data_start"],
        "data_through": profile["data_through"],
        "holdout_start": profile["holdout_start"],
        "summary": (
            f"{ticker.upper()} uses its own tuned {' and '.join(tuned)} signal settings."
            if tuned
            else f"{ticker.upper()} was tuned, but no setting beat the defaults reliably, so it keeps the default settings."
        ),
        **sides,
    }


def build_zone_report(
    features: pd.DataFrame,
    ticker: str,
    market_close: pd.Series | None = None,
    benchmarks: dict | None = None,
    data_warnings: list[str] | None = None,
    intraday: pd.DataFrame | None = None,
    signals: SignalParams | None = None,
) -> dict:
    """Build the report. Without explicit `signals`, a ticker with a stored profile uses its own settings."""
    benchmarks = benchmarks or {"equity_primary": "QQQ", "equity_secondary": "SPY", "treasury_yield": "^TNX", "oil": "CL=F"}
    profile = None
    if signals is None:
        signals, profile = profile_for(ticker)
    ohlc = adjusted_ohlcv(features)
    zones = compute_zone_frame(ohlc, signals)
    last = zones.iloc[-1]
    if not np.isfinite(last[["next_st_buy_low", "next_st_buy_high", "atr", "ema20"]].astype(float)).all():
        raise MarketDataUnavailableError(
            f"{ticker.upper()} needs at least about one year of daily history to calibrate its zones."
        )
    swings = swing_levels(ohlc)
    short_term, studies = _short_term_view(ohlc, features, zones, market_close, swings, signals, _holdout_evidence(profile))
    latest = features.iloc[-1]
    close = float(ohlc["close"].iloc[-1])
    previous_close = float(ohlc["close"].iloc[-2])
    ma200 = float(last["ma200"]) if np.isfinite(last["ma200"]) else None
    return {
        "schema_version": "4.0",
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
        "signal_profile": _signal_profile(ticker, signals, profile),
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
            studies,
            intraday,
        ),
        "methodology": {
            "summary": (
                "Zones are calibrated from this ticker's own history, not predicted. Short-term zones measure how far "
                f"price is stretched from its 20-day EMA in ATR units; momentum, (EMA{signals.fast} - EMA{signals.slow}) / ATR, measures how fast it got "
                "there, and buy or sell signals wait for that momentum to fade. "
                "Each zone is replayed against every earlier visit to report how price behaved afterwards."
            ),
            "short_term_lookback_sessions": SHORT_TERM_LOOKBACK,
            "levels_apply_to": "next trading session; recompute after every daily close",
        },
        "not_financial_advice": True,
    }

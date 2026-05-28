from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .decision_engine import DecisionResult, SIGNAL_DEFINITIONS
from .utils import OUTPUT_DIR, clean_float


def _build_chart_data(features: pd.DataFrame, periods: int = 180) -> dict:
    chart_frame = features.tail(periods).copy()
    points: list[dict] = []
    for index, row in chart_frame.iterrows():
        raw_close = clean_float(row.get("Close"))
        adjusted_close = clean_float(row.get("close"))
        adjustment_factor = (
            float(row.get("close")) / float(row.get("Close"))
            if raw_close not in {None, 0} and adjusted_close is not None
            else 1.0
        )
        def adjusted_price(column: str) -> float | None:
            raw_price = row.get(column)
            return clean_float(raw_price * adjustment_factor, 2) if clean_float(raw_price) is not None else adjusted_close

        points.append(
            {
                "date": str(index.date()),
                "open": adjusted_price("Open"),
                "high": adjusted_price("High"),
                "low": adjusted_price("Low"),
                "close": clean_float(row.get("close"), 2),
                "volume": clean_float(row.get("Volume"), 0),
                "ma20": clean_float(row.get("ma20"), 2),
                "ma50": clean_float(row.get("ma50"), 2),
                "ma200": clean_float(row.get("ma200"), 2),
                "rsi_14": clean_float(row.get("rsi_14"), 2),
                "macd": clean_float(row.get("macd")),
                "macd_signal": clean_float(row.get("macd_signal")),
                "macd_hist": clean_float(row.get("macd_hist")),
            }
        )
    return {
        "interval": "1d",
        "price_basis": "adjusted_ohlc",
        "display_periods": len(points),
        "points": points,
    }


def build_signal_json(
    ticker: str,
    as_of_date: str,
    benchmark_primary: str,
    benchmark_secondary: str,
    treasury_yield_ticker: str,
    oil_ticker: str,
    features: pd.DataFrame,
    regime: str,
    probabilities: dict[str, float],
    decision_result: DecisionResult,
    model_metadata: dict,
    five_day_forecast: dict | None = None,
    short_term_trade: dict | None = None,
    data_warnings: list[str] | None = None,
) -> dict:
    latest = features.iloc[-1]
    missing_fields = [col for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if col not in features.columns]
    vix_level = clean_float(latest.get("vix_level"), 2)
    vix_5d_change = clean_float(latest.get("vix_5d_change"))
    vix_change = clean_float(latest.get("vix_20d_change"))
    vix_60d_change = clean_float(latest.get("vix_60d_change"))
    vix_20d_slope = clean_float(latest.get("vix_20d_slope_pct"))
    yield_level = clean_float(latest.get("ten_year_yield"), 3)
    yield_5d_change = clean_float(latest.get("ten_year_yield_5d_change"), 3)
    yield_20d_change = clean_float(latest.get("ten_year_yield_20d_change"), 3)
    yield_60d_change = clean_float(latest.get("ten_year_yield_60d_change"), 3)
    yield_20d_slope = clean_float(latest.get("ten_year_yield_20d_slope"), 4)
    oil_price = clean_float(latest.get("oil_price"), 2)
    oil_5d_change = clean_float(latest.get("oil_5d_change"))
    oil_20d_change = clean_float(latest.get("oil_20d_change"))
    oil_60d_change = clean_float(latest.get("oil_60d_change"))
    oil_20d_slope = clean_float(latest.get("oil_20d_slope_pct"))
    macro_risk = "HIGH" if (
        (vix_level is not None and vix_level > 25)
        or (vix_change is not None and vix_change > 0.25)
        or (yield_60d_change is not None and yield_60d_change > 0.35)
        or (oil_20d_change is not None and oil_20d_change > 0.20)
    ) else "MEDIUM"
    if (
        vix_level is not None
        and vix_level < 18
        and (vix_change is None or vix_change < 0.10)
        and (yield_60d_change is None or yield_60d_change < 0.20)
        and (oil_20d_change is None or oil_20d_change < 0.15)
    ):
        macro_risk = "LOW"

    signal = {
        "schema_version": "2.2",
        "ticker": ticker.upper(),
        "as_of_date": as_of_date,
        "timeframe": "long_term_swing",
        "data_quality": {
            "price_source": "yahoo_chart_with_yfinance_fallback",
            "macro_source": "yahoo_chart_with_yfinance_fallback_vix_treasury_yield_oil",
            "missing_fields": missing_fields,
            "warnings": data_warnings or [],
        },
        "benchmarks": {
            "equity_primary": benchmark_primary.upper(),
            "equity_secondary": benchmark_secondary.upper(),
            "treasury_yield": treasury_yield_ticker.upper(),
            "oil": oil_ticker.upper(),
        },
        "current_market": {
            "current_price": clean_float(latest.get("close"), 2),
            "regime": regime,
            "model_probabilities": {
                "bull_trend": clean_float(probabilities.get("bullish")),
                "sideways": clean_float(probabilities.get("sideways")),
                "bear_trend": clean_float(probabilities.get("bearish")),
                "breakout_setup": clean_float(max(probabilities.get("bullish", 0) - 0.25, 0)),
                "trend_deterioration": clean_float(max(probabilities.get("bearish", 0) - 0.15, 0)),
            },
        },
        "decision": decision_result.decision,
        "available_signals": SIGNAL_DEFINITIONS,
        "price_zones": decision_result.price_zones,
        "five_day_forecast": five_day_forecast or {},
        "short_term_trade": short_term_trade or {},
        "near_term_forecast": decision_result.near_term_forecast,
        "chart_data": _build_chart_data(features),
        "technical_indicators": {
            "close": clean_float(latest.get("close"), 2),
            "ma20": clean_float(latest.get("ma20"), 2),
            "ma50": clean_float(latest.get("ma50"), 2),
            "ma200": clean_float(latest.get("ma200"), 2),
            "rsi_14": clean_float(latest.get("rsi_14"), 2),
            "adx_14": clean_float(latest.get("adx_14"), 2),
            "atr_14": clean_float(latest.get("atr_14"), 2),
            "atr_pct": clean_float(latest.get("atr_pct")),
            "bollinger_band_width": clean_float(latest.get("bollinger_band_width")),
            "return_20d": clean_float(latest.get("return_20d")),
            "return_60d": clean_float(latest.get("return_60d")),
            "volume_vs_20d_avg": clean_float(latest.get("volume_vs_20d_avg"), 2),
            "distance_to_50ma_pct": clean_float(latest.get("close_to_50ma_pct")),
            "distance_to_200ma_pct": clean_float(latest.get("close_to_200ma_pct")),
        },
        "relative_strength": {
            "benchmark_primary": benchmark_primary.upper(),
            "benchmark_secondary": benchmark_secondary.upper(),
            "target_vs_primary_20d_change": clean_float(latest.get("rs_primary_20d_change")),
            "target_vs_spy_20d_change": clean_float(latest.get("rs_20d_change")),
            "rs_status": "IMPROVING" if (latest.get("rs_20d_change", 0) or 0) > 0 else "WEAKENING",
        },
        "macro_context": {
            "vix_level": vix_level,
            "vix_5d_change": vix_5d_change,
            "vix_20d_change": vix_change,
            "vix_60d_change": vix_60d_change,
            "vix_20d_slope_per_session": vix_20d_slope,
            "vix_change_units": "decimal_return",
            "treasury_yield_benchmark": treasury_yield_ticker.upper(),
            "ten_year_yield": yield_level,
            "ten_year_yield_5d_change": yield_5d_change,
            "ten_year_yield_20d_change": yield_20d_change,
            "ten_year_yield_60d_change": yield_60d_change,
            "ten_year_yield_20d_slope_per_session": yield_20d_slope,
            "ten_year_yield_change_units": "percentage_points",
            "oil_benchmark": oil_ticker.upper(),
            "oil_price": oil_price,
            "oil_5d_change": oil_5d_change,
            "oil_20d_change": oil_20d_change,
            "oil_60d_change": oil_60d_change,
            "oil_20d_slope_per_session": oil_20d_slope,
            "oil_change_units": "decimal_return",
            "macro_risk_level": macro_risk,
        },
        "risk_management": decision_result.risk_management,
        "reasoning": {
            "technical_score": decision_result.technical_score,
            "key_reasons": decision_result.reasons,
            "main_risks": decision_result.risks,
        },
        "model_metadata": {
            **model_metadata,
            "last_trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "feature_count": model_metadata.get("feature_count"),
        },
    }
    return signal


def save_signal_outputs(signal: dict, output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latest_signal.json"
    md_path = output_dir / "latest_signal.md"
    json_path.write_text(json.dumps(signal, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(signal), encoding="utf-8")
    return json_path, md_path


def _to_markdown(signal: dict) -> str:
    decision = signal["decision"]
    zones = signal["price_zones"]
    current = signal["current_market"]
    five_day = signal.get("five_day_forecast") or {}
    short_term_trade = signal.get("short_term_trade") or {}
    available_signals = signal.get("available_signals") or {}
    lines = [
            f"# {signal['ticker']} Long-Term Signal",
            "",
            f"- As of: {signal['as_of_date']}",
            f"- Current price: {current['current_price']}",
            f"- Regime: {current['regime']}",
            f"- Final signal: {decision['final_signal']}",
            f"- Confidence: {decision['confidence_label']} / {decision['confidence_score']}",
            f"- Preferred buy zone: {zones['preferred_long_term_buy_zone']['low']} - {zones['preferred_long_term_buy_zone']['high']}",
            f"- Short-term tactical buy zone: {zones['short_term_tactical_buy_zone']['low']} - {zones['short_term_tactical_buy_zone']['high']}",
            f"- Breakout trigger: {zones['breakout_buy_zone']['trigger_price']}",
            f"- Trim zone: {zones['trim_or_sell_zone']['low']} - {zones['trim_or_sell_zone']['high']}",
            f"- Invalidation: {zones['invalidation_zone']['price']}",
            "",
    ]
    if five_day:
        lines.extend(
            [
                "## Five-Trading-Day Forecast",
                "",
                f"- Direction: {five_day['predicted_direction']}",
                f"- Confidence: {five_day['confidence_label']} / {five_day['confidence_score']}",
                f"- Direction-conditioned possible close: {five_day['possible_price_in_5_trading_days']}",
                (
                    "- Conditional regression error-based range: "
                    f"{five_day['possible_price_range']['low']} - {five_day['possible_price_range']['high']}"
                ),
                f"- Direction/price model agreement: {five_day['direction_price_models_agree']}",
                f"- Interpretation: {five_day['interpretation_note']}",
                "",
            ]
        )
    if short_term_trade:
        management = short_term_trade["trade_management"]
        pullback = short_term_trade["entry_points"]["pullback_entry"]
        breakout = short_term_trade["entry_points"]["breakout_entry"]
        lines.extend(
            [
                "## Short-Term Trade Plan",
                "",
                f"- Action: {short_term_trade['action']}",
                f"- Good short-term buy now: {short_term_trade['is_good_short_term_buy_now']}",
                f"- Confidence: {short_term_trade['confidence_label']} / {short_term_trade['confidence_score']}",
                f"- Pullback entry: {pullback['low']} - {pullback['high']} ({pullback['status']})",
                f"- Breakout entry trigger: {breakout['trigger_price']} ({breakout['status']})",
                f"- Five-session possible close: {management['possible_close_in_5_trading_days']}",
                f"- Summary: {short_term_trade['summary']}",
                "",
            ]
        )
        if management["active_for_recommended_setup"]:
            lines[-2:-2] = [
                f"- Stop loss: {management['stop_loss_price']}",
                f"- Estimated reward/risk: {management['estimated_reward_to_risk']}",
            ]
    if available_signals:
        lines.extend(["## Available Signals", ""])
        lines.extend(f"- {name}: {definition}" for name, definition in available_signals.items())
        lines.append("")
    lines.extend(
        [
            decision["summary"],
            "",
            "This is quantitative decision support, not financial advice.",
        ]
    )
    return "\n".join(lines)

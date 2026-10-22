"""Per-ticker momentum-signal profiles: tuned, validated out of sample, and stored for the dashboard.

A profile replaces the default signal settings (momentum speeds, turn length, zone-touch
window) for one ticker. With ~100 signals per ticker, picking the best of many settings on
the same data mostly finds noise, so tuning is guarded:

1. Settings are scored only on the training period (before HOLDOUT_START), as the average
   edge over three equal sub-periods, and must be positive in at least two of them.
2. The best candidate must beat the default's training score by MIN_IMPROVEMENT_R.
3. It must also do at least as well as the default on the untouched holdout period.
4. Across every tested setting, the training score must rank the holdout results in the same
   order (Spearman correlation of at least MIN_RANK_CORRELATION). A pick can beat the default
   on the holdout by luck; this checks that tuning itself carries information for the ticker.

Otherwise the ticker keeps the default settings. The bottom (buy) side chooses the momentum
speeds; the top (sell) side then tunes its own turn length and touch window on those speeds.

Train and save:  python -m src.ticker_profiles QQQ VOO SMH MU TSLA --end 2026-10-15
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data_loader import download_ohlcv
from .entry_zones import (
    DEFAULT_SIGNALS,
    SignalParams,
    adjusted_ohlcv,
    baseline_trades,
    compute_zone_frame,
    signal_trades,
)
from .utils import PROJECT_ROOT


PROFILE_PATH = PROJECT_ROOT / "models" / "ticker_profiles.json"
DATA_START = "2012-01-01"
HOLDOUT_START = "2021-01-01"
TRAIN_BLOCKS = 3
MIN_TRAIN_SIGNALS = 20
MIN_HOLDOUT_SIGNALS = 10
MIN_IMPROVEMENT_R = 0.05
MIN_RANK_CORRELATION = 0.3
SPANS = ((3, 10), (5, 13), (5, 20), (8, 21))
TURNS = (1, 2, 3)
TOUCHES = (3, 5, 8)


def load_profiles(path=PROFILE_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def profile_for(ticker: str, path=PROFILE_PATH) -> tuple[SignalParams, dict | None]:
    """Signal settings for a ticker: its stored profile when one exists, otherwise the defaults."""
    profile = load_profiles(path).get(ticker.upper())
    if not profile:
        return DEFAULT_SIGNALS, None
    return SignalParams(**profile["params"]), profile


def snapshot_ohlc(ticker: str, end: str, start: str = DATA_START) -> pd.DataFrame:
    """Adjusted OHLCV for a fixed date range (cached after the first download), so training is reproducible."""
    raw = download_ohlcv(ticker, start, end).dropna(subset=["Close"])
    raw["close"] = raw["Adj Close"]
    return adjusted_ohlcv(raw)


def _period_metrics(trades: list[dict], baseline: pd.DataFrame, start: str, end: str) -> dict:
    """Signal results and the any-day baseline restricted to entries in [start, end)."""
    rows = [trade for trade in trades if trade["complete"] and start <= trade["date"] < end]
    window = baseline[(baseline.index >= start) & (baseline.index < end)]
    r = [trade["r_multiple"] for trade in rows]
    expectancy = float(np.mean(r)) if r else None
    base = float(window["r_multiple"].mean()) if len(window) else None
    return {
        "signals": len(rows),
        "expectancy_r": None if expectancy is None else round(expectancy, 4),
        "baseline_expectancy_r": None if base is None else round(base, 4),
        "edge_r": None if expectancy is None or base is None else round(expectancy - base, 4),
        "held_rate": round(float(np.mean([trade["held"] for trade in rows])), 4) if rows else None,
        "baseline_held_rate": round(float(window["held"].mean()), 4) if len(window) else None,
    }


def _train_blocks(index: pd.DatetimeIndex) -> list[tuple[str, str]]:
    """Split the training period into TRAIN_BLOCKS equal spans of calendar time."""
    first = index[index < HOLDOUT_START][0]
    edges = pd.date_range(first, pd.Timestamp(HOLDOUT_START), periods=TRAIN_BLOCKS + 1)
    return [(str(edges[k].date()), str(edges[k + 1].date())) for k in range(TRAIN_BLOCKS)]


def evaluate(ohlc: pd.DataFrame, signals: SignalParams, side: str, baseline: pd.DataFrame, zones: pd.DataFrame | None = None) -> dict:
    """Training, per-block, and holdout metrics for one setting on one side."""
    zones = compute_zone_frame(ohlc, signals) if zones is None else zones
    trades = signal_trades(ohlc, zones, side, signals)
    end = str((ohlc.index[-1] + pd.Timedelta(days=1)).date())
    blocks = [_period_metrics(trades, baseline, start, stop) for start, stop in _train_blocks(ohlc.index)]
    scored = [block["edge_r"] for block in blocks if block["edge_r"] is not None and block["signals"] >= 3]
    train = _period_metrics(trades, baseline, str(ohlc.index[0].date()), HOLDOUT_START)
    eligible = train["signals"] >= MIN_TRAIN_SIGNALS and sum(edge > 0 for edge in scored) >= 2
    return {
        "train": train,
        "train_blocks": blocks,
        "train_score": round(float(np.mean(scored)), 4) if scored else None,
        "eligible": bool(eligible),
        "holdout": _period_metrics(trades, baseline, HOLDOUT_START, end),
    }


def _rank_correlation(results: dict) -> float | None:
    """Spearman correlation between training score and holdout edge across all tested settings."""
    pairs = [
        (result["train_score"], result["holdout"]["edge_r"])
        for result in results.values()
        if result["train_score"] is not None and result["holdout"]["edge_r"] is not None
    ]
    if len(pairs) < 5:
        return None
    rho = spearmanr(*zip(*pairs)).statistic
    return round(float(rho), 4) if np.isfinite(rho) else None


def _decide(candidate: dict, default: dict, rank_correlation: float | None) -> tuple[bool, str]:
    """Adopt a tuned setting only if it clearly won in training, did not lose on the holdout,
    and training rankings have predicted holdout rankings for this ticker."""
    if candidate is default:
        return False, "the default settings scored best in training"
    if (candidate["train_score"] or 0) < (default["train_score"] or 0) + MIN_IMPROVEMENT_R:
        return False, f"the best tuned setting beat the default by less than {MIN_IMPROVEMENT_R}R in training"
    tuned, base = candidate["holdout"], default["holdout"]
    if tuned["signals"] < MIN_HOLDOUT_SIGNALS or tuned["edge_r"] is None:
        return False, f"fewer than {MIN_HOLDOUT_SIGNALS} holdout signals to confirm it"
    if base["edge_r"] is not None and tuned["edge_r"] < base["edge_r"]:
        return False, "the tuned setting did worse than the default on the 2021+ holdout"
    if rank_correlation is None or rank_correlation < MIN_RANK_CORRELATION:
        shown = "n/a" if rank_correlation is None else f"{rank_correlation:+.2f}"
        return False, f"training scores did not predict holdout results across the tested settings (rank correlation {shown}), so the win is likely luck"
    return True, (
        "beat the default in training, did at least as well on the 2021+ holdout, and training scores "
        f"predicted holdout results across the tested settings (rank correlation {rank_correlation:+.2f})"
    )


def _pick(results: dict[SignalParams, dict], default: SignalParams) -> SignalParams:
    eligible = {params: result for params, result in results.items() if result["eligible"] and result["train_score"] is not None}
    if not eligible:
        return default
    # Ties go to the default so a profile never changes without a reason.
    return max(eligible, key=lambda params: (eligible[params]["train_score"], params == default))


def _side_report(side: str, chosen: SignalParams, candidate: SignalParams, results: dict, default: SignalParams, adopted: bool, reason: str) -> dict:
    keys = ("fast", "slow", f"{side}_turn", f"{side}_touch")
    return {
        "adopted": adopted,
        "reason": reason,
        "rank_correlation": _rank_correlation(results),
        "settings": {key: getattr(chosen, key) for key in keys},
        "best_candidate": {key: getattr(candidate, key) for key in keys},
        "candidate": results[candidate],
        "default": results[default],
        "settings_tested": len(results),
    }


def tune_ticker(ticker: str, ohlc: pd.DataFrame) -> dict:
    zones_by_span = {spans: compute_zone_frame(ohlc, SignalParams(*spans)) for spans in SPANS}
    reference = zones_by_span[(DEFAULT_SIGNALS.fast, DEFAULT_SIGNALS.slow)]
    baselines = {side: baseline_trades(ohlc, reference, side) for side in ("buy", "sell")}

    buy_results = {}
    for (fast, slow), turn, touch in product(SPANS, TURNS, TOUCHES):
        params = SignalParams(fast=fast, slow=slow, buy_turn=turn, buy_touch=touch)
        buy_results[params] = evaluate(ohlc, params, "buy", baselines["buy"], zones_by_span[(fast, slow)])
    buy_candidate = _pick(buy_results, DEFAULT_SIGNALS)
    buy_adopted, buy_reason = _decide(buy_results[buy_candidate], buy_results[DEFAULT_SIGNALS], _rank_correlation(buy_results))
    buy = buy_candidate if buy_adopted else DEFAULT_SIGNALS

    # The top side reuses the bottom side's momentum speeds so both read the same gauge.
    spans = (buy.fast, buy.slow)
    sell_default = SignalParams(fast=buy.fast, slow=buy.slow, buy_turn=buy.buy_turn, buy_touch=buy.buy_touch)
    sell_results = {}
    for turn, touch in product(TURNS, TOUCHES):
        params = SignalParams(fast=buy.fast, slow=buy.slow, buy_turn=buy.buy_turn, buy_touch=buy.buy_touch, sell_turn=turn, sell_touch=touch)
        sell_results[params] = evaluate(ohlc, params, "sell", baselines["sell"], zones_by_span[spans])
    sell_candidate = _pick(sell_results, sell_default)
    sell_adopted, sell_reason = _decide(sell_results[sell_candidate], sell_results[sell_default], _rank_correlation(sell_results))
    final = sell_candidate if sell_adopted else sell_default

    return {
        "ticker": ticker.upper(),
        "trained_on": str(date.today()),
        "data_start": str(ohlc.index[0].date()),
        "data_through": str(ohlc.index[-1].date()),
        "holdout_start": HOLDOUT_START,
        "params": final.to_dict(),
        "tuned": buy_adopted or sell_adopted,
        "buy": _side_report("buy", final, buy_candidate, buy_results, DEFAULT_SIGNALS, buy_adopted, buy_reason),
        "sell": _side_report("sell", final, sell_candidate, sell_results, sell_default, sell_adopted, sell_reason),
    }


def train_profiles(tickers: list[str], end: str, path=PROFILE_PATH) -> dict:
    profiles = load_profiles(path)
    for ticker in tickers:
        # The snapshot range is stored so the profile can be re-derived exactly (see tests/test_ticker_profiles.py).
        profiles[ticker.upper()] = {**tune_ticker(ticker, snapshot_ohlc(ticker, end)), "snapshot": {"start": DATA_START, "end": end}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profiles, indent=2) + "\n", encoding="utf-8")
    return profiles


def _print_summary(profile: dict) -> None:
    print(f"\n{profile['ticker']}  data {profile['data_start']} to {profile['data_through']}  params {profile['params']}")
    for side in ("buy", "sell"):
        report = profile[side]
        tuned, default = report["candidate"], report["default"]
        print(
            f"  {side:4s} {'ADOPTED' if report['adopted'] else 'default'}: {report['reason']}\n"
            f"       train->holdout rank correlation {report['rank_correlation']}\n"
            f"       best candidate {report['best_candidate']}\n"
            f"       train edge  tuned {tuned['train']['edge_r']} ({tuned['train']['signals']})  default {default['train']['edge_r']} ({default['train']['signals']})\n"
            f"       holdout edge tuned {tuned['holdout']['edge_r']} ({tuned['holdout']['signals']})  default {default['holdout']['edge_r']} ({default['holdout']['signals']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and store per-ticker momentum-signal profiles.")
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--end", required=True, help="last date of the fixed training snapshot, e.g. 2026-10-15")
    args = parser.parse_args()
    profiles = train_profiles(args.tickers, args.end)
    for ticker in args.tickers:
        _print_summary(profiles[ticker.upper()])
    print(f"\nSaved {PROFILE_PATH}")


if __name__ == "__main__":
    main()

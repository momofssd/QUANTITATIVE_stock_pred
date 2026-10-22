# Quant Entry Zone Analyzer

Python and React app that shows **where to buy and where to sell** a stock or ETF instead of predicting its price. It downloads daily OHLCV data, calibrates two short-term price zones from the ticker's own history, and replays every earlier visit to each zone so you can see how price behaved afterwards.

- **Short-term buy zone**: where price would be stretched below its 20-day EMA by as much as the ticker's 5th-20th percentile over the last 3 years, measured in ATRs. This is the V-bounce entry area.
- **Short-term take-profit zone**: the same stretch at the 80th-95th percentile above the EMA.

A zone only says price is stretched. Buy and sell signals also wait for the **momentum** that pushed price there to fade (see [Momentum signals](#momentum-signals-interim-bottoms-and-tops)).

For each zone the dashboard tells you whether the current price is inside, near, below, or away from it, and how far it would have to move. Zones are not forecasts; the only forecast is the trend outlook described below. An analysis takes a few seconds.

This is decision support, not financial advice.

## Setup

```bash
pip install -r requirements.txt
```

The repository includes `.python-version` set to Python `3.13` for consistent hosted builds. `yfinance` is pinned to the tested package version because data-download behavior is part of application runtime behavior.

The app reads `.env`. Supported Gemini key names:

```text
GEMINI_API_KEY=...
GOOGLE_API_KEY=...
gemini=...
GEMINI_MODEL=gemini-4-argon
```

Key lookup prefers `GEMINI_API_KEY`, then `gemini`, then `GOOGLE_API_KEY`, and ignores invalid-looking very short values.

The default Gemini model is `gemini-4-argon`. You can set `GEMINI_MODEL` or pass `--gemini-model` to override it.

## Web Dashboard

The React frontend shows:

- **Short-term card**: action (`TOP_SIGNAL`, `TOP_FORMING`, `TAKE_PROFIT_ZONE`, `SUPPORT_BREAKDOWN`, `BOTTOM_SIGNAL`, `BOTTOM_FORMING`, `ENTRY_ZONE_STILL_FALLING`, `BELOW_ENTRY_ZONE`, `APPROACHING_ENTRY`, `BOUNCE_UNDERWAY`, `WAIT_FOR_PULLBACK`), the latest bottom or top signal with its stop and target, current price, next-session buy and take-profit zones with distance, the interim low or high, a momentum gauge (value, phase, how much has faded, and the pivot close that turns it), a stretch-percentile gauge, and the replay of past signals (or, after a support breakdown, the historical continuation statistics)
- **Trend outlook**: bullish/bearish probability for the next 10 and 60 sessions, with the walk-forward record that decides whether the model or the base rate is trusted
- **Chart** (4H / 1D / 1W candles; 3M to 10Y ranges, with a log scale beyond one year and 3Y+ switching daily to weekly; 4-hour candles cover about the last two years and are built from Yahoo's hourly bars, adjusted to the daily price basis, with daily zones and indicators): adjusted candles with the zones drawn as they moved over time, next session's zones at the right edge, markers on earlier zone touches (▲/▼ buy/take-profit) and momentum signals (◆ bottom/top) coloured by outcome, and a momentum / stretch / RSI / MACD lower pane
- **Price ladder**: every level, highest first, with the move needed from the current price
- **Bounce checklist**: stretched, oversold, bearish momentum fading, capitulation volume, reversal candle, reclaim of the prior high, and market backdrop
- **Short-term trade plan**: entry, a stop under the interim low, and a 2R target, the same trade the signal replay grades
- **Historical replay tables**, technical and macro context, and an optional Gemini explanation

### Trend outlook

The trend panel gives the probability of a higher close 10 and 60 sessions ahead. Two forecasters compete:

- the ticker's **base rate**: how often it closed higher over that horizon in its own history
- a **logistic regression** on distance from the 200-day MA, its slope, and 60/120-session returns, tested walk-forward (re-fit every 63 sessions on data whose outcomes were already known, then scored on the next block)

The headline uses the model only when it beat the base rate out of sample (Brier skill score of at least 2% and AUC of at least 0.56). Otherwise the headline is the base rate, and the model's reading is shown for reference with its test record. When this was built, neither this model nor larger feature sets beat the base rate on SPY, QQQ, SMH, NVDA, AAPL, XLE, TLT or GLD at either horizon, so expect the base rate to be the usual headline.

### Momentum signals: interim bottoms and tops

A buy zone tells you price is stretched, not that the selling has stopped. Buying at the first touch catches a falling knife: across 29 cached tickers (2012–2026), a lower low followed within 10 sessions three times out of four. The signals therefore wait for momentum to fade.

- **Momentum** = (EMA5 − EMA13) / ATR14: how fast price has been moving, in ATRs. Below 0 is bearish, above 0 bullish. Whether it is building or fading is read from the EMA spread itself, so a quiet session cannot make a stalled decline look like it is accelerating.
- **Phase**: bearish building (still falling), bearish fading, bullish building, bullish fading. **Faded** = share of the 10-session extreme that has worn off.
- **Pivot price**: both EMAs move linearly with the close, so there is an exact next-session close above which momentum rises and below which it falls.
- **Bottom signal** (`BOTTOM_SIGNAL`): within 5 sessions of a buy-zone touch, momentum is still negative but has risen 2 sessions in a row. Entry at that close, stop 0.25 ATR under the 6-session low (at least 0.5 ATR away), target 2x the risk, exit after 10 sessions otherwise. One rising session is `BOTTOM_FORMING`; still falling is `ENTRY_ZONE_STILL_FALLING` or `BELOW_ENTRY_ZONE`.
- **Top signal** (`TOP_SIGNAL`): the mirror at the take-profit zone; bullish momentum fell 2 sessions in a row. The stop is 0.25 ATR above the 6-session high.

Every past signal is replayed as that trade and scored in **R** (profit or loss as a multiple of the amount risked) against the same trade from every session's close. The edge grade is `EDGE` at +0.10R or more, `SLIGHT_EDGE` above 0, otherwise `NO_EDGE`, and a buy signal is shown as positive only with `EDGE` or `SLIGHT_EDGE`. The replay also reports how often the interim low (high) held for 10 sessions.

How the rule was chosen: momentum speeds (3/10, 5/13, 5/20, 8/21), one to three turning sessions, a close above the prior high, a minimum trough depth, RSI(2), price/momentum divergence, and a "still falling but slower" entry were compared on 2012–2020 only, then checked on 2021–2026. Results across 29 cached tickers when this was built:

| Buy rule | Avg R 2012–2020 | Avg R 2021–2026 | Interim low held |
|---|---|---|---|
| Any day | +0.13 | +0.08 | 49–52% |
| First touch of the buy zone (previous entry) | +0.07 | −0.02 | 25–26% |
| Bottom signal | +0.22 | +0.08 | 53–62% |

The bottom signal beat the first-touch entry in 11 of 14 years and beat any day in 11 of 14 years (21 of 29 tickers), but in 2021–2026 the aggregate edge over any day was about zero. Divergence, the "slowing but still falling" entry, and requiring a close above the prior high did not hold up out of sample. Top signals beat selling at the first touch but not selling on any day (9 of 14 years, 14 of 29 tickers): calling tops in rising markets rarely pays. Read the ticker's own replay before trusting either signal.

### Per-ticker signal profiles

QQQ, VOO, SMH, MU and TSLA have their own tuned signal settings, stored in [models/ticker_profiles.json](models/ticker_profiles.json). When one of them is analyzed, the report uses its settings; any other ticker uses the defaults (EMA5/13, 2-session turn, zone touch within 5 sessions). The **Signal settings** panel shows which settings were used and why.

The tuner ([src/ticker_profiles.py](src/ticker_profiles.py)) tests 36 bottom-signal settings (EMA speeds 3/10, 5/13, 5/20, 8/21; turn length 1–3 sessions; touch window 3, 5 or 8 sessions), then 9 top-signal settings on the chosen speeds. With about 100 signals per ticker, picking the best of 36 on the same data mostly finds noise, so a setting replaces the default only if it passes all four checks:

1. It has the best training score (average edge over three 3-year blocks of 2012–2020, positive in at least two), with at least 20 training signals.
2. It beats the default's training score by at least 0.05R.
3. It does at least as well as the default on 2021–2026, which tuning never sees (at least 10 signals).
4. Across all tested settings, training scores rank the 2021–2026 results in the same order (Spearman ≥ 0.3). This rejects a ticker where the winner was lucky.

For a tuned side, the card's edge grade uses the 2021–2026 out-of-sample result, because the full-history replay includes the years the settings were chosen on. Results on the 2026-10-15 snapshot:

| Ticker | Bottom (buy) | Top (sell) | Out of sample, tuned vs default |
|---|---|---|---|
| QQQ | Tuned: EMA8/21, touch within 3 | Default | buy +0.23R vs −0.02R |
| SMH | Tuned: EMA5/20, touch within 3 | Default | buy +0.26R vs +0.16R |
| TSLA | Tuned: EMA3/10, 3-session turn, touch within 3 | Tuned: 1-session turn, touch within 3 | buy −0.08R vs −0.24R; sell −0.06R vs −0.19R |
| MU | Default (tuned winner was luck: rank correlation −0.07) | Tuned: 3-session turn | sell +0.29R vs +0.18R |
| VOO | Default (rank correlation −0.27) | Default | — |

TSLA's tuned signals lose less than the defaults but still do not beat trading on any day, and the card says so.

Retrain after new data (this overwrites the profiles and records the snapshot used):

```bash
python -m src.ticker_profiles QQQ VOO SMH MU TSLA --end 2026-10-15
```

`tests/test_ticker_profiles.py` is the regression test: it re-trains each stored profile from its recorded price snapshot and fails if any setting or validation number changes, so a code change that shifts a ticker's behaviour cannot slip through. It also checks every adopted setting obeys the four rules. The snapshots live in the gitignored `data/cache/`, so the reproduction test skips on machines that have not downloaded them.

### Reading the zone-touch statistics

A zone-touch setup "works" when price moves 1.5 ATR in the expected direction before moving 1.5 ATR against it within 10 sessions, so about 50% is break-even before costs. Every rate is shown next to an **any-day baseline**: the same bracket entered at every session's close.

### Support breakdowns

Support is the most recent confirmed swing low (the lowest low of an 11-session span, known only 5 sessions after it prints). A **breakdown** is the first close below it. While a breakdown from the last 5 sessions is still unreclaimed, the short-term action is `SUPPORT_BREAKDOWN`: it overrides the buy-zone reading, the long entry moves to a close back above the broken level, and the card shows a follow-through target (1.5 ATR below the close) and the next swing-low support below.

Every earlier breakdown is replayed from its close: it "worked" (kept falling) when price dropped another 1.5 ATR before rising 1.5 ATR within 10 sessions, compared with the same downside bracket from every session's close. The replay also reports how often price closed back above the broken level within 10 sessions (failed breakdowns). When this was built, breakdowns in several large single stocks (AAPL, AMZN, GOOGL) kept falling 9–12 points more often than an arbitrary day, while broad ETFs (DIA, IWB, SMH, XLI) showed no edge, and 60–80% of breakdowns were reclaimed within 10 sessions. Read the ticker's own row before assuming a breakdown will continue.

Levels are causal: the zone that applied on a day was computed only from data before it. The zones shown for the next session must be recomputed after every daily close.

From the repository root:

```bash
cd frontend
npm install
npm run dev
```

`npm run dev` starts both services:

```text
FastAPI: http://127.0.0.1:8000
React:   http://127.0.0.1:5173
```

The frontend calls `/api/analyze` for the zone report and can optionally call `/api/interpret` for the Gemini panel. In development, Vite proxies those requests to FastAPI; in production, FastAPI exposes `/api/*` directly and serves the built React application from `frontend/dist`. Gemini receives the report without the per-session chart rows and is instructed to explain the supplied levels and statistics, never to invent new ones.

Zones need roughly one year of history, and the replay is more reliable with more. The default history start is `2012-01-01`.

## API

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Endpoints:

```text
GET  /health                  GET  /api/health
POST /analyze                 POST /api/analyze
POST /interpret               POST /api/interpret
POST /analyze-and-interpret   POST /api/analyze-and-interpret
```

Example `/analyze` input:

```json
{
  "ticker": "TSLA",
  "benchmark_primary": "QQQ",
  "benchmark_secondary": "SPY",
  "treasury_yield_ticker": "^TNX",
  "oil_ticker": "CL=F",
  "start_date": "2012-01-01"
}
```

The response (`schema_version` 4.0, `analysis_type: price_zones`) contains `current_market`, `short_term`, `signal_profile` (the signal settings used and their validation), `trend_outlook`, `technical_indicators`, `macro_context`, `chart_data` (`points` with per-session zone bands and `zone_touches`), and `methodology`. The zone engine lives in [src/entry_zones.py](src/entry_zones.py), the trend outlook in [src/trend_probability.py](src/trend_probability.py), and the report is assembled in [src/zone_analysis.py](src/zone_analysis.py).

## Render Deployment

Use a single Render Web Service from the repository root. The build must generate the Vite production files before the FastAPI server starts.

Build command:

```bash
pip install -r requirements.txt && cd frontend && npm ci && npm run build
```

Start command:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

Health check path:

```text
/health
```

Render honors the root `.python-version` file during a new build. If Yahoo cannot be reached from the running service, analysis requests return `503` with a market-data availability message; the dashboard displays that message instead of a server traceback.

Daily histories for the target, benchmarks, VIX, Treasury yield, and oil are requested concurrently to keep the first analysis request responsive on hosted instances. The backend logs the start and completion time for each analysis request.

## Notes

- The previous prediction pipeline (`predict.py`, `train.py`, `five_day_forecast.py`, `decision_engine.py`, `short_term_trade.py`) is no longer used by the API or dashboard. It remains in the repository for reference.
- Only one analysis runs at a time in a server process; a concurrent request returns HTTP `429`.
- If Yahoo cannot be reached, analysis requests return `503` with a market-data availability message.

# Quant Entry Zone Analyzer

Python and React app that shows **where to buy and where to sell** a stock or ETF instead of predicting its price. It downloads daily OHLCV data, calibrates four price zones from the ticker's own history, and replays every earlier visit to each zone so you can see how price behaved afterwards.

- **Short-term buy zone**: where price would be stretched below its 20-day EMA by as much as the ticker's 5th-20th percentile over the last 3 years, measured in ATRs. This is the V-bounce entry area.
- **Short-term take-profit zone**: the same stretch at the 80th-95th percentile above the EMA.
- **Long-term value zone**: the 200-day MA adjusted by the ticker's 5th-25th percentile distance from it over the last 5 years.
- **Long-term trim zone**: the same distance at the 85th-97th percentile.

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

- **Short-term entry card**: action (`ENTRY_CONFIRMED`, `ENTRY_ZONE_BOUNCE_FORMING`, `ENTRY_ZONE`, `BELOW_ENTRY_ZONE`, `APPROACHING_ENTRY`, `BOUNCE_UNDERWAY`, `WAIT_FOR_PULLBACK`, `TAKE_PROFIT_ZONE`), current price, next-session buy and take-profit zones with distance, a stretch-percentile gauge, and the historical bounce statistics
- **Long-term card**: action (`ACCUMULATE`, `DEEP_VALUE_TREND_RISK`, `APPROACHING_VALUE`, `HOLD_WAIT`, `APPROACHING_TRIM`, `TRIM_ZONE`), value and trim zones, distance from the 200-day MA, and 60-session outcomes after earlier visits
- **Trend outlook**: bullish/bearish probability for the next 10 and 60 sessions, with the walk-forward record that decides whether the model or the base rate is trusted
- **Chart** (3M, 6M, 1Y daily; 3Y and 5Y weekly on a log scale): adjusted candles with the zones drawn as they moved over time (short-term solid, long-term hatched), next session's zones at the right edge, markers on earlier zone touches coloured by outcome, and a stretch / RSI / MACD lower pane
- **Price ladder**: every level, highest first, with the move needed from the current price
- **Bounce checklist**: stretched, oversold, capitulation volume, reversal candle, reclaim of the prior high, and market backdrop
- **Short-term trade plan**: entry, stop, and targets that use the same 1.5 ATR bracket as the historical replay
- **Historical replay tables**, technical and macro context, and an optional Gemini explanation

### Trend outlook

The trend panel gives the probability of a higher close 10 and 60 sessions ahead. Two forecasters compete:

- the ticker's **base rate**: how often it closed higher over that horizon in its own history
- a **logistic regression** on distance from the 200-day MA, its slope, and 60/120-session returns, tested walk-forward (re-fit every 63 sessions on data whose outcomes were already known, then scored on the next block)

The headline uses the model only when it beat the base rate out of sample (Brier skill score of at least 2% and AUC of at least 0.56). Otherwise the headline is the base rate, and the model's reading is shown for reference with its test record. When this was built, neither this model nor larger feature sets beat the base rate on SPY, QQQ, SMH, NVDA, AAPL, XLE, TLT or GLD at either horizon, so expect the base rate to be the usual headline.

### Reading the bounce statistics

A short-term setup "works" when price moves 1.5 ATR in the expected direction before moving 1.5 ATR against it within 10 sessions, so about 50% is break-even before costs. Every rate is shown next to an **any-day baseline**: the same bracket entered at every session's close. The `historical_edge` grade compares the two, and an entry is never shown as a green light unless the ticker's own history shows an edge. The report also replays the "wait for a reversal close" entry so you can see whether waiting for confirmation helped for that ticker.

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

Long-term zones need roughly two years of history, and the replay is more reliable with more. The default history start is `2012-01-01`.

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

The response (`schema_version` 3.0, `analysis_type: price_zones`) contains `current_market`, `short_term`, `long_term`, `trend_outlook`, `technical_indicators`, `macro_context`, `chart_data` (`points` with per-session zone bands and `zone_touches`), and `methodology`. The zone engine lives in [src/entry_zones.py](src/entry_zones.py), the trend outlook in [src/trend_probability.py](src/trend_probability.py), and the report is assembled in [src/zone_analysis.py](src/zone_analysis.py).

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

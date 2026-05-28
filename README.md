# Quant Swing Regime Model V2

Python and React app for medium-to-long-term stock/ETF regime analysis. It downloads daily OHLCV data with `yfinance`, builds technical, macro, and relative-strength features, trains purged time-split forecast models, calculates long-term buy/trim/invalidation zones, and can send the quant JSON to Gemini for a structured second interpretation.

This is decision support, not financial advice. It is not a day-trading signal.

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
GEMINI_MODEL=gemini-3.1-pro-preview
```

Key lookup prefers `GEMINI_API_KEY`, then `gemini`, then `GOOGLE_API_KEY`, and ignores invalid-looking very short values.

The default Gemini model is `gemini-3.1-pro-preview`. You can set `GEMINI_MODEL` or pass `--gemini-model` to override it.

## Web Dashboard

The React frontend provides a readable research dashboard with:

- adjusted daily candlestick chart and volume
- MA20, MA50, and MA200 overlays
- RSI and MACD study panes
- concise long-term signal, five-session forecast, short-term entry plan, price zones, technical indicators, and macro context
- expandable definitions for every final signal

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

The frontend calls `/api/predict`. In development, Vite proxies that request to FastAPI; in production, FastAPI exposes `/api/*` directly and serves the built React application from `frontend/dist`. Web requests default to the lower-resource Extra Trees long-term model and do not write saved report or model artifacts unless `save_outputs: true` or `save_models: true` is supplied. Gemini interpretation is not required for the dashboard's core quantitative results.

The web page loads without running the models. A prediction request is made only after the user clicks `Analyze`.

Select `Auto ensemble (slower)` in the web form, or submit `"model": "auto"` in an API request, to train multiple long-term models and use a validation-weighted soft-vote ensemble. This can take materially longer than the interactive Extra Trees default on small hosted instances.

The optional Gemini endpoints receive the full quant JSON, including technical indicators, macro context, price zones, model probabilities, and forecast outputs. They may explain supplied levels, but must not invent replacement prices.

## API

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

The FastAPI server entry point is [src/api.py](src/api.py), where `app = FastAPI(...)` and the `/predict` routes are defined.

Endpoints:

```text
GET  /health
POST /predict
POST /interpret
POST /predict-and-interpret
GET  /api/health
POST /api/predict
POST /api/interpret
POST /api/predict-and-interpret
```

Example `/predict` input:

```json
{
  "ticker": "TSLA",
  "benchmark_primary": "QQQ",
  "benchmark_secondary": "SPY",
  "treasury_yield_ticker": "^TNX",
  "oil_ticker": "CL=F",
  "start_date": "2012-01-01",
  "horizon_days": 60
}
```

The `/predict` response includes `chart_data.points` for the web chart, `five_day_forecast`, `short_term_trade`, macro time-series context, and `available_signals`.

`short_term_trade` is a separate short-horizon execution assessment. The dashboard displays its buy-now decision in a banner above the Daily Price Chart and repeats the status in the chart legend. It exposes tactical pullback and confirmed-breakout entry points, and only supplies a stop-loss and estimated reward/risk when a conditional buy setup is active. A bearish five-session model results in `AVOID_SHORT_TERM_BUY`, even when future levels remain visible to monitor.

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

Render honors the root `.python-version` file during a new build. If Yahoo cannot be reached from the running service, prediction requests return `503` with a market-data availability message; the dashboard displays that message instead of a server traceback.

Daily histories for the target, benchmarks, VIX, Treasury yield, and oil are requested concurrently to keep the first analysis request responsive on hosted instances. The backend logs the start and completion time for each analysis request.

Only one model analysis runs at a time in a server process. A concurrent request returns HTTP `429` instead of competing for the instance's memory. Tree-based estimators use one internal worker on the hosted web path, and selecting the default Extra Trees model does not load XGBoost.

## Notes

- Features use past and current daily data only.
- Labels use future 60-day return for supervised training, then labels are excluded from features.
- Validation uses a purged time-based holdout, not a random split.
- If `xgboost` is unavailable, training falls back to `RandomForestClassifier`.

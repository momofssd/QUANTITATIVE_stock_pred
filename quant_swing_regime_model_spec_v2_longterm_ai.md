# Quant Swing / Market Regime Model V2 Current Spec

Last maintained: 2026-05-26

## Purpose

Build and maintain a React and FastAPI web application for medium-to-long-term stock and ETF analysis.

The system is not a day-trading tool. It classifies the current market regime, estimates forward 60-trading-day directional probabilities, calculates long-term and tactical price zones, trains a separate five-trading-day bullish/bearish and possible-close forecast, derives a five-session short-term entry plan from that forecast and technical confirmation, and sends the resulting structured JSON to Gemini for a second explanatory interpretation.

The product includes a React web dashboard backed by FastAPI. The dashboard presents chart history, technical studies, macro inputs, price zones, and brief forecast output in a readable brokerage-research layout.

The model should support questions such as:

- Is this ticker in a bull trend, bear trend, sideways consolidation, or trend deterioration phase?
- Is the current price a long-term buy point, wait zone, hold area, trim area, or avoid zone?
- If current price is not ideal, where are the preferred long-term and tactical pullback zones?
- Is the ticker more likely bullish or bearish over the next five trading days, and what conditional close estimate accompanies that direction?
- Is the current price a supported short-term buy, or should a trader wait for a tactical pullback or confirmed breakout entry?
- What are the important technical, relative-strength, volatility, macro, and risk/reward factors?

This system is quantitative decision support, not financial advice.

## Web Application Entry Point

The frontend lives under `frontend/` and is powered by Vite and React. From that directory:

```bash
npm install
npm run dev
```

The development command launches:

```text
FastAPI backend: http://127.0.0.1:8000
React frontend:  http://127.0.0.1:5173
```

Vite proxies frontend calls from `/api/*` to the FastAPI routes during development. In production, FastAPI exposes `/api/*` endpoints and serves the Vite build from `frontend/dist` at `/`; the deployment build step must run `npm run build` before Uvicorn starts. Initial page load must not train models or issue a `/predict` request; analysis begins only after the user submits the form with the `Analyze` button. Web prediction requests default to the interactive `extratrees` long-term model and to `save_outputs: false` and `save_models: false` so dashboard browsing is responsive and does not replace console-generated reports or write trained model artifacts. The full `auto` ensemble remains a selectable slower analysis path. The dashboard must remain usable without a Gemini request; Gemini is optional explanatory output rather than a dependency for charting or core quant results.

Hosted production builds use the repository-root `.python-version` to stay on Python `3.13`, and the tested `yfinance` version is pinned in `requirements.txt`. OHLCV retrieval uses Yahoo's chart response for direct daily-history reads and falls back to `yfinance` if needed. Histories for the target, comparison benchmarks, and macro inputs are loaded concurrently so a first hosted request does not wait for six serialized network calls. If required data remains unavailable, the API returns HTTP `503` with a readable market-data provider error for the dashboard to display. The server logs per-analysis start and completion timing for production diagnosis.

The hosted API permits one active model analysis per server process; concurrent analysis submissions return HTTP `429` rather than multiplying memory use. Tree-based estimators set `n_jobs=1` to constrain peak resource use on small instances, and the default `extratrees` execution path must not import XGBoost unless the user explicitly selects XGBoost or `auto`.

## Project Structure

```text
data/                        # Runtime-created; ignored by Git
models/                      # Runtime-created saved models; ignored by Git
outputs/                     # Optional runtime-created reports; ignored by Git
skills/
  long_term_trade_interpreter/
    SKILL.md
src/
  api.py
  data_loader.py
  decision_engine.py
  features.py
  five_day_forecast.py
  short_term_trade.py
  gpt_interpreter.py
  indicators.py
  json_exporter.py
  labels.py
  predict.py
  train.py
  utils.py
frontend/
  index.html
  package.json
  vite.config.js
  src/
    App.jsx
    main.jsx
    styles.css
    components/
      MarketChart.jsx
README.md
requirements.txt
tests/
```

## Data Sources

Use `yfinance` and cache downloaded CSV files under `data/cache/`.

Required price fields:

- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

Default benchmark inputs:

- Target ticker: submitted by the frontend or API caller
- Primary benchmark: `QQQ`
- Secondary benchmark: `SPY`
- Volatility benchmark: `^VIX`
- Treasury yield benchmark: `^TNX`
- Oil benchmark: `CL=F`

Optional data feeds must not block the app. If VIX, Treasury yield, or oil data is unavailable, record a warning in `data_quality.warnings` and continue.

## Feature Engineering

Features must use current and historical daily data only. Do not use future data as features.

### Trend Features

```text
close_above_20ma
close_above_50ma
close_above_200ma
ma20_slope
ma50_slope
ma200_slope
close_to_50ma_pct
close_to_200ma_pct
ma50_above_ma200
```

### Momentum Features

```text
return_20d
return_60d
return_120d
rsi_14
macd
macd_signal
macd_hist
adx_14
```

### Volatility Features

```text
atr_14
atr_pct
bollinger_band_width
rolling_vol_20d
rolling_vol_60d
max_drawdown_20d
max_drawdown_60d
```

### Volume Features

```text
volume_vs_20d_avg
volume_vs_60d_avg
up_day_volume_ratio
volume_dry_up_score
```

### Relative Strength Features

```text
relative_strength_vs_primary
rs_primary_20d_change
rs_primary_60d_change
rs_primary_above_50ma
relative_strength_vs_spy
rs_20d_change
rs_60d_change
rs_above_50ma
```

### Macro / Risk Features

```text
vix_level
vix_5d_change
vix_20d_change
vix_60d_change
vix_20d_slope_pct
ten_year_yield
ten_year_yield_5d_change
ten_year_yield_20d_change
ten_year_yield_60d_change
ten_year_yield_20d_slope
oil_price
oil_5d_change
oil_20d_change
oil_60d_change
oil_20d_slope_pct
```

These macro inputs are time-series features calculated at each historical model row, not only latest-value reporting fields. They allow both trained forecast paths to learn from whether volatility, yields, and oil are rising or falling at short and medium horizons.

## Label Design

### Long-Term Regime Target

The long-term supervised label is a 3-class forward-return label.

```text
0 = Bearish
1 = Sideways
2 = Bullish
```

Default rule:

```python
future_return = df["Adj Close"].shift(-horizon_days) / df["Adj Close"] - 1
future_return > 0.08   => Bullish
future_return < -0.08  => Bearish
otherwise              => Sideways
```

Labels are used for training only and must never be included in the feature matrix.

### Five-Trading-Day Targets

The short-horizon forecast uses two five-session forward targets:

```python
forward_return_5d = df["Adj Close"].shift(-5) / df["Adj Close"] - 1
direction_5d = 1 if forward_return_5d >= 0 else 0
```

```text
0 = Bearish
1 = Bullish
```

`forward_return_5d` trains the possible-close regression path. `direction_5d` trains the bullish/bearish probability classifier.

## Model Training

The full-ensemble model mode is:

```python
MODEL = "auto"
```

`auto` trains multiple classifiers and uses a validation-weighted soft-vote ensemble.

The interactive web/API default is `extratrees`, because a hosted request must return promptly with controlled resource use. The dashboard exposes `auto` as an explicit slower choice when the user wants the ensemble analysis.

Current candidates:

- `XGBClassifier`
- `RandomForestClassifier`
- `ExtraTreesClassifier`
- `GradientBoostingClassifier`
- `HistGradientBoostingClassifier`
- `LogisticRegression`

Training uses a purged time-based holdout split only. No random train/test shuffle is allowed. For a forward target of `N` sessions, the final `N` labeled rows before the validation block are excluded from the training portion used to score validation performance. This prevents overlapping forward outcomes from leaking across the split.

Each candidate is scored on the holdout set using macro-F1. The ensemble weights model probability votes by validation macro-F1 and records per-model validation scores in:

```json
model_metadata.validation_scores_macro_f1
```

The long-term classifier uses `purge_gap=horizon_days`, records benchmark baseline macro-F1 scores in `model_metadata.validation_baselines_macro_f1`, and records validation metadata including:

```text
validation_method = purged_time_based_holdout
purged_rows
final_fit_rows
```

After validation metrics have been computed, the final live-prediction model is refitted on all valid labeled historical rows.

Single-model options remain available:

```text
xgboost
randomforest
extratrees
gradientboosting
histgradientboosting
logistic
```

## Five-Trading-Day Forecast

`src/five_day_forecast.py` trains a short-horizon forecasting path independently of the long-term regime/decision engine.

### Models

```text
Direction model: ExtraTreesClassifier
Price/return model: ExtraTreesRegressor
Trees per fitted model: 100
Horizon: 5 trading sessions
Validation: purged time-based holdout with 5 purged rows
```

The direction classifier provides:

```text
predicted_direction = BULLISH | BEARISH
direction_probabilities.bullish
direction_probabilities.bearish
confidence_score = maximum direction probability
```

The displayed possible close is based on a forward-return regression conditioned on the selected predicted direction when enough historical samples are present. A bearish classification therefore uses historical bearish five-session return examples for its displayed price estimate. If the selected class has insufficient history for a conditional model, the general return model is the fallback. The general, unconditional regression estimate is also retained in JSON as supplemental diagnostic information.

The range around the possible close is based on validation mean absolute return error. It is an uncertainty range around a model estimate, not a guaranteed price interval.

## Short-Term Trade Plan

`src/short_term_trade.py` converts the trained five-session forecast and existing tactical technical levels into an explicit short-term long-entry assessment. It does not change the long-term signal.

Supported actions:

| Action | Meaning |
| --- | --- |
| `BUY_NOW_AT_SUPPORT` | Current price is inside the tactical pullback entry zone and the bullish five-session forecast is confirmed by technical conditions. |
| `BUY_BREAKOUT_CONFIRMED` | A short-term breakout has confirmed with supportive forecast, volume, and relative strength. |
| `BUY_ON_PULLBACK` | The short-term setup is constructive, but entry should wait for the tactical support zone. |
| `BUY_ON_BREAKOUT_CONFIRMATION` | The setup is constructive, but entry should wait for a confirmed close above the short-term trigger. |
| `WAIT` | Forecast or technical confirmation is insufficient for a new short-term buy. |
| `AVOID_SHORT_TERM_BUY` | The five-session forecast or market condition is unfavorable for a short-term long entry. |

The output includes:

- `is_good_short_term_buy_now`
- `has_conditional_buy_setup`
- tactical pullback entry range
- short-term breakout entry trigger and confirmation rules
- five-session possible close and modeled range
- ATR-based stop and estimated reward/risk only when a conditional buy setup is active

A bearish five-session forecast with sufficient confidence must not result in a favorable current short-term buy assessment.
The React dashboard displays `is_good_short_term_buy_now` and `action` in a prominent banner immediately above the Daily Price Chart, with entry levels and confidence visible without scrolling, and repeats the buy-now status inside the chart legend.

## Regime Inference

The model predicts 3-class probabilities, then rule logic maps current technical state into practical regime labels such as:

- `BULL_TREND`
- `BEAR_TREND`
- `SIDEWAYS_CONSOLIDATION`
- `UPTREND_CONSOLIDATION`
- `BREAKOUT_SETUP`
- `TREND_DETERIORATION`
- `BEAR_MARKET_RALLY`

The regime is derived from current trend, moving averages, returns, relative strength, volatility, and model probabilities.

## Decision Engine

`src/decision_engine.py` calculates:

- current price
- preferred long-term buy zone
- short-term tactical buy zone
- breakout buy trigger
- trim/sell zone
- invalidation zone
- risk/reward ratio
- confidence score and label
- final signal
- key reasons
- main risks

### Final Signals

| Signal | Implemented Meaning |
| --- | --- |
| `STRONG_LONG_TERM_BUY` | Strong long-term setup near a favorable entry area; normal long-term exposure may be appropriate. |
| `LONG_TERM_BUY` | Constructive long-term setup at an acceptable entry area; consider initiating or adding exposure. |
| `ACCUMULATE_ON_PULLBACK` | Trend is constructive, but price is above the preferred buy area; add only on a pullback toward support. |
| `HOLD` | Setup remains constructive, but evidence is not strong enough for a new buy signal; maintain existing exposure. |
| `WAIT` | Evidence or confidence is insufficient for entry; wait for better price or stronger confirmation. |
| `TRIM` | Price is extended or near resistance; consider reducing part of an existing position and avoid adding. |
| `SELL` | Trend is broken with meaningful bearish model support; consider exiting or materially reducing exposure. |
| `AVOID` | Risk is elevated or technical support is weak; avoid new exposure. |

The signal definitions are maintained in `SIGNAL_DEFINITIONS` and exported in `available_signals`. The console output prints the complete legend and marks the current final signal.

### Long-Term Buy Zone

The preferred long-term buy zone is calculated from:

- 50-day moving average
- 200-day moving average
- prior 120-day low
- ATR pullback bands

### Short-Term Tactical Buy Zone

The short-term tactical buy zone is calculated separately from:

- 20-day moving average
- 50-day moving average
- recent 20-day low
- ATR pullback band

This zone is tactical only. It must not be described as the main long-term accumulation zone unless it overlaps with long-term support.

### Breakout Buy Trigger

The breakout trigger uses prior 120-day resistance and ATR extension logic. Confirmation should require:

- close above resistance
- volume above 20-day average
- relative strength rising

### Trim / Sell Zone

The trim/sell zone uses:

- prior 120-day high
- prior 252-day high
- ATR extension
- overextension above moving averages

### Invalidation Zone

The invalidation level uses wider long-term risk logic:

- below 200-day moving average with volatility buffer
- below prior 120-day low
- below 2 ATR from current price

## Confidence Calculation

The confidence score is a 0.00 to 1.00 blend of:

- model probability strength
- technical score strength
- market regime alignment
- relative strength confirmation
- risk/reward quality
- signal agreement count

Labels:

```text
0.80 - 1.00 = VERY_HIGH
0.65 - 0.79 = HIGH
0.50 - 0.64 = MEDIUM
0.00 - 0.49 = LOW
```

Low confidence should not be hidden.

## JSON Output

`src/json_exporter.py` writes:

```text
outputs/latest_signal.json
outputs/latest_signal.md
```

Required top-level JSON keys:

```text
schema_version
ticker
as_of_date
timeframe
data_quality
benchmarks
current_market
decision
available_signals
price_zones
five_day_forecast
short_term_trade
near_term_forecast
chart_data
technical_indicators
relative_strength
macro_context
risk_management
reasoning
model_metadata
```

`schema_version` is currently `2.2`.

`available_signals` must provide the user-visible definition for each allowed final signal.

`chart_data` supplies the frontend's daily chart payload:

```json
{
  "interval": "1d",
  "price_basis": "adjusted_ohlc",
  "display_periods": 180,
  "points": [
    {
      "date": "2026-05-26",
      "open": 600.0,
      "high": 607.0,
      "low": 597.0,
      "close": 602.14,
      "volume": 1234567,
      "ma20": 580.0,
      "ma50": 560.0,
      "ma200": 470.0,
      "rsi_14": 58.0,
      "macd": 0.015,
      "macd_signal": 0.012,
      "macd_hist": 0.003
    }
  ]
}
```

The React dashboard renders adjusted candlesticks, volume bars, moving average overlays, and switchable RSI/MACD lower studies from this payload.

`five_day_forecast` contains a separately trained five-session forecast:

```json
{
  "horizon_trading_days": 5,
  "predicted_direction": "BEARISH",
  "confidence_score": 0.5856,
  "confidence_label": "MEDIUM",
  "confidence_basis": "direction_model_probability",
  "direction_probabilities": {
    "bullish": 0.4144,
    "bearish": 0.5856
  },
  "projected_return": -0.012,
  "possible_price_in_5_trading_days": 594.91,
  "price_estimate_basis": "direction_conditioned_forward_return_regression",
  "unconditional_expected_return": 0.0207,
  "unconditional_expected_price_in_5_trading_days": 614.6,
  "possible_price_range": {
    "low": 570.0,
    "high": 620.0,
    "basis": "validation_mean_absolute_return_error"
  },
  "direction_price_models_agree": true,
  "validation": {
    "method": "purged_time_based_holdout",
    "purged_rows": 5
  }
}
```

`short_term_trade` turns the five-session forecast and tactical technical confirmation into a user-facing entry plan:

```json
{
  "horizon_trading_days": 5,
  "action": "BUY_ON_PULLBACK",
  "is_good_short_term_buy_now": false,
  "has_conditional_buy_setup": true,
  "confidence_score": 0.64,
  "confidence_label": "MEDIUM",
  "entry_points": {
    "pullback_entry": {
      "low": 588.0,
      "high": 595.0,
      "status": "WAIT_FOR_PULLBACK"
    },
    "breakout_entry": {
      "trigger_price": 607.0,
      "status": "WAIT_FOR_CONFIRMED_CLOSE"
    }
  },
  "trade_management": {
    "active_for_recommended_setup": true,
    "stop_loss_price": 581.0,
    "possible_close_in_5_trading_days": 614.6,
    "possible_close_range": {
      "low": 588.93,
      "high": 640.28
    },
    "estimated_reward_to_risk": 1.8
  }
}
```

For `WAIT` or `AVOID_SHORT_TERM_BUY`, monitored entry points may still be supplied, but `active_for_recommended_setup` is false and active stop/reward-risk values are null.

`near_term_forecast` must include quant-computed scenario zones. These zones are not guaranteed predictions; they are scenario boundaries for Gemini to interpret.

Required scenario structure:

```json
{
  "timeframe_scope": "near_term_swing",
  "horizon_candidates": [
    "2-4 weeks if breakout or momentum continuation confirms",
    "4-8 weeks if price consolidates before moving",
    "1-3 months if volatility remains elevated or macro risk delays follow-through"
  ],
  "bullish_scenario": {
    "target_zone_low": 100.0,
    "target_zone_high": 110.0,
    "probability_hint": 0.42
  },
  "base_case_scenario": {
    "target_zone_low": 92.0,
    "target_zone_high": 101.0,
    "probability_hint": 0.35
  },
  "bearish_scenario": {
    "pullback_zone_low": 82.0,
    "pullback_zone_high": 90.0,
    "probability_hint": 0.23
  }
}
```

For this scenario-based `near_term_forecast`, Gemini may determine the likely near-term timeframe, but it must select price levels from these quant-provided scenario zones. This restriction is separate from the trained five-session possible-close value supplied in `five_day_forecast`.

`macro_context` must include VIX, 10-year Treasury yield, and oil fields when available:

```json
{
  "vix_level": 18.06,
  "vix_5d_change": -0.012,
  "vix_20d_change": -0.0738,
  "vix_60d_change": 0.044,
  "vix_20d_slope_per_session": -0.0037,
  "vix_change_units": "decimal_return",
  "treasury_yield_benchmark": "^TNX",
  "ten_year_yield": 4.667,
  "ten_year_yield_5d_change": 0.042,
  "ten_year_yield_20d_change": 0.375,
  "ten_year_yield_60d_change": 0.638,
  "ten_year_yield_20d_slope_per_session": 0.0188,
  "ten_year_yield_change_units": "percentage_points",
  "oil_benchmark": "CL=F",
  "oil_price": 104.28,
  "oil_5d_change": 0.018,
  "oil_20d_change": 0.1319,
  "oil_60d_change": 0.5726,
  "oil_20d_slope_per_session": 0.0066,
  "oil_change_units": "decimal_return",
  "macro_risk_level": "HIGH"
}
```

`model_metadata.macro_feature_columns` records the exact macro columns supplied to the trained long-term model. The five-day forecast also reports its macro feature columns inside `five_day_forecast.model_metadata`.

## Gemini AI Interpretation Layer

The AI interpreter is implemented in:

```text
src/gpt_interpreter.py
```

The function name `interpret_signal_with_gpt()` exists as a compatibility alias, but the actual integration uses Gemini:

```python
def interpret_signal_with_gemini(signal_json: dict, model: str | None = None) -> dict:
    ...
```

Supported API key environment variables:

```text
GEMINI_API_KEY
gemini
GOOGLE_API_KEY
```

Key lookup order:

1. `GEMINI_API_KEY`
2. `gemini`
3. `GOOGLE_API_KEY`

Very short invalid-looking keys are ignored.

The default model constant is:

```python
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
```

## Skill File Usage

The local skill file is:

```text
skills/long_term_trade_interpreter/SKILL.md
```

`src/gpt_interpreter.py` reads this file through `_skill_instructions()` and prepends it to the Gemini prompt.

This means Gemini receives:

1. The local skill policy from `SKILL.md`
2. The explicit technical-analysis prompt
3. The full quant JSON payload
4. The structured output schema

When `five_day_forecast` is present, Gemini must treat its predicted direction, confidence, and possible price as the explicit trained five-session forecast. The direction-conditioned possible close is distinct from any supplemental unconditional expected close. Gemini must not invent a replacement forecast price. When `short_term_trade` is present, Gemini must preserve its computed short-term buy-now decision, action, entry points, and active trade-management levels rather than inventing an alternate trade plan.

The application uses `skills/long_term_trade_interpreter/SKILL.md`.

## Gemini Output Schema

The AI response must be structured JSON with:

```json
{
  "ticker": "string",
  "final_view": "string",
  "is_current_price_long_term_buy": false,
  "is_current_price_long_term_sell": false,
  "preferred_buy_zone": "string",
  "short_term_tactical_buy_zone": "string",
  "breakout_buy_trigger": "string",
  "trim_or_sell_zone": "string",
  "invalidation_level": "string",
  "confidence_score": 0.0,
  "confidence_label": "LOW|MEDIUM|HIGH|VERY_HIGH",
  "plain_english_summary": "string",
  "technical_data_used": {
    "trend": ["string"],
    "momentum": ["string"],
    "volatility": ["string"],
    "volume": ["string"],
    "relative_strength": ["string"],
    "macro": ["string"]
  },
  "technical_assessment": {
    "trend_health": "string",
    "momentum_condition": "string",
    "volatility_condition": "string",
    "relative_strength_condition": "string",
    "risk_reward_condition": "string"
  },
  "near_term_price_forecast": {
    "selected_scenario": "string",
    "estimated_timeframe": "string",
    "target_price_zone": "string",
    "downside_or_failure_zone": "string",
    "confidence_label": "LOW|MEDIUM|HIGH|VERY_HIGH",
    "probability_comment": "string",
    "conditions_needed": ["string"],
    "what_would_invalidate_forecast": ["string"]
  },
  "key_reasons": ["string"],
  "main_risks": ["string"],
  "not_financial_advice": true
}
```

Hard rule: the AI may explain, summarize, and compare scenarios, but it must not invent price levels or indicator values outside the quant JSON.

## API

`src/api.py` provides:

```text
GET /health
POST /predict
POST /interpret
POST /predict-and-interpret
GET /api/health
POST /api/predict
POST /api/interpret
POST /api/predict-and-interpret
```

Example request:

```json
{
  "ticker": "TSLA",
  "benchmark_primary": "QQQ",
  "benchmark_secondary": "SPY",
  "treasury_yield_ticker": "^TNX",
  "oil_ticker": "CL=F",
  "start_date": "2012-01-01",
  "horizon_days": 60,
  "model": "auto",
  "use_gpt_interpreter": true,
  "gemini_model": "gemini-3.8-flash"
}
```

## Tests

Required tests:

- `test_decision_engine.py`
- `test_five_day_forecast.py`
- `test_short_term_trade.py`
- `test_json_schema.py`
- `test_gpt_interpreter_mock.py`
- `test_macro_features.py`
- `test_train_validation.py`

Minimum invariants:

- JSON output contains required fields
- confidence score is between 0 and 1
- long-term buy zone low < high
- short-term tactical buy zone low < high
- trim zone low < high
- invalidation price is below current price for long setups
- final signal is one of the allowed categories
- signal definitions are exported for every allowed final signal
- five-day direction/confidence/possible-price forecast is present and uses purged validation
- short-horizon price estimate is direction-conditioned and preserves unconditional diagnostic output
- short-term trade output reports whether buying now is supported and exports entry points
- a bearish five-session forecast does not produce a favorable current short-term buy assessment
- macro time-series movement and slope features enter the model matrix
- long-term validation purges forward-horizon overlap and reports baseline scores
- Gemini interpreter can be mocked

Run:

```bash
python -m pytest -q
```

## Anti-Leakage Rules

Allowed:

- historical returns
- moving averages and slopes
- historical volatility
- volume ratios
- relative strength based on current/past benchmark data
- current and past macro/risk benchmark values
- macro benchmark movements and slopes calculated from current/past observations

Not allowed:

- future return as a feature
- future high/low as a feature
- labels included in feature matrix
- random train/test split
- validation training labels whose forward outcome overlaps the validation block
- AI-created price zones not present in quant JSON

## Final Objective

Create a practical medium-term and long-term swing regime model that identifies:

- bull market continuation
- bear market risk
- sideways consolidation
- healthy pullback
- uptrend consolidation
- breakout setup
- trend deterioration
- long-term buy/wait/hold/trim/sell/avoid decisions
- short-term tactical pullback zones
- trained five-trading-day bullish/bearish probabilities and direction-conditioned possible close
- short-term buy-now assessment with tactical pullback and breakout entry points
- probabilistic near-term price forecast scenarios
- macro time-series training inputs and reported context from VIX, 10-year Treasury yield, and oil
- displayed definitions for every available final signal

The quant model produces the data-driven signal. Gemini explains it using the local skill file and the structured quant JSON.

import { useState } from "react";
import MarketChart from "./components/MarketChart";

const initialForm = {
  ticker: "SMH",
  start_date: "2018-01-01",
  benchmark_primary: "QQQ",
  benchmark_secondary: "SPY",
  treasury_yield_ticker: "^TNX",
  oil_ticker: "CL=F",
  horizon_days: 60,
  model: "extratrees",
};

const dollar = (value) => (typeof value === "number" ? `$${value.toFixed(2)}` : "-");
const percent = (value) => (typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "-");
const fixed = (value, digits = 2) => (typeof value === "number" ? value.toFixed(digits) : "-");

function toneForSignal(signal) {
  if (["STRONG_LONG_TERM_BUY", "LONG_TERM_BUY"].includes(signal)) return "positive";
  if (["ACCUMULATE_ON_PULLBACK", "HOLD", "WAIT"].includes(signal)) return "caution";
  return "negative";
}

function toneForShortTerm(action) {
  if (["BUY_NOW_AT_SUPPORT", "BUY_BREAKOUT_CONFIRMED"].includes(action)) return "positive";
  if (["BUY_ON_PULLBACK", "BUY_ON_BREAKOUT_CONFIRMATION", "WAIT"].includes(action)) return "caution";
  return "negative";
}

function MetricCard({ label, value, supporting, tone = "" }) {
  return (
    <div className={`metric-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{supporting}</small>
    </div>
  );
}

function ZoneRow({ label, value, muted = false }) {
  return (
    <div className={`zone-row ${muted ? "muted" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [signal, setSignal] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function runAnalysis(event) {
    event?.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!response.ok) {
        let message = `Prediction failed with ${response.status}`;
        try {
          const body = await response.json();
          message = body.detail || message;
        } catch {
          // Preserve the status-based message when the server does not return JSON.
        }
        throw new Error(message);
      }
      setSignal(await response.json());
    } catch (requestError) {
      setError(requestError.message || "Unable to reach the analysis API.");
    } finally {
      setLoading(false);
    }
  }

  const decision = signal?.decision;
  const market = signal?.current_market;
  const fiveDay = signal?.five_day_forecast;
  const shortTrade = signal?.short_term_trade;
  const zones = signal?.price_zones;
  const technical = signal?.technical_indicators;
  const macro = signal?.macro_context;

  return (
    <div className="page">
      <header className="site-header">
        <div className="brand">
          <span className="brand-block">Q</span>
          <div>
            <strong>Quant Market Research</strong>
            <small>Stocks and ETF analysis</small>
          </div>
        </div>
        <nav>
          <span className="active">Research</span>
          <span>Markets</span>
          <span>Watchlist</span>
          <span>Strategy</span>
        </nav>
        <div className="disclaimer">Decision support only</div>
      </header>

      <main className="content">
        <section className="research-header">
          <div>
            <p className="eyebrow">Equity Research</p>
            <h1>{signal?.ticker || form.ticker.toUpperCase()}</h1>
            <p className="subtitle">
              Long-term regime and five-trading-day forecast
              {signal && ` | Data through ${signal.as_of_date}`}
            </p>
          </div>
          <form className="ticker-form" onSubmit={runAnalysis}>
            <label>
              Symbol
              <input
                onChange={(event) => setForm({ ...form, ticker: event.target.value.toUpperCase() })}
                value={form.ticker}
              />
            </label>
            <label>
              History start
              <input
                onChange={(event) => setForm({ ...form, start_date: event.target.value })}
                type="date"
                value={form.start_date}
              />
            </label>
            <label>
              Model
              <select onChange={(event) => setForm({ ...form, model: event.target.value })} value={form.model}>
                <option value="extratrees">Extra Trees (fast)</option>
                <option value="randomforest">Random Forest</option>
                <option value="auto">Auto ensemble (slower)</option>
                <option value="xgboost">XGBoost</option>
                <option value="logistic">Logistic</option>
              </select>
            </label>
            <button disabled={loading} type="submit">{loading ? "Analyzing..." : "Analyze"}</button>
          </form>
        </section>

        {error && <div className="error-panel">Analysis unavailable: {error}</div>}
        {loading && !signal && <div className="loading-panel">Training models and preparing chart history...</div>}
        {!loading && !signal && !error && (
          <section className="welcome-panel">
            <div>
              <p className="eyebrow">Start Research</p>
              <h2>Run an analysis when you are ready.</h2>
              <p>
                Enter a ticker and click <strong>Analyze</strong> to train the models, load the daily chart,
                and generate the long-term and five-session outlook.
              </p>
            </div>
            <div className="welcome-steps">
              <div>
                <strong>1</strong>
                <span>Choose ticker and model</span>
              </div>
              <div>
                <strong>2</strong>
                <span>Click Analyze</span>
              </div>
              <div>
                <strong>3</strong>
                <span>Review chart and signals</span>
              </div>
            </div>
          </section>
        )}

        {signal && (
          <>
            <section className="metrics-grid">
              <MetricCard label="Last price" supporting={`As of ${signal.as_of_date}`} value={dollar(market.current_price)} />
              <MetricCard
                label="Long-term signal"
                supporting={`Confidence ${decision.confidence_label} | ${fixed(decision.confidence_score, 4)}`}
                tone={toneForSignal(decision.final_signal)}
                value={decision.final_signal.replaceAll("_", " ")}
              />
              <MetricCard label="Market regime" supporting="60-session regime model" value={market.regime.replaceAll("_", " ")} />
              <MetricCard
                label="Next 5 sessions"
                supporting={`${fiveDay.confidence_label} | ${percent(fiveDay.confidence_score)}`}
                tone={fiveDay.predicted_direction === "BULLISH" ? "positive" : "negative"}
                value={fiveDay.predicted_direction}
              />
            </section>

            {shortTrade && (
              <section className={`short-term-banner ${toneForShortTerm(shortTrade.action)}`}>
                <div>
                  <p className="eyebrow">Short-Term Buy Decision</p>
                  <h2>{shortTrade.is_good_short_term_buy_now ? "BUY NOW SUPPORTED" : "DO NOT BUY NOW"}</h2>
                  <p>{shortTrade.summary}</p>
                </div>
                <div className="short-term-banner-facts">
                  <div>
                    <span>Action</span>
                    <strong>{shortTrade.action.replaceAll("_", " ")}</strong>
                  </div>
                  <div>
                    <span>Pullback Entry</span>
                    <strong>
                      {dollar(shortTrade.entry_points.pullback_entry.low)} - {dollar(shortTrade.entry_points.pullback_entry.high)}
                    </strong>
                  </div>
                  <div>
                    <span>Breakout Entry</span>
                    <strong>{dollar(shortTrade.entry_points.breakout_entry.trigger_price)}</strong>
                  </div>
                  <div>
                    <span>Confidence</span>
                    <strong>{shortTrade.confidence_label} | {percent(shortTrade.confidence_score)}</strong>
                  </div>
                </div>
              </section>
            )}

            <div className="dashboard-layout">
              <div className="main-column">
                <MarketChart forecast={fiveDay} points={signal.chart_data.points} shortTrade={shortTrade} zones={zones} />
                <section className="details-grid">
                  <article className="panel">
                    <h2>Technical Snapshot</h2>
                    <div className="fact-grid">
                      <ZoneRow label="MA 20" value={dollar(technical.ma20)} />
                      <ZoneRow label="MA 50" value={dollar(technical.ma50)} />
                      <ZoneRow label="MA 200" value={dollar(technical.ma200)} />
                      <ZoneRow label="RSI (14)" value={fixed(technical.rsi_14)} />
                      <ZoneRow label="ATR (14)" value={dollar(technical.atr_14)} />
                      <ZoneRow label="20d return" value={percent(technical.return_20d)} />
                    </div>
                  </article>
                  <article className="panel">
                    <h2>Macro Context</h2>
                    <div className="fact-grid">
                      <ZoneRow label="VIX" value={fixed(macro.vix_level)} />
                      <ZoneRow label="VIX 5d" value={percent(macro.vix_5d_change)} />
                      <ZoneRow label="10Y yield" value={`${fixed(macro.ten_year_yield, 3)}%`} />
                      <ZoneRow label="10Y 20d" value={`${fixed(macro.ten_year_yield_20d_change, 3)} pts`} />
                      <ZoneRow label="Oil" value={dollar(macro.oil_price)} />
                      <ZoneRow label="Macro risk" value={macro.macro_risk_level} />
                    </div>
                  </article>
                </section>
              </div>

              <aside className="sidebar">
                <article className="panel result-panel">
                  <p className="eyebrow">Model Result</p>
                  <h2 className={toneForSignal(decision.final_signal)}>{decision.final_signal.replaceAll("_", " ")}</h2>
                  <p className="summary">{decision.summary}</p>
                  <div className="confidence">
                    <span>Confidence</span>
                    <strong>{decision.confidence_label}</strong>
                    <div><i style={{ width: `${decision.confidence_score * 100}%` }} /></div>
                  </div>
                </article>

                <article className="panel">
                  <h2>Price Zones</h2>
                  <ZoneRow
                    label="Preferred buy"
                    value={`${dollar(zones.preferred_long_term_buy_zone.low)} - ${dollar(zones.preferred_long_term_buy_zone.high)}`}
                  />
                  <ZoneRow
                    label="Tactical pullback"
                    value={`${dollar(zones.short_term_tactical_buy_zone.low)} - ${dollar(zones.short_term_tactical_buy_zone.high)}`}
                  />
                  <ZoneRow label="Breakout trigger" value={dollar(zones.breakout_buy_zone.trigger_price)} />
                  <ZoneRow
                    label="Trim zone"
                    value={`${dollar(zones.trim_or_sell_zone.low)} - ${dollar(zones.trim_or_sell_zone.high)}`}
                  />
                  <ZoneRow label="Invalidation" muted value={dollar(zones.invalidation_zone.price)} />
                </article>

                <article className="panel five-day">
                  <h2>Five-Session Outlook</h2>
                  <div className={`direction ${fiveDay.predicted_direction.toLowerCase()}`}>{fiveDay.predicted_direction}</div>
                  <ZoneRow label="Probability confidence" value={percent(fiveDay.confidence_score)} />
                  <ZoneRow label="Possible close" value={dollar(fiveDay.possible_price_in_5_trading_days)} />
                  <ZoneRow
                    label="Error range"
                    value={`${dollar(fiveDay.possible_price_range.low)} - ${dollar(fiveDay.possible_price_range.high)}`}
                  />
                  <p className="footnote">{fiveDay.interpretation_note}</p>
                </article>

                {shortTrade && (
                  <article className="panel short-trade">
                    <p className="eyebrow">Short-Term Entry Plan</p>
                    <h2 className={toneForShortTerm(shortTrade.action)}>
                      {shortTrade.action.replaceAll("_", " ")}
                    </h2>
                    <p className="summary">{shortTrade.summary}</p>
                    <ZoneRow
                      label="Good buy now?"
                      value={shortTrade.is_good_short_term_buy_now ? "YES" : "NO"}
                      muted={!shortTrade.is_good_short_term_buy_now}
                    />
                    <ZoneRow
                      label="Pullback entry"
                      value={`${dollar(shortTrade.entry_points.pullback_entry.low)} - ${dollar(shortTrade.entry_points.pullback_entry.high)}`}
                    />
                    <ZoneRow
                      label="Breakout entry"
                      value={dollar(shortTrade.entry_points.breakout_entry.trigger_price)}
                    />
                    {shortTrade.has_conditional_buy_setup && (
                      <ZoneRow label="Stop loss" muted value={dollar(shortTrade.trade_management.stop_loss_price)} />
                    )}
                    <ZoneRow
                      label="Possible 5D close"
                      value={dollar(shortTrade.trade_management.possible_close_in_5_trading_days)}
                    />
                    {shortTrade.has_conditional_buy_setup && (
                      <ZoneRow
                        label="Est. reward/risk"
                        value={`${fixed(shortTrade.trade_management.estimated_reward_to_risk)}x`}
                      />
                    )}
                    <div className="confidence short-confidence">
                      <span>Setup confidence</span>
                      <strong>{shortTrade.confidence_label}</strong>
                      <div><i style={{ width: `${shortTrade.confidence_score * 100}%` }} /></div>
                    </div>
                  </article>
                )}

                <details className="panel signal-key">
                  <summary>Signal Definitions</summary>
                  {Object.entries(signal.available_signals).map(([key, meaning]) => (
                    <div className={key === decision.final_signal ? "selected-signal" : ""} key={key}>
                      <strong>{key.replaceAll("_", " ")}</strong>
                      <p>{meaning}</p>
                    </div>
                  ))}
                </details>
              </aside>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

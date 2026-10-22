import { useState } from "react";
import AiPanel from "./components/AiPanel";
import MarketChart from "./components/MarketChart";
import {
  EvidenceTables,
  LevelLadder,
  ReadinessPanel,
  ReasonsPanel,
  ShortTermHero,
  TradePlanPanel,
  TrendOutlook,
} from "./components/ZonePanels";
import { dollar, fixed, percent } from "./components/format";

const initialForm = {
  ticker: "SMH",
  start_date: "2012-01-01",
  benchmark_primary: "QQQ",
  benchmark_secondary: "SPY",
  treasury_yield_ticker: "^TNX",
  oil_ticker: "CL=F",
  use_gpt_interpreter: false,
};

function ZoneRow({ label, value }) {
  return (
    <div className="zone-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [report, setReport] = useState(null);
  const [aiInterpretation, setAiInterpretation] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function readError(response, fallback) {
    try {
      const body = await response.json();
      return body.detail || fallback;
    } catch {
      return fallback;
    }
  }

  async function requestAiInterpretation(payload = report) {
    if (!payload) return;
    setAiLoading(true);
    setAiError("");
    try {
      const response = await fetch("/api/interpret", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(await readError(response, `AI interpretation failed with ${response.status}`));
      }
      setAiInterpretation(await response.json());
    } catch (requestError) {
      setAiError(requestError.message || "Unable to reach the AI interpretation API.");
    } finally {
      setAiLoading(false);
    }
  }

  async function runAnalysis(event) {
    event?.preventDefault();
    setLoading(true);
    setError("");
    setAiError("");
    setAiInterpretation(null);
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!response.ok) {
        throw new Error(await readError(response, `Analysis failed with ${response.status}`));
      }
      const nextReport = await response.json();
      setReport(nextReport);
      if (form.use_gpt_interpreter) {
        await requestAiInterpretation(nextReport);
      }
    } catch (requestError) {
      setError(requestError.message || "Unable to reach the analysis API.");
    } finally {
      setLoading(false);
    }
  }

  const shortTerm = report?.short_term;
  const technical = report?.technical_indicators;
  const macro = report?.macro_context;
  const warnings = report?.data_quality?.warnings || [];

  return (
    <div className="page">
      <header className="site-header">
        <div className="brand">
          <span className="brand-block">Q</span>
          <div>
            <strong>Entry Zone Analyzer</strong>
            <small>Stocks and ETFs</small>
          </div>
        </div>
        <div className="disclaimer">Decision support only · not financial advice</div>
      </header>

      <main className="content">
        <section className="research-header">
          <div>
            <p className="eyebrow">Price-zone research</p>
            <h1>{report?.ticker || form.ticker.toUpperCase()}</h1>
            <p className="subtitle">
              Short-term bounce entries and take-profit zones
              {report && ` | Data through ${report.as_of_date}`}
            </p>
          </div>
          <form className="ticker-form" onSubmit={runAnalysis}>
            <label>
              Symbol
              <input onChange={(event) => setForm({ ...form, ticker: event.target.value.toUpperCase() })} value={form.ticker} />
            </label>
            <label>
              History start
              <input onChange={(event) => setForm({ ...form, start_date: event.target.value })} type="date" value={form.start_date} />
            </label>
            <label className="ai-toggle">
              <input
                checked={form.use_gpt_interpreter}
                onChange={(event) => setForm({ ...form, use_gpt_interpreter: event.target.checked })}
                type="checkbox"
              />
              <span>AI explanation</span>
            </label>
            <button disabled={loading} type="submit">
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </form>
        </section>

        {error && <div className="error-panel">Analysis unavailable: {error}</div>}
        {loading && !report && <div className="loading-panel">Loading price history and calibrating zones...</div>}
        {!loading && !report && !error && (
          <section className="welcome-panel">
            <div>
              <p className="eyebrow">Start research</p>
              <h2>Find where to buy and where to sell.</h2>
              <p>
                Enter a ticker and click <strong>Analyze</strong>. The app calibrates short-term buy and sell zones
                from the ticker's own history, tells you how close the current price is to an entry, and shows how often earlier
                visits produced a V-shaped bounce.
              </p>
            </div>
            <div className="welcome-steps">
              <div>
                <strong>1</strong>
                <span>Choose a ticker</span>
              </div>
              <div>
                <strong>2</strong>
                <span>Check distance to the entry zone</span>
              </div>
              <div>
                <strong>3</strong>
                <span>Weigh the bounce history and checklist</span>
              </div>
            </div>
          </section>
        )}

        {report && (
          <>
            {warnings.length > 0 && (
              <div className="warning-panel">
                Data warnings: {warnings.join(" ")}
              </div>
            )}
            <section className="decision-grid">
              <ShortTermHero market={report.current_market} shortTerm={shortTerm} />
            </section>
            <TrendOutlook outlook={report.trend_outlook} ticker={report.ticker} />

            <div className="dashboard-layout">
              <div className="main-column">
                <MarketChart chart={report.chart_data} shortTerm={shortTerm} />
                <ReasonsPanel shortTerm={shortTerm} />
                <EvidenceTables shortTerm={shortTerm} />
                <section className="details-grid">
                  <article className="panel">
                    <h2>Technical snapshot</h2>
                    <div className="fact-grid">
                      <ZoneRow label="EMA 20" value={dollar(technical.ema20)} />
                      <ZoneRow label="MA 50" value={dollar(technical.ma50)} />
                      <ZoneRow label="MA 200" value={dollar(technical.ma200)} />
                      <ZoneRow label="ATR (14)" value={`${dollar(technical.atr_14)} · ${percent(technical.atr_pct)}`} />
                      <ZoneRow label="RSI (14)" value={fixed(technical.rsi_14, 1)} />
                      <ZoneRow label="RSI (2)" value={fixed(technical.rsi_2, 1)} />
                      <ZoneRow label="20d return" value={percent(technical.return_20d)} />
                      <ZoneRow label="Volume vs 20d" value={`${fixed(technical.volume_vs_20d_avg)}x`} />
                    </div>
                  </article>
                  <article className="panel">
                    <h2>Macro context</h2>
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
                <AiPanel
                  disabled={loading}
                  error={aiError}
                  interpretation={aiInterpretation}
                  loading={aiLoading}
                  onGenerate={() => requestAiInterpretation()}
                />
              </div>

              <aside className="sidebar">
                <LevelLadder report={report} />
                <ReadinessPanel readiness={shortTerm.bounce_readiness} />
                <TradePlanPanel shortTerm={shortTerm} />
                <details className="panel signal-key">
                  <summary>How the zones are built</summary>
                  <div>
                    <p>{report.methodology.summary}</p>
                  </div>
                  <div>
                    <strong>Short-term buy / take-profit</strong>
                    <p>{shortTerm.buy_zone.basis}; {shortTerm.sell_zone.basis}.</p>
                  </div>
                  <div>
                    <strong>Bounce replay</strong>
                    <p>{shortTerm.history.buy_zone.definition}</p>
                  </div>
                  <div>
                    <strong>Levels apply to</strong>
                    <p>The {report.methodology.levels_apply_to}.</p>
                  </div>
                </details>
              </aside>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

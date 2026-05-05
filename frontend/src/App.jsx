import axios from "axios";
import { useMemo, useState } from "react";
import "./App.css";

const API_BASE = "http://localhost:5000";

function formatMoney(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
}

function formatOptionalMoney(value) {
  return Number.isFinite(value) ? formatMoney(value) : "--";
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDate(value) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(new Date(value));
}

function getDomain(series, keys, fallback = [0, 1]) {
  const values = series
    .flatMap((point) => keys.map((key) => point[key]))
    .filter((value) => Number.isFinite(value));

  if (!values.length) return fallback;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const padding = (max - min || Math.abs(max) || 1) * 0.08;
  return [min - padding, max + padding];
}

function linePath(series, key, xFor, yFor) {
  let started = false;
  return series
    .map((point, index) => {
      const value = point[key];
      if (!Number.isFinite(value)) return null;
      const command = started ? "L" : "M";
      started = true;
      return `${command} ${xFor(index).toFixed(2)} ${yFor(value).toFixed(2)}`;
    })
    .filter(Boolean)
    .join(" ");
}

function LineChart({ series, lines, yDomain, markers = false, threshold = null }) {
  const width = 820;
  const height = 360;
  const pad = { top: 26, right: 30, bottom: 44, left: 58 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const domain = yDomain || getDomain(series, lines.map((line) => line.key));
  const [minY, maxY] = domain;
  const xFor = (index) => pad.left + (series.length <= 1 ? 0 : (index / (series.length - 1)) * plotW);
  const yFor = (value) => pad.top + ((maxY - value) / (maxY - minY || 1)) * plotH;
  const firstDate = series[0]?.date ?? "";
  const lastDate = series.at(-1)?.date ?? "";

  return (
    <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img">
      <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} className="axis" />
      <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} className="axis" />
      {[0.25, 0.5, 0.75].map((tick) => (
        <line
          key={tick}
          x1={pad.left}
          x2={width - pad.right}
          y1={pad.top + plotH * tick}
          y2={pad.top + plotH * tick}
          className="grid"
        />
      ))}
      {markers &&
        series.map((point, index) => (
          <rect
            key={`${point.date}-${index}`}
            x={xFor(index) - plotW / Math.max(series.length - 1, 1) / 2}
            y={pad.top}
            width={Math.max(plotW / Math.max(series.length - 1, 1), 2)}
            height={plotH}
            className={point.correct ? "correct-band" : "wrong-band"}
          />
        ))}
      {threshold !== null && (
        <line
          x1={pad.left}
          x2={width - pad.right}
          y1={yFor(threshold)}
          y2={yFor(threshold)}
          className="threshold"
        />
      )}
      {lines.map((line) => (
        <path
          key={line.key}
          d={linePath(series, line.key, xFor, yFor)}
          fill="none"
          stroke={line.color}
          strokeWidth={line.width ?? 2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ))}
      <text x={pad.left} y={height - 10} className="axis-label">
        {firstDate}
      </text>
      <text x={width - pad.right} y={height - 10} textAnchor="end" className="axis-label">
        {lastDate}
      </text>
      <text x={pad.left - 8} y={pad.top + 4} textAnchor="end" className="axis-label">
        {maxY.toFixed(maxY <= 1 ? 2 : 0)}
      </text>
      <text x={pad.left - 8} y={height - pad.bottom} textAnchor="end" className="axis-label">
        {minY.toFixed(minY <= 1 ? 2 : 0)}
      </text>
    </svg>
  );
}

function MetricsChart({ metrics }) {
  const width = 820;
  const height = 360;
  const pad = { top: 28, right: 28, bottom: 58, left: 54 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const groupW = plotW / Math.max(metrics.length, 1);
  const barW = Math.min(groupW * 0.28, 28);

  return (
    <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img">
      <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} className="axis" />
      <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} className="axis" />
      {[0.25, 0.5, 0.75].map((tick) => (
        <line key={tick} x1={pad.left} x2={width - pad.right} y1={pad.top + plotH * tick} y2={pad.top + plotH * tick} className="grid" />
      ))}
      {metrics.map((metric, index) => {
        const x = pad.left + index * groupW + groupW / 2;
        const accH = metric.accuracy * plotH;
        const aucH = metric.auc * plotH;
        return (
          <g key={metric.model}>
            <rect x={x - barW - 2} y={height - pad.bottom - accH} width={barW} height={accH} className="bar-primary" />
            <rect x={x + 2} y={height - pad.bottom - aucH} width={barW} height={aucH} className="bar-secondary" />
            <text x={x} y={height - 18} textAnchor="middle" className="axis-label">
              {metric.model.replace("RandomForest", "RF").replace("GradBoost", "GB")}
            </text>
          </g>
        );
      })}
      <text x={pad.left - 8} y={pad.top + 4} textAnchor="end" className="axis-label">
        1.00
      </text>
      <text x={pad.left - 8} y={height - pad.bottom} textAnchor="end" className="axis-label">
        0.00
      </text>
    </svg>
  );
}

function FeatureChart({ features }) {
  const max = Math.max(...features.map((item) => item.importance), 0.01);

  return (
    <div className="feature-bars">
      {features.slice(0, 10).map((item) => (
        <div className="feature-row" key={item.feature}>
          <span>{item.feature}</span>
          <div className="feature-track">
            <div style={{ width: `${(item.importance / max) * 100}%` }} />
          </div>
          <b>{item.importance.toFixed(3)}</b>
        </div>
      ))}
    </div>
  );
}

function App() {
  const [ticker, setTicker] = useState("SPY");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const progress = data?.training_progress;
  const projectedChange = data
    ? (data.conclusion.price_target - data.conclusion.last_price) / data.conclusion.last_price
    : 0;
  const priceDomain = useMemo(
    () => (progress ? getDomain(progress.price_series, ["close", "sma50", "sma200"]) : [0, 1]),
    [progress],
  );
  const validationStats = useMemo(() => {
    if (!progress) return { correct: 0, wrong: 0, accuracy: 0 };
    const correct = progress.validation_series.filter((point) => point.correct).length;
    const total = progress.validation_series.length || 1;
    return {
      correct,
      wrong: total - correct,
      accuracy: correct / total,
    };
  }, [progress]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const response = await axios.post(`${API_BASE}/api/predict`, {
        ticker: ticker.trim(),
      });
      if (response.data.status === "success") {
        setData(response.data);
      } else {
        setError(response.data.message || "Error running prediction");
      }
    } catch (err) {
      setError(err.response?.data?.message || err.message || "Error connecting to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">Trading model workspace</span>
          <h1>Quantitative Trend Predictor</h1>
        </div>
        <form onSubmit={handleSubmit} className="search-form">
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="Ticker"
            required
          />
          <button type="submit" disabled={loading}>
            {loading ? "Training..." : "Run Model"}
          </button>
        </form>
      </header>

      <main>
        {error && <div className="error">{error}</div>}

        {loading && (
          <section className="loading-panel">
            <div className="pulse-line" />
            <p>Fetching market history, engineering indicators, and fitting the ensemble.</p>
          </section>
        )}

        {data && (
          <>
            <section className="summary-band">
              <div>
                <span className="label">Ticker</span>
                <strong>{data.conclusion.ticker}</strong>
              </div>
              <div>
                <span className="label">Signal</span>
                <strong className={data.conclusion.trend === "BULLISH" ? "positive" : "negative"}>
                  {data.conclusion.trend}
                </strong>
              </div>
              <div>
                <span className="label">Confidence</span>
                <strong>{formatPercent(data.conclusion.confidence)}</strong>
              </div>
              <div>
                <span className="label">Current</span>
                <strong>{formatMoney(data.conclusion.last_price)}</strong>
              </div>
              <div>
                <span className="label">Target</span>
                <strong>{formatMoney(data.conclusion.price_target)}</strong>
              </div>
              <div>
                <span className="label">Projected Move</span>
                <strong className={projectedChange >= 0 ? "positive" : "negative"}>
                  {projectedChange >= 0 ? "+" : ""}
                  {formatPercent(projectedChange)}
                </strong>
              </div>
              <div>
                <span className="label">Stop</span>
                <strong>{formatMoney(data.conclusion.stop_loss)}</strong>
              </div>
              <div>
                <span className="label">Support</span>
                <strong>{formatOptionalMoney(data.conclusion.support ?? data.conclusion.near_sup)}</strong>
              </div>
              <div>
                <span className="label">Resistance</span>
                <strong>{formatOptionalMoney(data.conclusion.resistance ?? data.conclusion.near_res)}</strong>
              </div>
              <div>
                <span className="label">Time Frame</span>
                <strong>{data.conclusion.est_days} trading days</strong>
              </div>
              <div>
                <span className="label">Target Date</span>
                <strong>{formatDate(data.conclusion.est_end)}</strong>
              </div>
            </section>

            <section className="progress-strip">
              {progress.steps.map((step) => (
                <div className="progress-step" key={step.label}>
                  <span>{step.label}</span>
                  <strong>
                    {step.value.toLocaleString()} <small>{step.unit}</small>
                  </strong>
                </div>
              ))}
            </section>

            <section className="chart-row" aria-label="Training progress charts">
              <article className="chart-panel wide">
                <div className="chart-head">
                  <div>
                    <h2>Training and Validation Price Path</h2>
                    <p>
                      {progress.summary.start_date} to {progress.summary.end_date}
                    </p>
                  </div>
                  <div className="legend">
                    <span className="close">Close</span>
                    <span className="sma50">SMA 50</span>
                    <span className="sma200">SMA 200</span>
                  </div>
                </div>
                <LineChart
                  series={progress.price_series}
                  yDomain={priceDomain}
                  lines={[
                    { key: "close", color: "#004f2d", width: 3.2 },
                    { key: "sma50", color: "#c47f00", width: 2.3 },
                    { key: "sma200", color: "#263238", width: 2.1 },
                  ]}
                />
              </article>

              <article className="chart-panel">
                <div className="chart-head">
                  <div>
                    <h2>Validation Accuracy Trail</h2>
                    <p>
                      {formatPercent(validationStats.accuracy)} sampled accuracy,
                      {" "}
                      {validationStats.correct} correct / {validationStats.wrong} wrong
                    </p>
                  </div>
                  <div className="legend validation-legend">
                    <span className="probability">Bullish probability</span>
                    <span className="threshold-key">50% decision line</span>
                    <span className="correct-key">Correct</span>
                    <span className="wrong-key">Wrong</span>
                  </div>
                </div>
                <LineChart
                  series={progress.validation_series}
                  markers
                  yDomain={[0, 1]}
                  lines={[{ key: "probability", color: "#004f2d", width: 3 }]}
                  threshold={0.5}
                />
              </article>

              <article className="chart-panel compact">
                <div className="chart-head">
                  <div>
                    <h2>Model Scores</h2>
                    <p>Accuracy and AUC by model</p>
                  </div>
                  <div className="legend">
                    <span className="close">Accuracy</span>
                    <span className="sma50">AUC</span>
                  </div>
                </div>
                <MetricsChart metrics={progress.metrics} />
              </article>

              <article className="chart-panel compact">
                <div className="chart-head">
                  <div>
                    <h2>Feature Weight</h2>
                    <p>XGBoost top drivers</p>
                  </div>
                </div>
                <FeatureChart features={progress.feature_importance} />
              </article>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

export default App;

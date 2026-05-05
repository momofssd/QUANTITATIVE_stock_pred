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

function formatOptionalPercent(value) {
  return Number.isFinite(value) ? formatPercent(value) : "--";
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

function PriceIndicatorsChart({ series }) {
  const width = 820;
  const height = 620;
  const pad = { top: 26, right: 34, bottom: 42, left: 62 };
  const plotW = width - pad.left - pad.right;
  const priceTop = pad.top;
  const priceH = 235;
  const rsiTop = priceTop + priceH + 36;
  const rsiH = 88;
  const macdTop = rsiTop + rsiH + 36;
  const macdH = 150;
  const firstDate = series[0]?.date ?? "";
  const lastDate = series.at(-1)?.date ?? "";
  const [minPrice, maxPrice] = getDomain(series, ["close", "sma50", "sma200"]);
  const macdValues = series
    .flatMap((point) => [point.macd, point.macd_signal, point.macd_hist])
    .filter(Number.isFinite);
  const macdMax = Math.max(...macdValues.map((value) => Math.abs(value)), 0.01) * 1.15;
  const splitIndex = series.findIndex((point) => point.phase !== "train");
  const xFor = (index) => pad.left + (series.length <= 1 ? 0 : (index / (series.length - 1)) * plotW);
  const yPrice = (value) => priceTop + ((maxPrice - value) / (maxPrice - minPrice || 1)) * priceH;
  const yRsi = (value) => rsiTop + ((100 - value) / 100) * rsiH;
  const yMacd = (value) => macdTop + ((macdMax - value) / (macdMax * 2 || 1)) * macdH;
  const macdZeroY = yMacd(0);
  const histW = Math.max(plotW / Math.max(series.length, 1) * 0.62, 1.5);
  const formatMacdAxis = (value) => (Math.abs(value) >= 10 ? value.toFixed(0) : value.toFixed(2));

  return (
    <svg className="chart-svg price-indicators-chart" viewBox={`0 0 ${width} ${height}`} role="img">
      <line x1={pad.left} y1={priceTop} x2={pad.left} y2={priceTop + priceH} className="axis" />
      <line x1={pad.left} y1={priceTop + priceH} x2={width - pad.right} y2={priceTop + priceH} className="axis" />
      {[0.33, 0.66].map((tick) => (
        <line
          key={`price-${tick}`}
          x1={pad.left}
          x2={width - pad.right}
          y1={priceTop + priceH * tick}
          y2={priceTop + priceH * tick}
          className="grid"
        />
      ))}
      {splitIndex > 0 && (
        <line
          x1={xFor(splitIndex)}
          x2={xFor(splitIndex)}
          y1={priceTop}
          y2={macdTop + macdH}
          className="split-line"
        />
      )}
      {[
        { key: "close", color: "#004f2d", width: 3.1 },
        { key: "sma50", color: "#c47f00", width: 2.1 },
        { key: "sma200", color: "#263238", width: 2 },
      ].map((line) => (
        <path
          key={line.key}
          d={linePath(series, line.key, xFor, yPrice)}
          fill="none"
          stroke={line.color}
          strokeWidth={line.width}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ))}

      <line x1={pad.left} y1={rsiTop} x2={pad.left} y2={rsiTop + rsiH} className="axis" />
      <line x1={pad.left} y1={rsiTop + rsiH} x2={width - pad.right} y2={rsiTop + rsiH} className="axis" />
      {[70, 30].map((level) => (
        <line
          key={`rsi-${level}`}
          x1={pad.left}
          x2={width - pad.right}
          y1={yRsi(level)}
          y2={yRsi(level)}
          className="threshold soft-threshold"
        />
      ))}
      <path
        d={linePath(series, "rsi", xFor, yRsi)}
        fill="none"
        className="rsi-line"
      />

      <line x1={pad.left} y1={macdTop} x2={pad.left} y2={macdTop + macdH} className="axis" />
      <line x1={pad.left} y1={macdZeroY} x2={width - pad.right} y2={macdZeroY} className="zero-line" />
      {[-0.5, 0.5].map((tick) => (
        <line
          key={`macd-grid-${tick}`}
          x1={pad.left}
          x2={width - pad.right}
          y1={yMacd(macdMax * tick)}
          y2={yMacd(macdMax * tick)}
          className="grid"
        />
      ))}
      {series.map((point, index) => {
        const hist = Number.isFinite(point.macd_hist) ? point.macd_hist : 0;
        const y = yMacd(Math.max(hist, 0));
        const h = Math.max(Math.abs(yMacd(hist) - macdZeroY), 1);
        return (
          <rect
            key={`${point.date}-macd-hist-${index}`}
            x={xFor(index) - histW / 2}
            y={y}
            width={histW}
            height={h}
            className={hist >= 0 ? "macd-hist-positive" : "macd-hist-negative"}
          />
        );
      })}
      <path d={linePath(series, "macd", xFor, yMacd)} fill="none" className="macd-line" />
      <path d={linePath(series, "macd_signal", xFor, yMacd)} fill="none" className="macd-signal-line" />

      <text x={pad.left - 8} y={priceTop + 4} textAnchor="end" className="axis-label">
        {maxPrice.toFixed(0)}
      </text>
      <text x={pad.left - 8} y={priceTop + priceH} textAnchor="end" className="axis-label">
        {minPrice.toFixed(0)}
      </text>
      <text x={pad.left - 8} y={yRsi(70) + 4} textAnchor="end" className="axis-label">
        70
      </text>
      <text x={pad.left - 8} y={yRsi(30) + 4} textAnchor="end" className="axis-label">
        30
      </text>
      <text x={pad.left - 8} y={macdTop + 4} textAnchor="end" className="axis-label">
        {formatMacdAxis(macdMax)}
      </text>
      <text x={pad.left - 8} y={macdZeroY + 4} textAnchor="end" className="axis-label">
        0
      </text>
      <text x={pad.left - 8} y={macdTop + macdH} textAnchor="end" className="axis-label">
        -{formatMacdAxis(macdMax)}
      </text>
      <text x={pad.left} y={height - 10} className="axis-label">
        {firstDate}
      </text>
      <text x={width - pad.right} y={height - 10} textAnchor="end" className="axis-label">
        {lastDate}
      </text>
      <text x={width - pad.right} y={priceTop + 16} textAnchor="end" className="lane-label">
        Price
      </text>
      <text x={width - pad.right} y={rsiTop + 16} textAnchor="end" className="lane-label">
        RSI
      </text>
      <text x={width - pad.right} y={macdTop + 16} textAnchor="end" className="lane-label">
        MACD / Signal
      </text>
    </svg>
  );
}

function ValidationChart({ series, decisionThreshold = 0.5 }) {
  const width = 820;
  const height = 360;
  const pad = { top: 28, right: 34, bottom: 42, left: 62 };
  const plotW = width - pad.left - pad.right;
  const moveTop = pad.top;
  const moveH = 198;
  const probTop = moveTop + moveH + 42;
  const probH = 58;
  const firstDate = series[0]?.date ?? "";
  const lastDate = series.at(-1)?.date ?? "";
  const moves = series.map((point) => point.actual_move_pct).filter(Number.isFinite);
  const maxAbsMove = Math.max(...moves.map((value) => Math.abs(value)), 0.01);
  const moveMax = maxAbsMove * 1.15;
  const xFor = (index) => pad.left + (series.length <= 1 ? 0 : (index / (series.length - 1)) * plotW);
  const yMove = (value) => moveTop + ((moveMax - value) / (moveMax * 2 || 1)) * moveH;
  const yProb = (value) => probTop + ((1 - value) * probH);
  const zeroY = yMove(0);
  const barW = Math.max(plotW / Math.max(series.length, 1) * 0.76, 2);
  const probPath = linePath(series, "probability", xFor, yProb);
  const trianglePoints = (cx, cy, size, direction) => {
    if (direction === "up") {
      return `${cx},${cy - size} ${cx - size},${cy + size} ${cx + size},${cy + size}`;
    }
    return `${cx},${cy + size} ${cx - size},${cy - size} ${cx + size},${cy - size}`;
  };

  return (
    <svg className="chart-svg validation-chart" viewBox={`0 0 ${width} ${height}`} role="img">
      <line x1={pad.left} y1={moveTop} x2={pad.left} y2={moveTop + moveH} className="axis" />
      <line x1={pad.left} y1={zeroY} x2={width - pad.right} y2={zeroY} className="zero-line" />
      {[-0.5, 0.5].map((tick) => (
        <line
          key={tick}
          x1={pad.left}
          x2={width - pad.right}
          y1={yMove(moveMax * tick)}
          y2={yMove(moveMax * tick)}
          className="grid"
        />
      ))}
      {series.map((point, index) => {
        const move = Number.isFinite(point.actual_move_pct) ? point.actual_move_pct : 0;
        const y = yMove(Math.max(move, 0));
        const h = Math.max(Math.abs(yMove(move) - zeroY), 1);
        return (
          <rect
            key={`${point.date}-move-${index}`}
            x={xFor(index) - barW / 2}
            y={y}
            width={barW}
            height={h}
            className={move >= 0 ? "actual-bullish-bar" : "actual-bearish-bar"}
            opacity={point.correct ? 0.78 : 0.35}
          />
        );
      })}
      {series.map((point, index) => {
        if (point.correct) return null;
        const move = Number.isFinite(point.actual_move_pct) ? point.actual_move_pct : 0;
        const actualBullish = point.actual === 1 || move >= 0;
        const y = yMove(move);
        return (
          <polygon
            key={`${point.date}-miss-${index}`}
            points={trianglePoints(xFor(index), y, 6, actualBullish ? "up" : "down")}
            className={actualBullish ? "miss-actual-bullish" : "miss-actual-bearish"}
          />
        );
      })}

      <line x1={pad.left} y1={probTop} x2={pad.left} y2={probTop + probH} className="axis" />
      <line x1={pad.left} y1={probTop + probH} x2={width - pad.right} y2={probTop + probH} className="axis" />
      <line
        x1={pad.left}
        x2={width - pad.right}
        y1={yProb(decisionThreshold)}
        y2={yProb(decisionThreshold)}
        className="threshold"
      />
      <path d={probPath} fill="none" className="probability-line" />
      {series.map((point, index) => {
        const actualBullish = point.actual === 1;
        return point.correct ? (
          <circle
            key={`${point.date}-prob-${index}`}
            cx={xFor(index)}
            cy={yProb(point.probability)}
            r="2.1"
            className="hit-dot"
          />
        ) : (
          <polygon
            key={`${point.date}-prob-${index}`}
            points={trianglePoints(xFor(index), yProb(point.probability), 4.8, actualBullish ? "up" : "down")}
            className={actualBullish ? "miss-actual-bullish" : "miss-actual-bearish"}
          />
        );
      })}

      <text x={pad.left - 8} y={moveTop + 4} textAnchor="end" className="axis-label">
        +{formatPercent(moveMax)}
      </text>
      <text x={pad.left - 8} y={zeroY + 4} textAnchor="end" className="axis-label">
        0%
      </text>
      <text x={pad.left - 8} y={moveTop + moveH} textAnchor="end" className="axis-label">
        -{formatPercent(moveMax)}
      </text>
      <text x={pad.left - 8} y={probTop + 4} textAnchor="end" className="axis-label">
        100%
      </text>
      <text x={pad.left - 8} y={probTop + probH} textAnchor="end" className="axis-label">
        0%
      </text>
      <text x={pad.left} y={height - 10} className="axis-label">
        {firstDate}
      </text>
      <text x={width - pad.right} y={height - 10} textAnchor="end" className="axis-label">
        {lastDate}
      </text>
      <text x={width - pad.right} y={moveTop + 16} textAnchor="end" className="lane-label">
        Actual avg future move
      </text>
      <text x={width - pad.right} y={probTop - 8} textAnchor="end" className="lane-label">
        Bullish probability
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
  const validationStats = useMemo(() => {
    const emptyClassStats = { calls: 0, correct: 0, wrong: 0, accuracy: null };
    if (!progress) {
      return {
        correct: 0,
        wrong: 0,
        accuracy: 0,
        bullish: 0,
        bearish: 0,
        predictedBullish: emptyClassStats,
        predictedBearish: emptyClassStats,
        decisionThreshold: 0.5,
      };
    }
    if (progress.validation_stats) {
      return {
        correct: progress.validation_stats.correct,
        wrong: progress.validation_stats.wrong,
        accuracy: progress.validation_stats.accuracy,
        bullish: progress.validation_stats.actual_bullish,
        bearish: progress.validation_stats.actual_bearish,
        predictedBullish: progress.validation_stats.predicted_bullish ?? emptyClassStats,
        predictedBearish: progress.validation_stats.predicted_bearish ?? emptyClassStats,
        decisionThreshold: progress.validation_stats.decision_threshold ?? 0.5,
      };
    }
    const series = progress.validation_series;
    const correct = series.filter((point) => point.correct).length;
    const bullish = series.filter((point) => point.actual === 1).length;
    const bullishCalls = series.filter((point) => point.prediction === 1);
    const bearishCalls = series.filter((point) => point.prediction === 0);
    const bullishCorrect = bullishCalls.filter((point) => point.correct).length;
    const bearishCorrect = bearishCalls.filter((point) => point.correct).length;
    const total = series.length || 1;
    return {
      correct,
      wrong: total - correct,
      bullish,
      bearish: total - bullish,
      accuracy: correct / total,
      predictedBullish: {
        calls: bullishCalls.length,
        correct: bullishCorrect,
        wrong: bullishCalls.length - bullishCorrect,
        accuracy: bullishCalls.length ? bullishCorrect / bullishCalls.length : null,
      },
      predictedBearish: {
        calls: bearishCalls.length,
        correct: bearishCorrect,
        wrong: bearishCalls.length - bearishCorrect,
        accuracy: bearishCalls.length ? bearishCorrect / bearishCalls.length : null,
      },
      decisionThreshold: 0.5,
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
                    <span className="rsi">RSI</span>
                    <span className="macd">MACD</span>
                    <span className="macd-signal">MACD Signal</span>
                    <span className="macd-hist">Histogram</span>
                  </div>
                </div>
                <PriceIndicatorsChart series={progress.price_series} />
              </article>

              <article className="chart-panel">
                <div className="chart-head">
                  <div>
                    <h2>Validation Accuracy Trail</h2>
                    <p>
                      {formatPercent(validationStats.accuracy)} accuracy,
                      {" "}
                      {validationStats.correct} correct / {validationStats.wrong} wrong,
                      {" "}
                      {validationStats.bullish} bullish / {validationStats.bearish} bearish actuals,
                      {" "}
                      threshold {formatPercent(validationStats.decisionThreshold)}
                    </p>
                  </div>
                  <div className="legend validation-legend">
                    <span className="probability">Bullish probability</span>
                    <span className="threshold-key">Decision threshold</span>
                    <span className="actual-bullish-key">Actual bullish</span>
                    <span className="actual-bearish-key">Actual bearish</span>
                    <span className="missed-bullish-key">Wrong: actual bullish</span>
                    <span className="missed-bearish-key">Wrong: actual bearish</span>
                  </div>
                </div>
                <ValidationChart
                  series={progress.validation_series}
                  decisionThreshold={validationStats.decisionThreshold}
                />
                <div className="validation-call-stats">
                  <div>
                    <span>Bullish predictions</span>
                    <strong>{formatOptionalPercent(validationStats.predictedBullish.accuracy)}</strong>
                    <small>
                      {validationStats.predictedBullish.correct} / {validationStats.predictedBullish.calls} correct
                    </small>
                  </div>
                  <div>
                    <span>Bearish predictions</span>
                    <strong>{formatOptionalPercent(validationStats.predictedBearish.accuracy)}</strong>
                    <small>
                      {validationStats.predictedBearish.correct} / {validationStats.predictedBearish.calls} correct
                    </small>
                  </div>
                </div>
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

import axios from "axios";
import { useMemo, useState } from "react";
import "./App.css";

const API_BASE = "";

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

function formatTickDate(value) {
  if (!value) return "";
  const parts = String(value).split("-").map(Number);
  if (parts.length >= 3 && parts.every(Number.isFinite)) {
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
    }).format(new Date(parts[0], parts[1] - 1, parts[2]));
  }
  return value;
}

function formatAxisMoney(value) {
  if (!Number.isFinite(value)) return "--";
  const absValue = Math.abs(value);
  const digits = absValue >= 100 ? 0 : 2;
  return `$${value.toLocaleString("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })}`;
}

function getDateTicks(series, maxTicks = 6) {
  if (!series.length) return [];
  const tickCount = Math.min(maxTicks, series.length);
  const indexes = new Set();
  for (let i = 0; i < tickCount; i += 1) {
    const index =
      tickCount === 1 ? 0 : Math.round((i * (series.length - 1)) / (tickCount - 1));
    indexes.add(index);
  }
  return [...indexes].sort((a, b) => a - b);
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

function PriceIndicatorsChart({ series }) {
  const width = 1280;
  const height = 430;
  const pad = { top: 24, right: 38, bottom: 38, left: 62 };
  const plotW = width - pad.left - pad.right;
  const priceTop = pad.top;
  const priceH = 165;
  const rsiTop = priceTop + priceH + 30;
  const rsiH = 58;
  const macdTop = rsiTop + rsiH + 30;
  const macdH = 88;
  const firstDate = series[0]?.date ?? "";
  const lastDate = series.at(-1)?.date ?? "";
  const [minPrice, maxPrice] = getDomain(series, ["close", "sma50", "sma200"]);
  const macdValues = series
    .flatMap((point) => [point.macd, point.macd_signal, point.macd_hist])
    .filter(Number.isFinite);
  const macdMax =
    Math.max(...macdValues.map((value) => Math.abs(value)), 0.01) * 1.15;
  const splitIndex = series.findIndex((point) => point.phase !== "train");
  const xFor = (index) =>
    pad.left + (series.length <= 1 ? 0 : (index / (series.length - 1)) * plotW);
  const yPrice = (value) =>
    priceTop + ((maxPrice - value) / (maxPrice - minPrice || 1)) * priceH;
  const yRsi = (value) => rsiTop + ((100 - value) / 100) * rsiH;
  const yMacd = (value) =>
    macdTop + ((macdMax - value) / (macdMax * 2 || 1)) * macdH;
  const macdZeroY = yMacd(0);
  const histW = Math.max((plotW / Math.max(series.length, 1)) * 0.62, 1.5);
  const formatMacdAxis = (value) =>
    Math.abs(value) >= 10 ? value.toFixed(0) : value.toFixed(2);

  return (
    <svg
      className="chart-svg price-indicators-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
    >
      <line
        x1={pad.left}
        y1={priceTop}
        x2={pad.left}
        y2={priceTop + priceH}
        className="axis"
      />
      <line
        x1={pad.left}
        y1={priceTop + priceH}
        x2={width - pad.right}
        y2={priceTop + priceH}
        className="axis"
      />
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

      <line
        x1={pad.left}
        y1={rsiTop}
        x2={pad.left}
        y2={rsiTop + rsiH}
        className="axis"
      />
      <line
        x1={pad.left}
        y1={rsiTop + rsiH}
        x2={width - pad.right}
        y2={rsiTop + rsiH}
        className="axis"
      />
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

      <line
        x1={pad.left}
        y1={macdTop}
        x2={pad.left}
        y2={macdTop + macdH}
        className="axis"
      />
      <line
        x1={pad.left}
        y1={macdZeroY}
        x2={width - pad.right}
        y2={macdZeroY}
        className="zero-line"
      />
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
      <path
        d={linePath(series, "macd", xFor, yMacd)}
        fill="none"
        className="macd-line"
      />
      <path
        d={linePath(series, "macd_signal", xFor, yMacd)}
        fill="none"
        className="macd-signal-line"
      />

      <text
        x={pad.left - 8}
        y={priceTop + 4}
        textAnchor="end"
        className="axis-label"
      >
        {maxPrice.toFixed(0)}
      </text>
      <text
        x={pad.left - 8}
        y={priceTop + priceH}
        textAnchor="end"
        className="axis-label"
      >
        {minPrice.toFixed(0)}
      </text>
      <text
        x={pad.left - 8}
        y={yRsi(70) + 4}
        textAnchor="end"
        className="axis-label"
      >
        70
      </text>
      <text
        x={pad.left - 8}
        y={yRsi(30) + 4}
        textAnchor="end"
        className="axis-label"
      >
        30
      </text>
      <text
        x={pad.left - 8}
        y={macdTop + 4}
        textAnchor="end"
        className="axis-label"
      >
        {formatMacdAxis(macdMax)}
      </text>
      <text
        x={pad.left - 8}
        y={macdZeroY + 4}
        textAnchor="end"
        className="axis-label"
      >
        0
      </text>
      <text
        x={pad.left - 8}
        y={macdTop + macdH}
        textAnchor="end"
        className="axis-label"
      >
        -{formatMacdAxis(macdMax)}
      </text>
      <text x={pad.left} y={height - 10} className="axis-label">
        {firstDate}
      </text>
      <text
        x={width - pad.right}
        y={height - 10}
        textAnchor="end"
        className="axis-label"
      >
        {lastDate}
      </text>
      <text
        x={width - pad.right}
        y={priceTop + 16}
        textAnchor="end"
        className="lane-label"
      >
        Price
      </text>
      <text
        x={width - pad.right}
        y={rsiTop + 16}
        textAnchor="end"
        className="lane-label"
      >
        RSI
      </text>
      <text
        x={width - pad.right}
        y={macdTop + 16}
        textAnchor="end"
        className="lane-label"
      >
        MACD / Signal
      </text>
    </svg>
  );
}

function ValidationChart({ series, decisionThreshold = 0.5 }) {
  const width = 1280;
  const height = 365;
  const pad = { top: 24, right: 86, bottom: 52, left: 70 };
  const preparedSeries = series.map((point) => ({
    ...point,
    actual_price: Number.isFinite(point.actual_price)
      ? point.actual_price
      : point.future_avg_close,
    predicted_price: Number.isFinite(point.predicted_price)
      ? point.predicted_price
      : point.close,
  }));
  const plotW = width - pad.left - pad.right;
  const priceTop = pad.top;
  const priceH = 180;
  const directionTop = priceTop + priceH + 12;
  const directionH = 24;
  const directionMid = directionTop + directionH / 2;
  const probTop = directionTop + directionH + 34;
  const probH = 46;
  const priceValues = preparedSeries
    .flatMap((point) => [point.actual_price, point.predicted_price, point.close])
    .filter(Number.isFinite);
  const rawMinPrice = priceValues.length ? Math.min(...priceValues) : 0;
  const rawMaxPrice = priceValues.length ? Math.max(...priceValues) : 1;
  const priceRange = rawMaxPrice - rawMinPrice || rawMaxPrice || 1;
  const minPrice = Math.max(0, rawMinPrice - priceRange * 0.04);
  const maxPrice = rawMaxPrice + priceRange * 0.06;
  const midPrice = (minPrice + maxPrice) / 2;
  const dateTicks = getDateTicks(preparedSeries, 6);
  const lastPoint = preparedSeries.at(-1);
  const xFor = (index) =>
    pad.left +
    (preparedSeries.length <= 1
      ? 0
      : (index / (preparedSeries.length - 1)) * plotW);
  const yPrice = (value) =>
    priceTop + ((maxPrice - value) / (maxPrice - minPrice || 1)) * priceH;
  const yProb = (value) => probTop + (1 - value) * probH;
  const probPath = linePath(preparedSeries, "probability", xFor, yProb);
  const actualPath = linePath(preparedSeries, "actual_price", xFor, yPrice);
  const predictedPath = linePath(
    preparedSeries,
    "predicted_price",
    xFor,
    yPrice,
  );
  const closePath = linePath(preparedSeries, "close", xFor, yPrice);
  const lastActualY = lastPoint ? yPrice(lastPoint.actual_price) : 0;
  const lastPredictedY = lastPoint ? yPrice(lastPoint.predicted_price) : 0;
  const actualLabelY =
    lastPoint && Math.abs(lastActualY - lastPredictedY) < 20
      ? lastActualY - 10
      : lastActualY + 4;
  const predictedLabelY =
    lastPoint && Math.abs(lastActualY - lastPredictedY) < 20
      ? lastPredictedY + 18
      : lastPredictedY + 4;

  return (
    <svg
      className="chart-svg validation-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
    >
      <line
        x1={pad.left}
        y1={priceTop}
        x2={pad.left}
        y2={priceTop + priceH}
        className="axis"
      />
      <line
        x1={pad.left}
        y1={priceTop + priceH}
        x2={width - pad.right}
        y2={priceTop + priceH}
        className="axis"
      />
      {[maxPrice, midPrice, minPrice].map((tick) => (
        <line
          key={tick}
          x1={pad.left}
          x2={width - pad.right}
          y1={yPrice(tick)}
          y2={yPrice(tick)}
          className="grid"
        />
      ))}
      {dateTicks.map((index) => (
        <line
          key={`${preparedSeries[index]?.date}-date-grid`}
          x1={xFor(index)}
          x2={xFor(index)}
          y1={priceTop}
          y2={probTop + probH}
          className="date-grid"
        />
      ))}
      <path d={closePath} fill="none" className="close-price-line" />
      <path d={actualPath} fill="none" className="actual-price-line" />
      <path d={predictedPath} fill="none" className="predicted-price-line" />
      <line
        x1={pad.left}
        x2={width - pad.right}
        y1={directionTop}
        y2={directionTop}
        className="direction-band-line"
      />
      <line
        x1={pad.left}
        x2={width - pad.right}
        y1={directionTop + directionH}
        y2={directionTop + directionH}
        className="direction-band-line"
      />
      {preparedSeries.map((point, index) => {
        const actualPrice = point.actual_price;
        const predictedPrice = point.predicted_price;
        if (!Number.isFinite(actualPrice) || !Number.isFinite(predictedPrice)) {
          return null;
        }
        const x = xFor(index);
        const actualY = yPrice(actualPrice);
        const tooltip = [
          formatTickDate(point.date),
          `Predicted price: ${formatMoney(predictedPrice)}`,
          `Actual price: ${formatMoney(actualPrice)}`,
          `Current close: ${formatMoney(point.close)}`,
          `Prediction: ${point.prediction_trend}`,
          `Actual: ${point.actual_trend}`,
          `Bullish probability: ${formatPercent(point.probability)}`,
        ].join("\n");
        return (
          <g key={`${point.date}-price-${index}`}>
            <title>{tooltip}</title>
            <line
              x1={x}
              x2={x}
              y1={directionTop + 4}
              y2={directionTop + directionH - 4}
              className={
                point.prediction === 1
                  ? "prediction-bullish-rug"
                  : "prediction-bearish-rug"
              }
            />
            {!point.correct && (
              <circle cx={x} cy={directionMid} r="5.4" className="wrong-ring" />
            )}
            <circle
              cx={x}
              cy={actualY}
              r="6"
              className="validation-hover-target"
            />
          </g>
        );
      })}
      {lastPoint && (
        <>
          <circle
            cx={xFor(preparedSeries.length - 1)}
            cy={lastActualY}
            r="3.8"
            className="actual-end-dot"
          />
          <circle
            cx={xFor(preparedSeries.length - 1)}
            cy={lastPredictedY}
            r="3.4"
            className="predicted-end-dot"
          />
        </>
      )}

      <line
        x1={pad.left}
        y1={probTop}
        x2={pad.left}
        y2={probTop + probH}
        className="axis"
      />
      <line
        x1={pad.left}
        y1={probTop + probH}
        x2={width - pad.right}
        y2={probTop + probH}
        className="axis"
      />
      <line
        x1={pad.left}
        x2={width - pad.right}
        y1={yProb(decisionThreshold)}
        y2={yProb(decisionThreshold)}
        className="threshold"
      />
      <path d={probPath} fill="none" className="probability-line" />

      {[maxPrice, midPrice, minPrice].map((tick) => (
        <text
          key={`${tick}-price-label`}
          x={pad.left - 8}
          y={yPrice(tick) + 4}
          textAnchor="end"
          className="axis-label"
        >
          {formatAxisMoney(tick)}
        </text>
      ))}
      <text
        x={pad.left - 8}
        y={probTop + 4}
        textAnchor="end"
        className="axis-label"
      >
        100%
      </text>
      <text
        x={pad.left - 8}
        y={probTop + probH}
        textAnchor="end"
        className="axis-label"
      >
        0%
      </text>
      {dateTicks.map((index) => (
        <g key={`${preparedSeries[index]?.date}-date-tick`}>
          <line
            x1={xFor(index)}
            x2={xFor(index)}
            y1={probTop + probH}
            y2={probTop + probH + 5}
            className="axis"
          />
          <text
            x={xFor(index)}
            y={height - 18}
            textAnchor="middle"
            className="axis-label"
          >
            {formatTickDate(preparedSeries[index]?.date)}
          </text>
        </g>
      ))}
      <text
        x={pad.left + 8}
        y={priceTop + 16}
        className="lane-label"
      >
        Future avg price
      </text>
      <text
        x={pad.left + 8}
        y={probTop - 8}
        className="lane-label"
      >
        Bullish probability
      </text>
      <text
        x={pad.left + 8}
        y={directionTop - 5}
        className="lane-label"
      >
        Prediction direction
      </text>
      <text
        x={width - pad.right}
        y={directionTop - 5}
        textAnchor="end"
        className="axis-label prediction-direction-label"
      >
        Bullish / Bearish
      </text>
      {lastPoint && (
        <>
          <text
            x={width - pad.right + 8}
            y={actualLabelY}
            className="axis-label actual-price-end-label"
          >
            Actual
          </text>
          <text
            x={width - pad.right + 8}
            y={predictedLabelY}
            className="axis-label predicted-price-end-label"
          >
            Pred
          </text>
        </>
      )}
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
      <line
        x1={pad.left}
        y1={pad.top}
        x2={pad.left}
        y2={height - pad.bottom}
        className="axis"
      />
      <line
        x1={pad.left}
        y1={height - pad.bottom}
        x2={width - pad.right}
        y2={height - pad.bottom}
        className="axis"
      />
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
      {metrics.map((metric, index) => {
        const x = pad.left + index * groupW + groupW / 2;
        const accH = metric.accuracy * plotH;
        const aucH = metric.auc * plotH;
        return (
          <g key={metric.model}>
            <rect
              x={x - barW - 2}
              y={height - pad.bottom - accH}
              width={barW}
              height={accH}
              className="bar-primary"
            />
            <rect
              x={x + 2}
              y={height - pad.bottom - aucH}
              width={barW}
              height={aucH}
              className="bar-secondary"
            />
            <text
              x={x}
              y={height - 18}
              textAnchor="middle"
              className="axis-label"
            >
              {metric.model
                .replace("RandomForest", "RF")
                .replace("GradBoost", "GB")}
            </text>
          </g>
        );
      })}
      <text
        x={pad.left - 8}
        y={pad.top + 4}
        textAnchor="end"
        className="axis-label"
      >
        1.00
      </text>
      <text
        x={pad.left - 8}
        y={height - pad.bottom}
        textAnchor="end"
        className="axis-label"
      >
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
    ? (data.conclusion.price_target - data.conclusion.last_price) /
      data.conclusion.last_price
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
        predictedBullish:
          progress.validation_stats.predicted_bullish ?? emptyClassStats,
        predictedBearish:
          progress.validation_stats.predicted_bearish ?? emptyClassStats,
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
        accuracy: bullishCalls.length
          ? bullishCorrect / bullishCalls.length
          : null,
      },
      predictedBearish: {
        calls: bearishCalls.length,
        correct: bearishCorrect,
        wrong: bearishCalls.length - bearishCorrect,
        accuracy: bearishCalls.length
          ? bearishCorrect / bearishCalls.length
          : null,
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
      setError(
        err.response?.data?.message ||
          err.message ||
          "Error connecting to backend",
      );
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
            <p>
              Fetching market history, engineering indicators, and fitting the
              ensemble.
            </p>
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
                <strong
                  className={
                    data.conclusion.trend === "BULLISH"
                      ? "positive"
                      : "negative"
                  }
                >
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
                <strong
                  className={projectedChange >= 0 ? "positive" : "negative"}
                >
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
                <strong>
                  {formatOptionalMoney(
                    data.conclusion.support ?? data.conclusion.near_sup,
                  )}
                </strong>
              </div>
              <div>
                <span className="label">Resistance</span>
                <strong>
                  {formatOptionalMoney(
                    data.conclusion.resistance ?? data.conclusion.near_res,
                  )}
                </strong>
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

            <section
              className="chart-row"
              aria-label="Training progress charts"
            >
              <article className="chart-panel wide">
                <div className="chart-head">
                  <div>
                    <h2>Training and Validation Price Path</h2>
                    <p>
                      {progress.summary.start_date} to{" "}
                      {progress.summary.end_date}
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

              <article className="chart-panel validation-panel">
                <div className="chart-head">
                  <div>
                    <h2>Validation Accuracy Trail</h2>
                    <p>
                      {formatPercent(validationStats.accuracy)} accuracy,{" "}
                      {validationStats.correct} correct /{" "}
                      {validationStats.wrong} wrong, {validationStats.bullish}{" "}
                      bullish / {validationStats.bearish} bearish actuals,{" "}
                      predicted vs actual future price, threshold{" "}
                      {formatPercent(validationStats.decisionThreshold)}
                    </p>
                  </div>
                  <div className="legend validation-legend">
                    <span className="actual-price-key">Actual price</span>
                    <span className="predicted-price-key">
                      Predicted price
                    </span>
                    <span className="close-price-key">Current close</span>
                    <span className="probability">Bullish probability</span>
                    <span className="pred-bullish-key">Bullish call</span>
                    <span className="pred-bearish-key">Bearish call</span>
                    <span className="threshold-key">Decision threshold</span>
                    <span className="wrong-key">Incorrect prediction</span>
                  </div>
                </div>
                <ValidationChart
                  series={progress.validation_series}
                  decisionThreshold={validationStats.decisionThreshold}
                />
                <div className="validation-call-stats">
                  <div>
                    <span>Bullish predictions</span>
                    <strong>
                      {formatOptionalPercent(
                        validationStats.predictedBullish.accuracy,
                      )}
                    </strong>
                    <small>
                      {validationStats.predictedBullish.correct} /{" "}
                      {validationStats.predictedBullish.calls} correct
                    </small>
                  </div>
                  <div>
                    <span>Bearish predictions</span>
                    <strong>
                      {formatOptionalPercent(
                        validationStats.predictedBearish.accuracy,
                      )}
                    </strong>
                    <small>
                      {validationStats.predictedBearish.correct} /{" "}
                      {validationStats.predictedBearish.calls} correct
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

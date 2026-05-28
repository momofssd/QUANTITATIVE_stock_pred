import { useMemo, useState } from "react";

const WIDTH = 1210;
const HEIGHT = 560;
const LEFT = 18;
const RIGHT = 210;
const TOP = 22;
const PRICE_HEIGHT = 300;
const VOLUME_TOP = 334;
const VOLUME_HEIGHT = 62;
const LOWER_TOP = 425;
const LOWER_HEIGHT = 88;
const DATE_AXIS_Y = LOWER_TOP + LOWER_HEIGHT + 9;

const valid = (number) => typeof number === "number" && Number.isFinite(number);

function pathFor(points, key, xAt, yAt) {
  const entries = points.filter((point) => valid(point[key]));
  return entries
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xAt(points.indexOf(point)).toFixed(2)} ${yAt(point[key]).toFixed(2)}`)
    .join(" ");
}

function compactVolume(number) {
  if (!valid(number)) return "-";
  if (number >= 1_000_000) return `${(number / 1_000_000).toFixed(1)}M`;
  if (number >= 1_000) return `${(number / 1_000).toFixed(1)}K`;
  return number.toFixed(0);
}

function money(number) {
  return valid(number) ? `$${number.toFixed(2)}` : "-";
}

function formatDateTick(dateString, includeYear) {
  const [year, month, day] = dateString.split("-").map(Number);
  const date = new Date(Date.UTC(year, month - 1, day));
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    ...(includeYear ? { year: "numeric" } : {}),
    timeZone: "UTC",
  });
}

function spreadLabels(labels, minimumGap, top, bottom) {
  const arranged = [...labels].sort((left, right) => left.anchorY - right.anchorY);
  arranged.forEach((label, index) => {
    label.labelY = Math.max(label.anchorY, top);
    if (index > 0) {
      label.labelY = Math.max(label.labelY, arranged[index - 1].labelY + minimumGap);
    }
  });
  if (arranged.length && arranged[arranged.length - 1].labelY > bottom) {
    arranged[arranged.length - 1].labelY = bottom;
    for (let index = arranged.length - 2; index >= 0; index -= 1) {
      arranged[index].labelY = Math.min(arranged[index].labelY, arranged[index + 1].labelY - minimumGap);
    }
  }
  return arranged;
}

export default function MarketChart({ points = [], forecast = null, shortTrade = null, zones = null }) {
  const [range, setRange] = useState(120);
  const [lowerStudy, setLowerStudy] = useState("RSI");
  const [activeIndex, setActiveIndex] = useState(null);

  const visiblePoints = useMemo(() => points.slice(-range).filter((point) => valid(point.close)), [points, range]);
  if (!visiblePoints.length) {
    return <div className="empty-chart">Chart history is unavailable for this ticker.</div>;
  }

  const plotWidth = WIDTH - LEFT - RIGHT;
  const forecastPrice = forecast?.possible_price_in_5_trading_days;
  const forecastLow = forecast?.possible_price_range?.low;
  const forecastHigh = forecast?.possible_price_range?.high;
  const hasForecast = [forecastPrice, forecastLow, forecastHigh].every(valid);
  const forecastDirection = forecast?.predicted_direction === "BULLISH" ? "bullish" : "bearish";
  const priceZones = [
    {
      key: "preferred",
      label: "Preferred buy",
      low: zones?.preferred_long_term_buy_zone?.low,
      high: zones?.preferred_long_term_buy_zone?.high,
    },
    {
      key: "tactical",
      label: "Tactical pullback",
      low: zones?.short_term_tactical_buy_zone?.low,
      high: zones?.short_term_tactical_buy_zone?.high,
    },
    {
      key: "trim",
      label: "Trim zone",
      low: zones?.trim_or_sell_zone?.low,
      high: zones?.trim_or_sell_zone?.high,
    },
  ].filter((zone) => [zone.low, zone.high].every(valid));
  const lineZones = [
    { key: "breakout", label: "Breakout", price: zones?.breakout_buy_zone?.trigger_price },
    { key: "invalidation", label: "Invalidation", price: zones?.invalidation_zone?.price },
  ].filter((zone) => valid(zone.price));
  const projectionSlots = hasForecast ? 10 : 0;
  const step = plotWidth / (visiblePoints.length + projectionSlots);
  const candleWidth = Math.max(Math.min(step * 0.62, 9), 2);
  const priceValues = visiblePoints.flatMap((point) =>
    [point.high, point.low, point.ma20, point.ma50, point.ma200].filter(valid),
  );
  if (hasForecast) {
    priceValues.push(forecastPrice, forecastLow, forecastHigh);
  }
  priceZones.forEach((zone) => priceValues.push(zone.low, zone.high));
  lineZones.forEach((zone) => priceValues.push(zone.price));
  const minPrice = Math.min(...priceValues);
  const maxPrice = Math.max(...priceValues);
  const pricePad = Math.max((maxPrice - minPrice) * 0.08, maxPrice * 0.005);
  const priceLow = minPrice - pricePad;
  const priceHigh = maxPrice + pricePad;
  const priceRange = Math.max(priceHigh - priceLow, 0.01);
  const maximumVolume = Math.max(...visiblePoints.map((point) => point.volume || 0), 1);
  const xAt = (index) => LEFT + step * index + step / 2;
  const priceY = (price) => TOP + ((priceHigh - price) / priceRange) * PRICE_HEIGHT;
  const latestX = xAt(visiblePoints.length - 1);
  const projectionStart = latestX + step * 1.25;
  const projectionEnd = WIDTH - RIGHT - 6;
  const overlayLabels = spreadLabels(
    [
      ...priceZones.map((zone) => ({
        ...zone,
        type: "zone",
        anchorY: priceY((zone.low + zone.high) / 2),
        value: `${money(zone.low)} - ${money(zone.high)}`,
      })),
      ...lineZones.map((zone) => ({
        ...zone,
        type: "zone",
        anchorY: priceY(zone.price),
        value: money(zone.price),
      })),
      ...(hasForecast
        ? [
            {
              key: "five-day",
              type: "forecast",
              label: "5D close",
              anchorY: priceY(forecastPrice),
              value: money(forecastPrice),
            },
          ]
        : []),
    ],
    17,
    TOP + 12,
    TOP + PRICE_HEIGHT - 6,
  );
  const hovered = visiblePoints[activeIndex ?? visiblePoints.length - 1];
  const macdValues = visiblePoints.flatMap((point) => [point.macd, point.macd_signal, point.macd_hist].filter(valid));
  const macdBound = Math.max(...macdValues.map((number) => Math.abs(number)), 0.001);
  const lowerY =
    lowerStudy === "RSI"
      ? (value) => LOWER_TOP + ((100 - value) / 100) * LOWER_HEIGHT
      : (value) => LOWER_TOP + ((macdBound - value) / (macdBound * 2)) * LOWER_HEIGHT;
  const gridPrices = Array.from({ length: 5 }, (_, index) => priceHigh - (priceRange * index) / 4);
  const dateTickCount = range === 60 ? 6 : range === 120 ? 7 : 8;
  const dateTickIndexes = Array.from(
    new Set(
      Array.from({ length: dateTickCount }, (_, index) =>
        Math.round((index * (visiblePoints.length - 1)) / (dateTickCount - 1)),
      ),
    ),
  );
  const firstVisibleYear = visiblePoints[0].date.slice(0, 4);

  function handlePointerMove(event) {
    const bounds = event.currentTarget.getBoundingClientRect();
    const relativeX = ((event.clientX - bounds.left) / bounds.width) * WIDTH;
    const index = Math.max(0, Math.min(visiblePoints.length - 1, Math.floor((relativeX - LEFT) / step)));
    setActiveIndex(index);
  }

  return (
    <section className="chart-card">
      <div className="chart-toolbar">
        <div>
          <h2>Daily Price Chart</h2>
          <p>Adjusted candlesticks with moving averages and technical studies</p>
        </div>
        <div className="toolbar-controls">
          {[60, 120, 180].map((period) => (
            <button
              className={range === period ? "selected" : ""}
              key={period}
              onClick={() => setRange(period)}
              type="button"
            >
              {period === 60 ? "3M" : period === 120 ? "6M" : "9M"}
            </button>
          ))}
          {["RSI", "MACD"].map((study) => (
            <button
              className={lowerStudy === study ? "selected study" : "study"}
              key={study}
              onClick={() => setLowerStudy(study)}
              type="button"
            >
              {study}
            </button>
          ))}
        </div>
      </div>
      <div className="chart-legend">
        <span className="legend-item ma20">MA20</span>
        <span className="legend-item ma50">MA50</span>
        <span className="legend-item ma200">MA200</span>
        {hasForecast && (
          <span className={`legend-item forecast ${forecastDirection}`}>
            5D close {money(forecastPrice)} | range {money(forecastLow)} - {money(forecastHigh)}
          </span>
        )}
        {priceZones.map((zone) => (
          <span className={`legend-item zone ${zone.key}`} key={zone.key}>
            {zone.label}
          </span>
        ))}
        {lineZones.map((zone) => (
          <span className={`legend-item zone ${zone.key}`} key={zone.key}>
            {zone.label}
          </span>
        ))}
        {shortTrade && (
          <span className={`short-trade-chart-status ${shortTrade.is_good_short_term_buy_now ? "buy" : "avoid"}`}>
            Short-term buy now: {shortTrade.is_good_short_term_buy_now ? "YES" : "NO"} |{" "}
            {shortTrade.action.replaceAll("_", " ")}
          </span>
        )}
        <span className="quote-detail">{hovered.date}</span>
        <span>O {money(hovered.open)}</span>
        <span>H {money(hovered.high)}</span>
        <span>L {money(hovered.low)}</span>
        <span>C {money(hovered.close)}</span>
        <span>Vol {compactVolume(hovered.volume)}</span>
      </div>
      <svg
        className="market-chart"
        onMouseLeave={() => setActiveIndex(null)}
        onMouseMove={handlePointerMove}
        preserveAspectRatio="none"
        role="img"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      >
        {gridPrices.map((price) => (
          <g key={price}>
            <line className="grid-line" x1={LEFT} x2={WIDTH - RIGHT} y1={priceY(price)} y2={priceY(price)} />
            <text className="axis-label" x={WIDTH - RIGHT + 8} y={priceY(price) + 4}>
              {price.toFixed(2)}
            </text>
          </g>
        ))}
        {priceZones.map((zone) => (
          <g className={`price-zone-overlay ${zone.key}`} key={`band-${zone.key}`}>
            <rect
              className="price-zone-band"
              height={Math.max(priceY(zone.low) - priceY(zone.high), 2)}
              width={WIDTH - LEFT - RIGHT}
              x={LEFT}
              y={priceY(zone.high)}
            />
          </g>
        ))}
        {lineZones.map((zone) => (
          <g className={`price-zone-overlay ${zone.key}`} key={`line-${zone.key}`}>
            <line
              className="price-zone-line"
              x1={LEFT}
              x2={WIDTH - RIGHT}
              y1={priceY(zone.price)}
              y2={priceY(zone.price)}
            />
          </g>
        ))}
        {overlayLabels.map((zone) => (
          <g
            className={zone.type === "forecast" ? `forecast-overlay ${forecastDirection}` : `price-zone-overlay ${zone.key}`}
            key={`label-${zone.key}`}
          >
            <line
              className={zone.type === "forecast" ? "forecast-leader" : "price-zone-leader"}
              x1={WIDTH - RIGHT - 24}
              x2={WIDTH - RIGHT + 48}
              y1={zone.anchorY}
              y2={zone.labelY}
            />
            <text
              className={zone.type === "forecast" ? "forecast-price-label" : "price-zone-label"}
              x={WIDTH - RIGHT + 54}
              y={zone.labelY + 4}
            >
              {zone.label} {zone.value}
            </text>
          </g>
        ))}
        {hasForecast && (
          <g className={`forecast-overlay ${forecastDirection}`}>
            <rect
              className="forecast-band"
              height={Math.max(priceY(forecastLow) - priceY(forecastHigh), 2)}
              width={projectionEnd - projectionStart}
              x={projectionStart}
              y={priceY(forecastHigh)}
            />
            <line
              className="forecast-connector"
              x1={latestX}
              x2={projectionStart}
              y1={priceY(visiblePoints[visiblePoints.length - 1].close)}
              y2={priceY(forecastPrice)}
            />
            <line
              className="forecast-target"
              x1={projectionStart}
              x2={projectionEnd}
              y1={priceY(forecastPrice)}
              y2={priceY(forecastPrice)}
            />
          </g>
        )}
        {visiblePoints.map((point, index) => {
          const open = valid(point.open) ? point.open : point.close;
          const high = valid(point.high) ? point.high : point.close;
          const low = valid(point.low) ? point.low : point.close;
          const rising = point.close >= open;
          const colorClass = rising ? "up" : "down";
          const bodyY = priceY(Math.max(open, point.close));
          const bodyHeight = Math.max(Math.abs(priceY(open) - priceY(point.close)), 1);
          return (
            <g className={colorClass} key={point.date}>
              <line className="wick" x1={xAt(index)} x2={xAt(index)} y1={priceY(high)} y2={priceY(low)} />
              <rect
                className="candle"
                height={bodyHeight}
                width={candleWidth}
                x={xAt(index) - candleWidth / 2}
                y={bodyY}
              />
              <rect
                className="volume"
                height={(point.volume / maximumVolume) * VOLUME_HEIGHT}
                width={candleWidth}
                x={xAt(index) - candleWidth / 2}
                y={VOLUME_TOP + VOLUME_HEIGHT - (point.volume / maximumVolume) * VOLUME_HEIGHT}
              />
            </g>
          );
        })}
        <path className="moving-average line-ma20" d={pathFor(visiblePoints, "ma20", xAt, priceY)} />
        <path className="moving-average line-ma50" d={pathFor(visiblePoints, "ma50", xAt, priceY)} />
        <path className="moving-average line-ma200" d={pathFor(visiblePoints, "ma200", xAt, priceY)} />
        <line className="pane-separator" x1={LEFT} x2={WIDTH - RIGHT} y1={VOLUME_TOP - 10} y2={VOLUME_TOP - 10} />
        <text className="pane-label" x={LEFT} y={VOLUME_TOP - 16}>Volume</text>
        <line className="pane-separator" x1={LEFT} x2={WIDTH - RIGHT} y1={LOWER_TOP - 12} y2={LOWER_TOP - 12} />
        <text className="pane-label" x={LEFT} y={LOWER_TOP - 18}>{lowerStudy}</text>
        {lowerStudy === "RSI" ? (
          <>
            {[30, 70].map((level) => (
              <g key={level}>
                <line className="study-boundary" x1={LEFT} x2={WIDTH - RIGHT} y1={lowerY(level)} y2={lowerY(level)} />
                <text className="axis-label" x={WIDTH - RIGHT + 8} y={lowerY(level) + 4}>{level}</text>
              </g>
            ))}
            <path className="study-line rsi" d={pathFor(visiblePoints, "rsi_14", xAt, lowerY)} />
          </>
        ) : (
          <>
            <line className="study-boundary" x1={LEFT} x2={WIDTH - RIGHT} y1={lowerY(0)} y2={lowerY(0)} />
            {visiblePoints.map((point, index) => (
              <rect
                className={point.macd_hist >= 0 ? "histogram up" : "histogram down"}
                height={Math.abs(lowerY(point.macd_hist || 0) - lowerY(0))}
                key={`macd-${point.date}`}
                width={candleWidth}
                x={xAt(index) - candleWidth / 2}
                y={Math.min(lowerY(point.macd_hist || 0), lowerY(0))}
              />
            ))}
            <path className="study-line macd" d={pathFor(visiblePoints, "macd", xAt, lowerY)} />
            <path className="study-line signal" d={pathFor(visiblePoints, "macd_signal", xAt, lowerY)} />
          </>
        )}
        {activeIndex !== null && (
          <line className="crosshair" x1={xAt(activeIndex)} x2={xAt(activeIndex)} y1={TOP} y2={LOWER_TOP + LOWER_HEIGHT} />
        )}
        <line className="date-axis-line" x1={LEFT} x2={WIDTH - RIGHT} y1={DATE_AXIS_Y} y2={DATE_AXIS_Y} />
        {dateTickIndexes.map((index) => {
          const point = visiblePoints[index];
          const includeYear = index === 0 || point.date.slice(0, 4) !== firstVisibleYear;
          return (
            <g className="date-tick" key={`date-${point.date}`}>
              <line x1={xAt(index)} x2={xAt(index)} y1={DATE_AXIS_Y} y2={DATE_AXIS_Y + 7} />
              <text className="axis-label date" x={xAt(index)} y={DATE_AXIS_Y + 23}>
                {formatDateTick(point.date, includeYear)}
              </text>
            </g>
          );
        })}
      </svg>
    </section>
  );
}

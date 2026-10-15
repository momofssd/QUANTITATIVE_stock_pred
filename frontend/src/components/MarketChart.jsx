import { useMemo, useState } from "react";

const WIDTH = 1210;
const HEIGHT = 604;
const LEFT = 18;
const RIGHT = 214;
const TOP = 22;
const PRICE_HEIGHT = 322;
const VOLUME_TOP = 364;
const VOLUME_HEIGHT = 54;
const LOWER_TOP = 450;
const LOWER_HEIGHT = 96;
const DATE_AXIS_Y = LOWER_TOP + LOWER_HEIGHT + 9;
const NEXT_SLOTS = 6;
const RANGES = [
  { sessions: 63, label: "3M" },
  { sessions: 126, label: "6M" },
  { sessions: 252, label: "1Y" },
];

const ZONES = [
  { key: "st_buy", label: "ST buy", kind: "buy", term: "short" },
  { key: "st_sell", label: "ST take-profit", kind: "sell", term: "short" },
  { key: "lt_buy", label: "LT value", kind: "buy", term: "long" },
  { key: "lt_sell", label: "LT trim", kind: "sell", term: "long" },
];

function outcomeText(zone, outcome) {
  if (outcome === "success") return zone === "st_buy" ? "bounced 1.5 ATR before dropping 1.5 ATR" : "pulled back 1.5 ATR before extending 1.5 ATR";
  if (outcome === "failed") return zone === "st_buy" ? "dropped 1.5 ATR first" : "extended 1.5 ATR higher first";
  if (outcome === "expired") return "no decision within 10 sessions";
  return "still open";
}

const valid = (number) => typeof number === "number" && Number.isFinite(number);
const money = (number) => (valid(number) ? `$${number.toFixed(2)}` : "-");

function compactVolume(number) {
  if (!valid(number)) return "-";
  if (number >= 1_000_000) return `${(number / 1_000_000).toFixed(1)}M`;
  if (number >= 1_000) return `${(number / 1_000).toFixed(1)}K`;
  return number.toFixed(0);
}

function formatDateTick(dateString, includeYear) {
  const [year, month, day] = dateString.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day)).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    ...(includeYear ? { year: "numeric" } : {}),
    timeZone: "UTC",
  });
}

function linePath(points, key, xAt, yAt) {
  let path = "";
  let drawing = false;
  points.forEach((point, index) => {
    if (!valid(point[key])) {
      drawing = false;
      return;
    }
    path += `${drawing ? "L" : "M"} ${xAt(index).toFixed(2)} ${yAt(point[key]).toFixed(2)} `;
    drawing = true;
  });
  return path;
}

// Closed area between two keys, split wherever either edge is missing.
function bandPath(points, lowKey, highKey, xAt, yAt) {
  const segments = [];
  let current = [];
  points.forEach((point, index) => {
    if (valid(point[lowKey]) && valid(point[highKey])) {
      current.push(index);
    } else if (current.length) {
      segments.push(current);
      current = [];
    }
  });
  if (current.length) segments.push(current);
  return segments
    .map((segment) => {
      const top = segment.map((index, order) => `${order ? "L" : "M"} ${xAt(index).toFixed(2)} ${yAt(points[index][highKey]).toFixed(2)}`);
      const bottom = [...segment].reverse().map((index) => `L ${xAt(index).toFixed(2)} ${yAt(points[index][lowKey]).toFixed(2)}`);
      return `${top.join(" ")} ${bottom.join(" ")} Z`;
    })
    .join(" ");
}

function spreadLabels(labels, minimumGap, top, bottom) {
  const arranged = [...labels].sort((left, right) => left.anchorY - right.anchorY);
  arranged.forEach((label, index) => {
    label.labelY = Math.max(label.anchorY, top);
    if (index > 0) label.labelY = Math.max(label.labelY, arranged[index - 1].labelY + minimumGap);
  });
  if (arranged.length && arranged[arranged.length - 1].labelY > bottom) {
    arranged[arranged.length - 1].labelY = bottom;
    for (let index = arranged.length - 2; index >= 0; index -= 1) {
      arranged[index].labelY = Math.min(arranged[index].labelY, arranged[index + 1].labelY - minimumGap);
    }
  }
  return arranged;
}

function Triangle({ x, y, up, className, title }) {
  const size = 5;
  const points = up
    ? `${x},${y - size} ${x - size},${y + size} ${x + size},${y + size}`
    : `${x},${y + size} ${x - size},${y - size} ${x + size},${y - size}`;
  return (
    <polygon className={className} points={points}>
      <title>{title}</title>
    </polygon>
  );
}

export default function MarketChart({ chart, shortTerm, longTerm }) {
  const [range, setRange] = useState(126);
  const [lowerStudy, setLowerStudy] = useState("STRETCH");
  const [layers, setLayers] = useState({ short: true, long: true, touches: true });
  const [activeIndex, setActiveIndex] = useState(null);
  const points = chart?.points || [];

  const visiblePoints = useMemo(() => points.slice(-range).filter((point) => valid(point.close)), [points, range]);
  const touchesByDate = useMemo(() => {
    const map = new Map();
    (chart?.zone_touches || []).forEach((touch) => {
      if (touch.zone.startsWith("st_")) map.set(`${touch.date}:${touch.zone}`, touch);
    });
    return map;
  }, [chart]);

  if (!visiblePoints.length) {
    return <div className="empty-chart">Chart history is unavailable for this ticker.</div>;
  }

  const nextZones = {
    st_buy: shortTerm?.buy_zone,
    st_sell: shortTerm?.sell_zone,
    lt_buy: longTerm?.available ? longTerm.buy_zone : null,
    lt_sell: longTerm?.available ? longTerm.sell_zone : null,
  };
  const shownZones = ZONES.filter((zone) => layers[zone.term]);
  const plotRight = WIDTH - RIGHT;
  const step = (plotRight - LEFT) / (visiblePoints.length + NEXT_SLOTS);
  const candleWidth = Math.max(Math.min(step * 0.62, 9), 2);
  const xAt = (index) => LEFT + step * index + step / 2;
  const latestIndex = visiblePoints.length - 1;
  const latest = visiblePoints[latestIndex];
  const nextStart = xAt(latestIndex) + step / 2;

  const priceValues = visiblePoints.flatMap((point) => [point.high, point.low, point.ema20, point.ma50, point.ma200].filter(valid));
  // Long-term zones can sit far from price; they are clipped to the pane instead of stretching the scale.
  shownZones.filter((zone) => zone.term === "short").forEach((zone) => {
    visiblePoints.forEach((point) => priceValues.push(...[point[`${zone.key}_low`], point[`${zone.key}_high`]].filter(valid)));
    const next = nextZones[zone.key];
    if (next) priceValues.push(...[next.low, next.high].filter(valid));
  });
  const minPrice = Math.min(...priceValues);
  const maxPrice = Math.max(...priceValues);
  const pricePad = Math.max((maxPrice - minPrice) * 0.05, maxPrice * 0.004);
  const priceLow = minPrice - pricePad;
  const priceHigh = maxPrice + pricePad;
  const priceRange = Math.max(priceHigh - priceLow, 0.01);
  const priceY = (price) => TOP + ((priceHigh - price) / priceRange) * PRICE_HEIGHT;
  const maximumVolume = Math.max(...visiblePoints.map((point) => point.volume || 0), 1);
  const gridPrices = Array.from({ length: 5 }, (_, index) => priceHigh - (priceRange * index) / 4);

  const zoneLabels = spreadLabels(
    [
      ...shownZones
        .filter((zone) => nextZones[zone.key] && valid(nextZones[zone.key].low) && valid(nextZones[zone.key].high))
        .map((zone) => {
          const next = nextZones[zone.key];
          const above = next.low > priceHigh;
          const below = next.high < priceLow;
          const anchor = Math.min(Math.max((next.low + next.high) / 2, priceLow), priceHigh);
          return {
            ...zone,
            anchorY: priceY(anchor),
            text: `${above ? "↑ " : below ? "↓ " : ""}${zone.label} ${money(next.low)}–${money(next.high)}`,
          };
        }),
      { key: "last", kind: "last", anchorY: priceY(latest.close), text: `Last ${money(latest.close)}` },
    ],
    16,
    TOP + 8,
    TOP + PRICE_HEIGHT - 4,
  );

  let lowerY;
  let lowerTicks = [];
  if (lowerStudy === "STRETCH") {
    const values = visiblePoints
      .flatMap((point) => [point.stretch, point.stretch_p05, point.stretch_p95])
      .filter(valid)
      .map((value) => Math.max(-6, Math.min(6, value)));
    const bound = Math.max(...values.map(Math.abs), 1);
    lowerY = (value) => LOWER_TOP + ((bound - Math.max(-bound, Math.min(bound, value))) / (bound * 2)) * LOWER_HEIGHT;
    lowerTicks = [-Math.floor(bound), 0, Math.floor(bound)].filter((value, index, list) => list.indexOf(value) === index);
  } else if (lowerStudy === "RSI") {
    lowerY = (value) => LOWER_TOP + ((100 - value) / 100) * LOWER_HEIGHT;
    lowerTicks = [30, 70];
  } else {
    const macdValues = visiblePoints.flatMap((point) => [point.macd, point.macd_signal, point.macd_hist].filter(valid));
    const bound = Math.max(...macdValues.map(Math.abs), 0.001);
    lowerY = (value) => LOWER_TOP + ((bound - value) / (bound * 2)) * LOWER_HEIGHT;
  }

  const dateTickCount = range <= 63 ? 6 : range <= 126 ? 7 : 9;
  const dateTickIndexes = Array.from(
    new Set(Array.from({ length: dateTickCount }, (_, index) => Math.round((index * latestIndex) / (dateTickCount - 1)))),
  );
  const firstVisibleYear = visiblePoints[0].date.slice(0, 4);
  const hovered = visiblePoints[activeIndex ?? latestIndex];
  const hoveredTouches = ["st_buy", "st_sell"].map((zone) => touchesByDate.get(`${hovered.date}:${zone}`)).filter(Boolean);

  function handlePointerMove(event) {
    const bounds = event.currentTarget.getBoundingClientRect();
    const relativeX = ((event.clientX - bounds.left) / bounds.width) * WIDTH;
    setActiveIndex(Math.max(0, Math.min(latestIndex, Math.floor((relativeX - LEFT) / step))));
  }

  function toggle(layer) {
    setLayers((current) => ({ ...current, [layer]: !current[layer] }));
  }

  return (
    <section className="chart-card">
      <div className="chart-toolbar">
        <div>
          <h2>Price and entry zones</h2>
          <p>
            Zones move with the trend. The block at the right edge is next session's zones; ↑/↓ marks a long-term zone outside
            the visible range.
          </p>
        </div>
        <div className="toolbar-controls">
          {RANGES.map((option) => (
            <button
              className={range === option.sessions ? "selected" : ""}
              key={option.sessions}
              onClick={() => setRange(option.sessions)}
              type="button"
            >
              {option.label}
            </button>
          ))}
          {[
            ["STRETCH", "Stretch"],
            ["RSI", "RSI"],
            ["MACD", "MACD"],
          ].map(([study, label]) => (
            <button
              className={lowerStudy === study ? "selected study" : "study"}
              key={study}
              onClick={() => setLowerStudy(study)}
              type="button"
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="chart-legend">
        <label className="layer-toggle">
          <input checked={layers.short} onChange={() => toggle("short")} type="checkbox" />
          <span className="swatch zone-short" /> Short-term zones
        </label>
        <label className="layer-toggle">
          <input checked={layers.long} onChange={() => toggle("long")} type="checkbox" />
          <span className="swatch zone-long" /> Long-term zones
        </label>
        <label className="layer-toggle">
          <input checked={layers.touches} onChange={() => toggle("touches")} type="checkbox" />
          <span className="swatch touch" /> Past touches
        </label>
        <span className="legend-item ma20">EMA20</span>
        <span className="legend-item ma50">MA50</span>
        <span className="legend-item ma200">MA200</span>
        <span className="quote-detail">{hovered.date}</span>
        <span>O {money(hovered.open)}</span>
        <span>H {money(hovered.high)}</span>
        <span>L {money(hovered.low)}</span>
        <span>C {money(hovered.close)}</span>
        <span>Vol {compactVolume(hovered.volume)}</span>
        {valid(hovered.st_buy_low) && (
          <span>
            Buy zone {money(hovered.st_buy_low)}–{money(hovered.st_buy_high)}
          </span>
        )}
        {hoveredTouches.map((touch) => (
          <span className={`touch-note ${touch.zone}`} key={touch.zone}>
            {touch.zone === "st_buy" ? "Buy-zone touch" : "Sell-zone touch"}: {outcomeText(touch.zone, touch.outcome)}
          </span>
        ))}
      </div>
      <svg
        aria-label="Daily candlestick chart with short-term and long-term buy and sell zones"
        className="market-chart"
        onMouseLeave={() => setActiveIndex(null)}
        onMouseMove={handlePointerMove}
        preserveAspectRatio="none"
        role="img"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      >
        <defs>
          <clipPath id="price-pane">
            <rect height={PRICE_HEIGHT} width={plotRight - LEFT} x={LEFT} y={TOP} />
          </clipPath>
          {["buy", "sell"].map((kind) => (
            <pattern height="7" id={`hatch-${kind}`} key={kind} patternTransform="rotate(45)" patternUnits="userSpaceOnUse" width="7">
              <line className={`hatch-line ${kind}`} x1="0" x2="0" y1="0" y2="7" />
            </pattern>
          ))}
        </defs>
        {gridPrices.map((price) => (
          <g key={price}>
            <line className="grid-line" x1={LEFT} x2={plotRight} y1={priceY(price)} y2={priceY(price)} />
            <text className="axis-label" x={plotRight + 8} y={priceY(price) + 4}>
              {price.toFixed(2)}
            </text>
          </g>
        ))}
        <rect className="next-session-bg" height={PRICE_HEIGHT} width={plotRight - nextStart} x={nextStart} y={TOP} />
        <text className="pane-label" textAnchor="end" x={plotRight - 4} y={TOP + PRICE_HEIGHT - 6}>
          Next session
        </text>
        {shownZones.map((zone) => {
          const next = nextZones[zone.key];
          return (
            <g className={`zone-layer ${zone.kind} ${zone.term}`} clipPath="url(#price-pane)" key={zone.key}>
              <path className="zone-fill" d={bandPath(visiblePoints, `${zone.key}_low`, `${zone.key}_high`, xAt, priceY)} />
              <path className="zone-edge" d={linePath(visiblePoints, `${zone.key}_low`, xAt, priceY)} />
              <path className="zone-edge" d={linePath(visiblePoints, `${zone.key}_high`, xAt, priceY)} />
              {next && valid(next.low) && valid(next.high) && (
                <rect
                  className="zone-fill next"
                  height={Math.max(priceY(next.low) - priceY(next.high), 2)}
                  width={plotRight - nextStart}
                  x={nextStart}
                  y={priceY(next.high)}
                />
              )}
            </g>
          );
        })}
        {visiblePoints.map((point, index) => {
          const open = valid(point.open) ? point.open : point.close;
          const high = valid(point.high) ? point.high : point.close;
          const low = valid(point.low) ? point.low : point.close;
          const bodyY = priceY(Math.max(open, point.close));
          const bodyHeight = Math.max(Math.abs(priceY(open) - priceY(point.close)), 1);
          const volumeHeight = ((point.volume || 0) / maximumVolume) * VOLUME_HEIGHT;
          return (
            <g className={point.close >= open ? "up" : "down"} key={point.date}>
              <line className="wick" x1={xAt(index)} x2={xAt(index)} y1={priceY(high)} y2={priceY(low)} />
              <rect className="candle" height={bodyHeight} width={candleWidth} x={xAt(index) - candleWidth / 2} y={bodyY} />
              <rect
                className="volume"
                height={volumeHeight}
                width={candleWidth}
                x={xAt(index) - candleWidth / 2}
                y={VOLUME_TOP + VOLUME_HEIGHT - volumeHeight}
              />
            </g>
          );
        })}
        <path className="moving-average line-ma20" d={linePath(visiblePoints, "ema20", xAt, priceY)} />
        <path className="moving-average line-ma50" d={linePath(visiblePoints, "ma50", xAt, priceY)} />
        <path className="moving-average line-ma200" d={linePath(visiblePoints, "ma200", xAt, priceY)} />
        {layers.touches &&
          visiblePoints.map((point, index) =>
            ["st_buy", "st_sell"].map((zone) => {
              const touch = touchesByDate.get(`${point.date}:${zone}`);
              if (!touch) return null;
              const buy = zone === "st_buy";
              const y = buy ? priceY(point.low) + 10 : priceY(point.high) - 10;
              return (
                <Triangle
                  className={`touch-marker ${buy ? "buy" : "sell"} ${touch.outcome}`}
                  key={`${zone}-${point.date}`}
                  title={`${point.date} ${buy ? "buy" : "sell"}-zone touch at ${money(touch.entry)}: ${outcomeText(touch.zone, touch.outcome)}`}
                  up={buy}
                  x={xAt(index)}
                  y={y}
                />
              );
            }),
          )}
        {zoneLabels.map((label) => (
          <g className={label.kind === "last" ? "last-price" : `zone-label-group ${label.kind}`} key={`label-${label.key}`}>
            <line className="label-leader" x1={plotRight} x2={plotRight + 54} y1={label.anchorY} y2={label.labelY} />
            <text className="zone-label" x={plotRight + 58} y={label.labelY + 4}>
              {label.text}
            </text>
          </g>
        ))}
        <line className="last-price-line" x1={LEFT} x2={plotRight} y1={priceY(latest.close)} y2={priceY(latest.close)} />

        <line className="pane-separator" x1={LEFT} x2={plotRight} y1={VOLUME_TOP - 10} y2={VOLUME_TOP - 10} />
        <text className="pane-label" x={LEFT} y={VOLUME_TOP - 16}>
          Volume
        </text>
        <line className="pane-separator" x1={LEFT} x2={plotRight} y1={LOWER_TOP - 12} y2={LOWER_TOP - 12} />
        <text className="pane-label" x={LEFT} y={LOWER_TOP - 18}>
          {lowerStudy === "STRETCH" ? "Stretch from EMA20 (ATRs) · shaded = zone thresholds" : lowerStudy}
        </text>
        {lowerStudy === "STRETCH" && (
          <>
            <path className="zone-layer buy short stretch-band" d={bandPath(visiblePoints, "stretch_p05", "stretch_p20", xAt, lowerY)} />
            <path className="zone-layer sell short stretch-band" d={bandPath(visiblePoints, "stretch_p80", "stretch_p95", xAt, lowerY)} />
            {lowerTicks.map((level) => (
              <g key={level}>
                <line className="study-boundary" x1={LEFT} x2={plotRight} y1={lowerY(level)} y2={lowerY(level)} />
                <text className="axis-label" x={plotRight + 8} y={lowerY(level) + 4}>
                  {level > 0 ? `+${level}` : level}
                </text>
              </g>
            ))}
            <path className="study-line stretch" d={linePath(visiblePoints, "stretch", xAt, lowerY)} />
          </>
        )}
        {lowerStudy === "RSI" && (
          <>
            {lowerTicks.map((level) => (
              <g key={level}>
                <line className="study-boundary" x1={LEFT} x2={plotRight} y1={lowerY(level)} y2={lowerY(level)} />
                <text className="axis-label" x={plotRight + 8} y={lowerY(level) + 4}>
                  {level}
                </text>
              </g>
            ))}
            <path className="study-line rsi" d={linePath(visiblePoints, "rsi_14", xAt, lowerY)} />
          </>
        )}
        {lowerStudy === "MACD" && (
          <>
            <line className="study-boundary" x1={LEFT} x2={plotRight} y1={lowerY(0)} y2={lowerY(0)} />
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
            <path className="study-line macd" d={linePath(visiblePoints, "macd", xAt, lowerY)} />
            <path className="study-line signal" d={linePath(visiblePoints, "macd_signal", xAt, lowerY)} />
          </>
        )}
        {activeIndex !== null && (
          <line className="crosshair" x1={xAt(activeIndex)} x2={xAt(activeIndex)} y1={TOP} y2={LOWER_TOP + LOWER_HEIGHT} />
        )}
        <line className="date-axis-line" x1={LEFT} x2={plotRight} y1={DATE_AXIS_Y} y2={DATE_AXIS_Y} />
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
      <p className="chart-footnote">
        ▲ below a candle = first touch of the short-term buy zone; ▼ above = first touch of the take-profit zone. Filled = price
        moved 1.5 ATR in the expected direction before 1.5 ATR against; hollow = it did not. Hover a marker for details.
      </p>
    </section>
  );
}

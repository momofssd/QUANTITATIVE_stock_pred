import { useEffect, useMemo, useRef, useState } from "react";

const WIDTH = 1210;
const LEFT = 18;
const RIGHT = 214;
const TOP = 22;
const VOLUME_HEIGHT = 54;
const LOWER_HEIGHT = 96;
const NEXT_SLOTS = 6;
// The price pane is resizable by dragging the handle under the chart; the panes below keep their height.
const DEFAULT_PRICE_HEIGHT = 322;
const MIN_PRICE_HEIGHT = 180;
const MAX_PRICE_HEIGHT = 760;
// Wheel zoom limits, in daily sessions.
const MIN_SESSIONS = 20;
const ZOOM_FACTOR = 1.15;

function chartLayout(priceHeight) {
  const volumeTop = TOP + priceHeight + 20;
  const lowerTop = volumeTop + VOLUME_HEIGHT + 32;
  const dateAxisY = lowerTop + LOWER_HEIGHT + 9;
  return { volumeTop, lowerTop, dateAxisY, height: dateAxisY + 49 };
}

const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
const RANGES = [
  { sessions: 63, label: "3M" },
  { sessions: 126, label: "6M" },
  { sessions: 252, label: "1Y" },
  { sessions: 756, label: "3Y" },
  { sessions: 1260, label: "5Y" },
  { sessions: 2520, label: "10Y" },
];
// Beyond one year, daily candles get thinner than a pixel, so the range buttons switch to weekly bars
// and the price scale turns logarithmic.
const WEEKLY_AFTER = 252;
const BAR_SIZES = [
  { key: "4H", label: "4H", name: "4-hour" },
  { key: "1D", label: "1D", name: "Daily" },
  { key: "1W", label: "1W", name: "Weekly" },
];

const ZONES = [
  { key: "st_buy", label: "ST buy", kind: "buy", term: "short" },
  { key: "st_sell", label: "ST take-profit", kind: "sell", term: "short" },
];

// A zone touch only becomes a marker once momentum has faded, so every marker is a momentum signal.
const TOUCH_LABELS = {
  mom_buy: "Bottom signal",
  mom_sell: "Top signal",
};

function priceText(zone, outcome) {
  if (outcome === "success") return "price reached the 2R target";
  if (outcome === "failed") return zone === "mom_buy" ? "price broke the interim low (stop)" : "price broke the interim high (stop)";
  if (outcome === "expired") return "price hit neither level in 10 sessions";
  return "price outcome still open";
}

function momentumText(zone, followed) {
  if (followed === null || followed === undefined) return "momentum check still open";
  if (zone === "mom_buy") return followed ? "momentum kept rising (valid)" : "momentum turned back down (failed)";
  return followed ? "momentum kept falling (valid)" : "momentum turned back up (failed)";
}

function signalText(touch) {
  const kind = touch.strength === "strong" ? "strong " : "";
  const blocked = touch.during_breakdown ? " · under broken support, not recommended" : "";
  return `${kind}${TOUCH_LABELS[touch.zone].toLowerCase()}: ${momentumText(touch.zone, touch.momentum_followed)}; ${priceText(touch.zone, touch.outcome)}${blocked}`;
}

// Marker fill follows the momentum check: filled = valid, hollow = failed, dashed = not yet known.
function validityClass(followed) {
  if (followed === null || followed === undefined) return "pending";
  return followed ? "success" : "failed";
}

const valid = (number) => typeof number === "number" && Number.isFinite(number);
const money = (number) => (valid(number) ? `$${number.toFixed(2)}` : "-");

function compactVolume(number) {
  if (!valid(number)) return "-";
  if (number >= 1_000_000) return `${(number / 1_000_000).toFixed(1)}M`;
  if (number >= 1_000) return `${(number / 1_000).toFixed(1)}K`;
  return number.toFixed(0);
}

function formatDateTick(dateString, includeYear, monthOnly = false) {
  const [year, month, day] = dateString.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day)).toLocaleDateString("en-US", {
    month: "short",
    ...(monthOnly ? {} : { day: "numeric" }),
    ...(includeYear ? { year: "numeric" } : {}),
    timeZone: "UTC",
  });
}

function weekStart(dateString) {
  const date = new Date(`${dateString}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() - ((date.getUTCDay() + 6) % 7));
  return date.toISOString().slice(0, 10);
}

// One bar per calendar week: OHLC and volume aggregate, indicators and zones take the week's last session.
function toWeekly(points) {
  const buckets = [];
  points.forEach((point) => {
    const key = weekStart(point.date);
    if (!buckets.length || buckets[buckets.length - 1].key !== key) buckets.push({ key, rows: [] });
    buckets[buckets.length - 1].rows.push(point);
  });
  return buckets.map(({ rows }) => {
    const first = rows[0];
    const last = rows[rows.length - 1];
    const highs = rows.map((row) => row.high).filter(valid);
    const lows = rows.map((row) => row.low).filter(valid);
    return {
      ...last,
      date: first.date,
      key: first.date,
      sessionDates: rows.map((row) => row.date),
      open: first.open,
      high: highs.length ? Math.max(...highs) : last.close,
      low: lows.length ? Math.min(...lows) : last.close,
      volume: rows.reduce((total, row) => total + (row.volume || 0), 0),
    };
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

function Diamond({ x, y, className, title, size = 5.5 }) {
  const shape = (half) => `${x},${y - half} ${x + half},${y} ${x},${y + half} ${x - half},${y}`;
  return (
    <g>
      {/* A halo in the page colour keeps the marker readable over zone bands and nearby candles. */}
      <polygon className="marker-halo" points={shape(size + 2.5)} />
      <polygon className={className} points={shape(size)}>
        <title>{title}</title>
      </polygon>
    </g>
  );
}

// Signals fire a few sessions after the turn, so neighbouring candles often reach further than the
// signal bar. Markers sit clear of the extreme of the surrounding bars rather than the bar itself.
const MARKER_NEIGHBOURS = 3;
const MARKER_GAP = 5;

function markerY(points, index, bottom, size, priceY, paneTop, paneBottom) {
  const around = points.slice(Math.max(index - MARKER_NEIGHBOURS, 0), index + MARKER_NEIGHBOURS + 1);
  if (bottom) {
    const lows = around.map((point) => point.low).filter(valid);
    return Math.min(priceY(Math.min(...lows)) + size + MARKER_GAP, paneBottom - size);
  }
  const highs = around.map((point) => point.high).filter(valid);
  return Math.max(priceY(Math.max(...highs)) - size - MARKER_GAP, paneTop + size);
}

export default function MarketChart({ chart, shortTerm }) {
  // `range` is the number of daily sessions shown; `offset` is how many sessions the right edge sits before the latest.
  const [view, setView] = useState({ range: 126, offset: 0 });
  const [priceHeight, setPriceHeight] = useState(DEFAULT_PRICE_HEIGHT);
  const [dragging, setDragging] = useState(false);
  const [lowerStudy, setLowerStudy] = useState("MOMENTUM");
  const [layers, setLayers] = useState({ short: true, momentum: true });
  const [activeIndex, setActiveIndex] = useState(null);
  const svgRef = useRef(null);
  const dragRef = useRef(null);
  const zoomRef = useRef(null);
  const [barSize, setBarSize] = useState("1D");
  const points = chart?.points || [];
  const fourHourBars = chart?.bars_4h || [];
  const { range, offset } = view;

  const dailyByDate = useMemo(() => new Map(points.map((point) => [point.date, point])), [points]);
  const fourHourDates = useMemo(
    () => [...new Set(fourHourBars.map((bar) => bar.date))].filter((date) => dailyByDate.has(date)),
    [fourHourBars, dailyByDate],
  );
  // `range` and `offset` count trading days for every candle size, so switching keeps the same period on screen.
  const totalSessions = barSize === "4H" ? fourHourDates.length : points.length;
  const shownSessions = Math.max(Math.min(range, totalSessions - offset), 0);
  const logScale = range > WEEKLY_AFTER;
  const visiblePoints = useMemo(() => {
    const end = totalSessions - offset;
    const start = Math.max(end - range, 0);
    if (barSize === "4H") {
      const days = new Set(fourHourDates.slice(start, end));
      const bars = fourHourBars.filter((bar) => days.has(bar.date) && valid(bar.close));
      // Zones, moving averages and the lower-pane studies are daily: each candle carries its day's values.
      // Touches are daily too, so only a day's first candle claims them.
      return bars.map((bar, index) => ({
        ...dailyByDate.get(bar.date),
        ...bar,
        key: `${bar.date} ${bar.time}`,
        sessionDates: index === 0 || bars[index - 1].date !== bar.date ? [bar.date] : [],
      }));
    }
    const daily = points.slice(start, end).filter((point) => valid(point.close)).map((point) => ({ ...point, key: point.date }));
    return barSize === "1W" ? toWeekly(daily) : daily;
  }, [barSize, points, fourHourBars, fourHourDates, dailyByDate, totalSessions, range, offset]);
  const hasChart = visiblePoints.length > 0;

  // React registers wheel listeners as passive, so zoom needs a native listener to stop the page scrolling.
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return undefined;
    function handleWheel(event) {
      const zoom = zoomRef.current;
      if (!zoom || event.deltaY === 0) return;
      event.preventDefault();
      const bounds = svg.getBoundingClientRect();
      const x = ((event.clientX - bounds.left) / bounds.width) * WIDTH;
      // Keep the session under the cursor in place while zooming.
      const fraction = clamp((x - LEFT) / (zoom.dataRight - LEFT), 0, 1);
      setView((current) => {
        const nextRange = clamp(
          Math.round(event.deltaY > 0 ? current.range * ZOOM_FACTOR : current.range / ZOOM_FACTOR),
          Math.min(MIN_SESSIONS, zoom.total),
          zoom.total,
        );
        const nextOffset = Math.round(current.offset + (1 - fraction) * (current.range - nextRange));
        return { range: nextRange, offset: clamp(nextOffset, 0, zoom.total - nextRange) };
      });
    }
    svg.addEventListener("wheel", handleWheel, { passive: false });
    return () => svg.removeEventListener("wheel", handleWheel);
  }, [hasChart]);
  // Touches are keyed by the bar they fall in, so weekly bars show the week's first touch per zone.
  const touchesByDate = useMemo(() => {
    const barDate = new Map();
    visiblePoints.forEach((point) => (point.sessionDates || [point.date]).forEach((date) => barDate.set(date, point.key)));
    const map = new Map();
    (chart?.zone_touches || []).forEach((touch) => {
      const bar = barDate.get(touch.date);
      const key = `${bar}:${touch.zone}`;
      if (bar && touch.zone in TOUCH_LABELS && !map.has(key)) map.set(key, touch);
    });
    return map;
  }, [chart, visiblePoints]);

  if (!hasChart) {
    return <div className="empty-chart">Chart history is unavailable for this ticker.</div>;
  }

  const PRICE_HEIGHT = priceHeight;
  const { volumeTop: VOLUME_TOP, lowerTop: LOWER_TOP, dateAxisY: DATE_AXIS_Y, height: HEIGHT } = chartLayout(priceHeight);
  const atLatest = offset === 0;

  const nextZones = {
    st_buy: shortTerm?.buy_zone,
    st_sell: shortTerm?.sell_zone,
  };
  const shownZones = ZONES.filter((zone) => layers[zone.term]);
  const plotRight = WIDTH - RIGHT;
  // The next-session block only makes sense when the latest session is on screen.
  const step = (plotRight - LEFT) / (visiblePoints.length + (atLatest ? NEXT_SLOTS : 0));
  const candleWidth = Math.max(Math.min(step * 0.62, 9), 2);
  const xAt = (index) => LEFT + step * index + step / 2;
  const latestIndex = visiblePoints.length - 1;
  const latest = visiblePoints[latestIndex];
  const nextStart = xAt(latestIndex) + step / 2;
  const sessionsPerBar = shownSessions / visiblePoints.length;
  zoomRef.current = { dataRight: nextStart, total: totalSessions };

  // Fit the price scale to what is on screen: the bars in view, their zones, and the next-session
  // block only when it is drawn. Today's levels would otherwise squash a view from years ago.
  const priceValues = visiblePoints.flatMap((point) => [point.high, point.low, point.ema20, point.ma50, point.ma200].filter(valid));
  shownZones.forEach((zone) => {
    visiblePoints.forEach((point) => priceValues.push(...[point[`${zone.key}_low`], point[`${zone.key}_high`]].filter(valid)));
    const next = nextZones[zone.key];
    if (atLatest && next) priceValues.push(...[next.low, next.high].filter(valid));
  });
  const minPrice = Math.min(...priceValues);
  const maxPrice = Math.max(...priceValues);
  const pricePad = Math.max((maxPrice - minPrice) * 0.05, maxPrice * 0.004);
  const priceLow = Math.max(minPrice - pricePad, minPrice * 0.9);
  const priceHigh = maxPrice + pricePad;
  // Multi-year ranges use a log scale so equal percentage moves take equal height.
  const toScale = logScale ? (price) => Math.log(Math.max(price, priceLow * 0.5)) : (price) => price;
  const fromScale = logScale ? Math.exp : (value) => value;
  const scaleRange = Math.max(toScale(priceHigh) - toScale(priceLow), 1e-6);
  const priceY = (price) => TOP + ((toScale(priceHigh) - toScale(price)) / scaleRange) * PRICE_HEIGHT;
  const maximumVolume = Math.max(...visiblePoints.map((point) => point.volume || 0), 1);
  const gridPrices = Array.from({ length: 5 }, (_, index) => fromScale(toScale(priceHigh) - (scaleRange * index) / 4));

  const zoneLabels = spreadLabels(
    [
      ...shownZones
        .filter(() => atLatest)
        .filter((zone) => nextZones[zone.key] && valid(nextZones[zone.key].low) && valid(nextZones[zone.key].high))
        .map((zone) => {
          const next = nextZones[zone.key];
          return {
            ...zone,
            anchorY: priceY((next.low + next.high) / 2),
            text: `${zone.label} ${money(next.low)}–${money(next.high)}`,
          };
        }),
      { key: "last", kind: "last", anchorY: priceY(latest.close), text: `${atLatest ? "Last" : latest.date} ${money(latest.close)}` },
    ],
    16,
    TOP + 8,
    TOP + PRICE_HEIGHT - 4,
  );

  let lowerY;
  let lowerTicks = [];
  if (lowerStudy === "MOMENTUM") {
    const values = visiblePoints.map((point) => point.momentum).filter(valid);
    const bound = Math.max(Math.min(Math.max(...values.map(Math.abs), 1), 6), 1);
    lowerY = (value) => LOWER_TOP + ((bound - Math.max(-bound, Math.min(bound, value))) / (bound * 2)) * LOWER_HEIGHT;
    lowerTicks = [-Math.floor(bound), 0, Math.floor(bound)].filter((value, index, list) => list.indexOf(value) === index);
  } else if (lowerStudy === "STRETCH") {
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
  const hovered = visiblePoints[Math.min(activeIndex ?? latestIndex, latestIndex)];
  const markerZones = layers.momentum ? ["mom_buy", "mom_sell"] : [];
  const hoveredTouches = markerZones.map((zone) => touchesByDate.get(`${hovered.key}:${zone}`)).filter(Boolean);
  const barName = BAR_SIZES.find((size) => size.key === barSize).name;
  const hoveredLabel = barSize === "1W" ? `Week of ${hovered.date}` : barSize === "4H" ? `${hovered.date} ${hovered.time}` : hovered.date;

  function handlePointerDown(event) {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      kind: "pan",
      startX: event.clientX,
      startOffset: offset,
      scale: event.currentTarget.getBoundingClientRect().width / WIDTH,
    };
    setDragging(true);
  }

  function handlePointerMove(event) {
    const drag = dragRef.current;
    if (drag?.kind === "pan") {
      // Dragging right brings older sessions into view.
      const bars = (event.clientX - drag.startX) / drag.scale / step;
      const nextOffset = clamp(Math.round(drag.startOffset + bars * sessionsPerBar), 0, Math.max(totalSessions - range, 0));
      if (nextOffset !== offset) setView((current) => ({ ...current, offset: nextOffset }));
      return;
    }
    const bounds = event.currentTarget.getBoundingClientRect();
    const relativeX = ((event.clientX - bounds.left) / bounds.width) * WIDTH;
    setActiveIndex(Math.max(0, Math.min(latestIndex, Math.floor((relativeX - LEFT) / step))));
  }

  function endDrag() {
    dragRef.current = null;
    setDragging(false);
  }

  function handleResizeStart(event) {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      kind: "resize",
      startY: event.clientY,
      startHeight: priceHeight,
      scale: svgRef.current.getBoundingClientRect().width / WIDTH,
    };
  }

  function handleResizeMove(event) {
    const drag = dragRef.current;
    if (drag?.kind !== "resize") return;
    setPriceHeight(clamp(Math.round(drag.startHeight + (event.clientY - drag.startY) / drag.scale), MIN_PRICE_HEIGHT, MAX_PRICE_HEIGHT));
  }

  function chooseRange(sessions) {
    let size = barSize;
    if (barSize === "1D" && sessions > WEEKLY_AFTER) size = "1W";
    if (barSize === "1W" && sessions <= WEEKLY_AFTER) size = "1D";
    const total = size === "4H" ? fourHourDates.length : points.length;
    setBarSize(size);
    setView({ range: Math.min(sessions, total), offset: 0 });
  }

  function chooseBarSize(size) {
    const total = size === "4H" ? fourHourDates.length : points.length;
    setBarSize(size);
    setView((current) => ({ range: clamp(current.range, Math.min(MIN_SESSIONS, total), total), offset: 0 }));
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
            Zones move with the trend. The block at the right edge is next session's zones. Scroll to zoom, drag to pan, double-click to reset, and drag the bar under the chart to resize it.
          </p>
        </div>
        <div className="toolbar-controls">
          <div aria-label="Candle size" className="bar-size" role="group">
            {BAR_SIZES.map((size) => (
              <button
                className={barSize === size.key ? "selected" : ""}
                disabled={size.key === "4H" && !fourHourDates.length}
                key={size.key}
                onClick={() => chooseBarSize(size.key)}
                title={size.key === "4H" ? "4-hour candles cover about the last two years (live data only)" : `${size.name} candles`}
                type="button"
              >
                {size.label}
              </button>
            ))}
          </div>
          {RANGES.map((option, index) => (
            <button
              className={range === option.sessions && atLatest ? "selected" : ""}
              disabled={index > 0 && totalSessions <= RANGES[index - 1].sessions}
              key={option.sessions}
              onClick={() => chooseRange(option.sessions)}
              type="button"
            >
              {option.label}
            </button>
          ))}
          {[
            ["MOMENTUM", "Momentum"],
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
          <input checked={layers.momentum} onChange={() => toggle("momentum")} type="checkbox" />
          <span className="swatch signal" />
          <span className="swatch signal sell" /> Momentum signals
        </label>
        <span className="legend-item ma20">EMA20</span>
        <span className="legend-item ma50">MA50</span>
        <span className="legend-item ma200">MA200</span>
        {!atLatest && (
          <button className="chart-latest" onClick={() => setView((current) => ({ ...current, offset: 0 }))} type="button">
            Back to latest →
          </button>
        )}
      </div>
      {/* Fixed-height readout with fixed-width fields, so hovering never reflows the chart. */}
      <div className="chart-readout">
        <span className="readout-date">{hoveredLabel}</span>
        <span className="readout-field">O {money(hovered.open)}</span>
        <span className="readout-field">H {money(hovered.high)}</span>
        <span className="readout-field">L {money(hovered.low)}</span>
        <span className="readout-field">C {money(hovered.close)}</span>
        <span className="readout-field">Vol {compactVolume(hovered.volume)}</span>
        <span className="readout-zone">
          {valid(hovered.st_buy_low) ? `Buy zone ${money(hovered.st_buy_low)}–${money(hovered.st_buy_high)}` : ""}
        </span>
        <span className="readout-touch">
          {hoveredTouches.map((touch) => (
            <span className={`touch-note ${touch.zone}`} key={touch.zone}>
              {signalText(touch)}
            </span>
          ))}
        </span>
      </div>
      <svg
        aria-label="Daily candlestick chart with short-term buy and sell zones"
        className={`market-chart${dragging ? " panning" : ""}`}
        onDoubleClick={() => setView({ range: Math.min(126, totalSessions), offset: 0 })}
        onPointerCancel={endDrag}
        onPointerDown={handlePointerDown}
        onPointerLeave={() => setActiveIndex(null)}
        onPointerMove={handlePointerMove}
        onPointerUp={endDrag}
        preserveAspectRatio="none"
        role="img"
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      >
        <defs>
          <clipPath id="price-pane">
            <rect height={PRICE_HEIGHT} width={plotRight - LEFT} x={LEFT} y={TOP} />
          </clipPath>
        </defs>
        {gridPrices.map((price) => (
          <g key={price}>
            <line className="grid-line" x1={LEFT} x2={plotRight} y1={priceY(price)} y2={priceY(price)} />
            <text className="axis-label" x={plotRight + 8} y={priceY(price) + 4}>
              {price.toFixed(2)}
            </text>
          </g>
        ))}
        {atLatest && (
          <>
            <rect className="next-session-bg" height={PRICE_HEIGHT} width={plotRight - nextStart} x={nextStart} y={TOP} />
            <text className="pane-label" textAnchor="end" x={plotRight - 4} y={TOP + PRICE_HEIGHT - 6}>
              Next session
            </text>
          </>
        )}
        <text className="pane-label" x={LEFT + 4} y={TOP + 12}>
          {barName} bars{logScale ? " · log scale" : ""}
          {barSize === "4H" ? " · zones, averages and studies are daily" : ""}
        </text>
        {shownZones.map((zone) => {
          const next = nextZones[zone.key];
          return (
            <g className={`zone-layer ${zone.kind} ${zone.term}`} clipPath="url(#price-pane)" key={zone.key}>
              <path className="zone-fill" d={bandPath(visiblePoints, `${zone.key}_low`, `${zone.key}_high`, xAt, priceY)} />
              <path className="zone-edge" d={linePath(visiblePoints, `${zone.key}_low`, xAt, priceY)} />
              <path className="zone-edge" d={linePath(visiblePoints, `${zone.key}_high`, xAt, priceY)} />
              {atLatest && next && valid(next.low) && valid(next.high) && (
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
            <g className={point.close >= open ? "up" : "down"} key={point.key}>
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
        {markerZones.length > 0 &&
          visiblePoints.map((point, index) =>
            markerZones.map((zone) => {
              const touch = touchesByDate.get(`${point.key}:${zone}`);
              if (!touch) return null;
              const bottom = zone === "mom_buy";
              const strong = touch.strength === "strong";
              const classes = [
                "touch-marker signal",
                bottom ? "buy" : "sell",
                validityClass(touch.momentum_followed),
                strong ? "strong" : "",
                touch.during_breakdown ? "blocked" : "",
              ];
              const size = strong ? 7 : 5;
              return (
                <Diamond
                  className={classes.filter(Boolean).join(" ")}
                  key={`${zone}-${point.key}`}
                  size={size}
                  title={`${touch.date} at ${money(touch.entry)}: ${signalText(touch)}${valid(touch.r_multiple) ? ` (${touch.r_multiple > 0 ? "+" : ""}${touch.r_multiple.toFixed(2)}R)` : ""}`}
                  x={xAt(index)}
                  y={markerY(visiblePoints, index, bottom, size, priceY, TOP, TOP + PRICE_HEIGHT)}
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
          {lowerStudy === "STRETCH"
            ? "Stretch from EMA20 (ATRs) · shaded = zone thresholds"
            : lowerStudy === "MOMENTUM"
              ? `Momentum: EMA${shortTerm?.momentum?.fast_span ?? 5} − EMA${shortTerm?.momentum?.slow_span ?? 13} (ATRs) · solid = building, light = fading`
              : lowerStudy}
        </text>
        {lowerStudy === "MOMENTUM" && (
          <>
            {lowerTicks.map((level) => (
              <g key={level}>
                <line className="study-boundary" x1={LEFT} x2={plotRight} y1={lowerY(level)} y2={lowerY(level)} />
                <text className="axis-label" x={plotRight + 8} y={lowerY(level) + 4}>
                  {level > 0 ? `+${level}` : level}
                </text>
              </g>
            ))}
            {visiblePoints.map((point, index) => {
              if (!valid(point.momentum)) return null;
              // Fading = the day's change points back toward zero.
              const fading = valid(point.momentum_change) && point.momentum_change * point.momentum < 0;
              return (
                <rect
                  className={`momentum-bar ${point.momentum >= 0 ? "bull" : "bear"}${fading ? " fading" : ""}`}
                  height={Math.max(Math.abs(lowerY(point.momentum) - lowerY(0)), 0.5)}
                  key={`momentum-${point.key}`}
                  width={candleWidth}
                  x={xAt(index) - candleWidth / 2}
                  y={Math.min(lowerY(point.momentum), lowerY(0))}
                />
              );
            })}
          </>
        )}
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
                key={`macd-${point.key}`}
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
          const includeYear = logScale || index === 0 || point.date.slice(0, 4) !== firstVisibleYear;
          return (
            <g className="date-tick" key={`date-${point.key}`}>
              <line x1={xAt(index)} x2={xAt(index)} y1={DATE_AXIS_Y} y2={DATE_AXIS_Y + 7} />
              <text className="axis-label date" x={xAt(index)} y={DATE_AXIS_Y + 23}>
                {formatDateTick(point.date, includeYear, logScale)}
              </text>
            </g>
          );
        })}
      </svg>
      <div
        aria-label="Drag to resize the chart"
        className="chart-resize-handle"
        onDoubleClick={() => setPriceHeight(DEFAULT_PRICE_HEIGHT)}
        onPointerCancel={endDrag}
        onPointerDown={handleResizeStart}
        onPointerMove={handleResizeMove}
        onPointerUp={endDrag}
        role="separator"
        title="Drag to resize the chart · double-click to reset"
      >
        <span />
      </div>
      <p className="chart-footnote">
        ◆ below a candle = bottom signal, above = top signal: price touched the zone and momentum then faded for the ticker's turn
        length. Large ◆ = strong (price reached the far edge of the zone). Filled = momentum kept moving the expected way for the
        next 10 sessions (valid); hollow = it turned back (failed); dashed = fewer than 10 sessions so far. Grey = fired under
        broken support, not recommended. Hover a marker for the price result.
      </p>
    </section>
  );
}

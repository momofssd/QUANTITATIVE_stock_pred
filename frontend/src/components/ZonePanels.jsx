import { dollar, fixed, isNumber, ordinal, percent, points, range, signedPercent, zonePosition } from "./format";

const STAGE_LABELS = {
  CONFIRMED: "Bounce confirmed",
  FORMING: "Bounce forming",
  WASHOUT_NO_REVERSAL_YET: "Washed out, no reversal yet",
  NOT_SET_UP: "No bounce setup",
};

const EDGE_TONE = { EDGE: "positive", SLIGHT_EDGE: "caution", NO_EDGE: "negative", THIN_SAMPLE: "neutral" };

// After a breakdown the edge measures continuation, so a strong edge is bad news for longs.
const BREAKDOWN_EDGE = {
  EDGE: ["negative", "Breakdowns here usually kept falling"],
  SLIGHT_EDGE: ["caution", "Breakdowns here slightly favored more downside"],
  NO_EDGE: ["neutral", "Breakdowns here often reversed"],
  THIN_SAMPLE: ["neutral", "Too few past breakdowns to judge"],
};

function Fact({ label, value, detail }) {
  return (
    <div className="fact">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}

function Stat({ label, value, detail }) {
  return (
    <div className="stat">
      <strong>{value}</strong>
      <span>{label}</span>
      {detail && <small>{detail}</small>}
    </div>
  );
}

function Row({ label, value, detail }) {
  return (
    <div className="zone-row">
      <span>
        {label}
        {detail && <small>{detail}</small>}
      </span>
      <strong>{value}</strong>
    </div>
  );
}

// Percentile strip: where today sits in the ticker's own history, with the zone bands marked.
function PercentileGauge({ label, valueLabel, percentile, buyBand, sellBand, sellLabel = "Sell zone" }) {
  const marker = isNumber(percentile) ? Math.min(Math.max(percentile, 0), 1) * 100 : null;
  return (
    <div className="gauge">
      <div className="gauge-head">
        <span>{label}</span>
        <strong>{valueLabel}</strong>
      </div>
      <div className="gauge-track" role="img" aria-label={`${label}: ${valueLabel}`}>
        <i className="gauge-band buy" style={{ left: `${buyBand[0]}%`, width: `${buyBand[1] - buyBand[0]}%` }} />
        <i className="gauge-band sell" style={{ left: `${sellBand[0]}%`, width: `${sellBand[1] - sellBand[0]}%` }} />
        {marker !== null && <b className="gauge-marker" style={{ left: `${marker}%` }} />}
      </div>
      <div className="gauge-scale">
        <span>Cheapest</span>
        <span className="buy">Buy zone</span>
        <span className="sell">{sellLabel}</span>
        <span>Most stretched</span>
      </div>
    </div>
  );
}

export function ShortTermHero({ shortTerm, market }) {
  const history = shortTerm.history;
  const touch = history.buy_zone;
  const confirmed = history.confirmed_entry;
  const readiness = shortTerm.bounce_readiness;
  const edge = shortTerm.historical_edge;
  const breakdown = shortTerm.breakdown?.active ? shortTerm.breakdown : null;
  const [edgeTone, edgeLabel] = breakdown ? BREAKDOWN_EDGE[edge.grade] : [EDGE_TONE[edge.grade], edge.label];
  return (
    <section className={`hero-card ${shortTerm.tone}`}>
      <div className="hero-head">
        <p className="eyebrow">
          Short-term entry · {shortTerm.horizon_sessions}-session {breakdown ? "support breakdown" : "V-bounce"}
        </p>
        <span className={`chip ${edgeTone}`}>{edgeLabel}</span>
      </div>
      <h2>{shortTerm.action_label}</h2>
      <p className="summary">{shortTerm.action_definition}</p>
      <div className="hero-facts">
        <Fact
          detail={`${signedPercent(market.change_pct)} today · ${market.trend.toLowerCase()}`}
          label="Current price"
          value={dollar(market.current_price)}
        />
        <Fact detail={zonePosition(shortTerm.buy_zone, "buy")} label="Buy zone (next session)" value={range(shortTerm.buy_zone)} />
        <Fact detail={zonePosition(shortTerm.sell_zone, "sell")} label="Take-profit zone" value={range(shortTerm.sell_zone)} />
        {breakdown ? (
          <Fact
            detail={`${signedPercent(breakdown.close_vs_broken_level_pct)} below it · broke ${breakdown.breakdown_date}`}
            label="Broken support (swing low)"
            value={dollar(breakdown.broken_level)}
          />
        ) : (
          <Fact
            detail={`${readiness.score} of ${readiness.max_score} checklist items met`}
            label="Bounce readiness"
            value={STAGE_LABELS[readiness.stage]}
          />
        )}
      </div>
      <PercentileGauge
        buyBand={shortTerm.buy_zone.percentile_band}
        label="Stretch from the 20-day EMA, vs. the last 3 years"
        percentile={shortTerm.stretch.percentile}
        sellBand={shortTerm.sell_zone.percentile_band}
        sellLabel="Take-profit"
        valueLabel={`${fixed(shortTerm.stretch.current_atr, 2)} ATR · ${ordinal(shortTerm.stretch.percentile)} percentile`}
      />
      {breakdown ? (
        <BreakdownStats breakdown={breakdown} study={history.breakdown} />
      ) : (
        <BounceStats confirmed={confirmed} touch={touch} />
      )}
      <p className="footnote">
        {breakdown
          ? `A breakdown kept falling when price dropped another ${touch.bracket_atr} ATR before rising ${touch.bracket_atr} ATR within ${touch.horizon_sessions} sessions. Compare the rate with the any-day baseline, not with 100%.`
          : `A bounce means price rose ${touch.bracket_atr} ATR before falling ${touch.bracket_atr} ATR within ${touch.horizon_sessions} sessions, so about 50% is break-even. Compare each rate with the any-day baseline, not with 100%.`}
      </p>
    </section>
  );
}

function BounceStats({ touch, confirmed }) {
  return (
    <div className="hero-stats">
      <Stat
        detail={`vs ${percent(touch.baseline_success_rate, 0)} on any day · ${touch.episodes} touches`}
        label="Bounce rate buying at the zone"
        value={percent(touch.success_rate, 0)}
      />
      <Stat
        detail={`${points(confirmed.edge_vs_baseline)} vs any day · ${confirmed.episodes} setups`}
        label="Bounce rate waiting for a reversal close"
        value={percent(confirmed.success_rate, 0)}
      />
      <Stat
        detail={`worst dip ${signedPercent(touch.median_adverse_move)} (median)`}
        label="Median best bounce"
        value={signedPercent(touch.median_favorable_move)}
      />
      <Stat
        detail={`target hit in ${fixed(touch.median_days_to_target, 0)} sessions when it worked`}
        label="Sessions to the low (median)"
        value={fixed(touch.median_days_to_extreme, 0)}
      />
    </div>
  );
}

function BreakdownStats({ breakdown, study }) {
  return (
    <div className="hero-stats">
      <Stat
        detail={`vs ${percent(study.baseline_success_rate, 0)} on any day · ${study.episodes} breakdowns`}
        label="Kept falling after a breakdown"
        value={percent(study.success_rate, 0)}
      />
      <Stat
        detail={`closed back above the broken level within ${study.horizon_sessions} sessions`}
        label="Failed breakdowns (reclaimed)"
        value={percent(study.reclaim_rate, 0)}
      />
      <Stat
        detail={`worst rebound ${signedPercent(-study.median_adverse_move)} (median)`}
        label="Median further drop"
        value={signedPercent(-study.median_favorable_move)}
      />
      <Stat
        detail={`${study.bracket_atr} ATR below today's close; next support ${dollar(breakdown.next_support)}`}
        label="Follow-through target"
        value={dollar(breakdown.follow_through_target)}
      />
    </div>
  );
}

export function LevelLadder({ report }) {
  const { short_term: shortTerm, current_market: market, technical_indicators: technical } = report;
  const price = market.current_price;
  const levels = [
    { key: "st_sell", label: "Short-term take-profit", zone: shortTerm.sell_zone, kind: "sell" },
    { key: "st_buy", label: "Short-term buy zone", zone: shortTerm.buy_zone, kind: "buy" },
    { key: "ema20", label: "20-day EMA (mean)", price: technical.ema20, kind: "line" },
    { key: "ma200", label: "200-day MA", price: technical.ma200, kind: "line" },
    { key: "current", label: "Current price", price, kind: "current" },
  ];
  const breakdown = shortTerm.breakdown;
  if (breakdown?.active) levels.push({ key: "broken", label: "Broken support (reclaim level)", price: breakdown.broken_level, kind: "line broken" });
  if (breakdown) levels.push({ key: "support", label: "Next swing-low support", price: breakdown.next_support, kind: "line support" });
  const middle = (level) => (level.zone ? (level.zone.low + level.zone.high) / 2 : level.price);
  const ordered = levels.filter((level) => isNumber(middle(level))).sort((left, right) => middle(right) - middle(left));
  const distance = (level) => {
    if (level.kind === "current") return "";
    if (level.zone) {
      if (price >= level.zone.low && price <= level.zone.high) return "here";
      const edge = price > level.zone.high ? level.zone.high : level.zone.low;
      return signedPercent(edge / price - 1);
    }
    return signedPercent(level.price / price - 1);
  };
  return (
    <article className="panel ladder">
      <h2>Price ladder</h2>
      <p className="panel-note">Every level, highest first, with the move needed from the current price.</p>
      <table>
        <tbody>
          {ordered.map((level) => (
            <tr className={`ladder-row ${level.kind}`} key={level.key}>
              <td className="ladder-swatch">
                <i />
              </td>
              <th scope="row">{level.label}</th>
              <td className="ladder-value">{level.zone ? range(level.zone) : dollar(level.price)}</td>
              <td className="ladder-distance">{distance(level)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </article>
  );
}

export function ReadinessPanel({ readiness }) {
  return (
    <article className="panel readiness">
      <h2>Bounce checklist</h2>
      <p className="panel-note">
        {STAGE_LABELS[readiness.stage]} · {readiness.score}/{readiness.max_score} ·{" "}
        {readiness.touched_buy_zone_recently ? "buy zone touched in the last 3 sessions" : "no recent buy-zone touch"}
      </p>
      <ul className="checklist">
        {readiness.checks.map((check) => (
          <li className={check.passed ? "passed" : "missing"} key={check.key}>
            <span aria-hidden="true" className="check-icon">
              {check.passed ? "✓" : "–"}
            </span>
            <div>
              <strong>
                {check.label}
                <span className="visually-hidden">{check.passed ? " (met)" : " (not met)"}</span>
              </strong>
              <small>{check.detail}</small>
            </div>
          </li>
        ))}
      </ul>
    </article>
  );
}

export function TradePlanPanel({ shortTerm }) {
  const plan = shortTerm.trade_plan;
  return (
    <article className="panel">
      <h2>Short-term trade plan</h2>
      <p className="panel-note">
        Entry basis: {plan.entry_reference_basis}. Levels apply to the next session; recompute after each close.
      </p>
      <Row label="Entry" value={dollar(plan.entry_reference)} />
      {shortTerm.breakdown?.active && (
        <Row
          detail="where a continuation would reach; no long entry until the broken support is reclaimed"
          label="Follow-through target"
          value={dollar(shortTerm.breakdown.follow_through_target)}
        />
      )}
      <Row detail={plan.stop_basis} label="Stop" value={dollar(plan.stop_loss)} />
      <Row detail={plan.target_1_basis} label="Target 1" value={`${dollar(plan.target_1)} · ${fixed(plan.reward_risk_target_1, 1)}R`} />
      <Row detail={plan.target_2_basis} label="Target 2" value={`${dollar(plan.target_2)} · ${fixed(plan.reward_risk_target_2, 1)}R`} />
      <Row detail="where a V-bounce usually reverts to" label="20-day EMA" value={dollar(plan.mean_reversion_level)} />
    </article>
  );
}

export function ReasonsPanel({ shortTerm }) {
  return (
    <article className="panel">
      <h2>Why this reading</h2>
      <div className="ai-columns">
        <div>
          <h3>Supporting</h3>
          {shortTerm.reasons.length ? (
            <ul className="ai-list">
              {shortTerm.reasons.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : (
            <p className="panel-note">No supporting evidence.</p>
          )}
        </div>
        <div>
          <h3>Risks</h3>
          {shortTerm.risks.length ? (
            <ul className="ai-list">
              {shortTerm.risks.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : (
            <p className="panel-note">No specific risk flags.</p>
          )}
        </div>
      </div>
    </article>
  );
}

function contextRate(study) {
  const context = study.by_trend;
  if (!context || !context.current_context_episodes) return "-";
  return `${percent(context.current_context_success_rate, 0)} (${context.current_context_episodes}, ${context.current_context})`;
}

export function EvidenceTables({ shortTerm }) {
  const st = shortTerm.history;
  const shortRows = [
    { label: "Buy zone, first touch", study: st.buy_zone, context: true },
    { label: "Buy zone, wait for reversal close", study: st.confirmed_entry },
    { label: "Take-profit zone, first touch", study: st.sell_zone, context: true },
    { label: "Close below swing-low support (worked = kept falling)", study: st.breakdown, context: true },
  ].filter((row) => row.study);
  return (
    <section className="panel evidence">
      <h2>Historical replay</h2>
      <p className="panel-note">
        Every earlier visit to each zone, graded with no hindsight: the zone that applied on a day was computed only from data
        before it. These are frequencies, not forecasts.
      </p>
      <h3>
        Short-term: ±{st.buy_zone.bracket_atr} ATR bracket over {st.buy_zone.horizon_sessions} sessions
      </h3>
      <table className="evidence-table">
        <thead>
          <tr>
            <th scope="col">Setup</th>
            <th scope="col">Setups</th>
            <th scope="col">Worked</th>
            <th scope="col">Any day</th>
            <th scope="col">Edge</th>
            <th scope="col">Median best move</th>
            <th scope="col">Current trend context</th>
          </tr>
        </thead>
        <tbody>
          {shortRows.map(({ label, study, context }) => (
            <tr key={label}>
              <th scope="row">{label}</th>
              <td>{study.episodes}</td>
              <td>{percent(study.success_rate, 0)}</td>
              <td>{percent(study.baseline_success_rate, 0)}</td>
              <td className={isNumber(study.edge_vs_baseline) ? (study.edge_vs_baseline > 0 ? "good" : "bad") : ""}>
                {points(study.edge_vs_baseline)}
              </td>
              <td>{signedPercent(study.median_favorable_move)}</td>
              <td>{context ? contextRate(study) : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

const FEATURE_LABELS = {
  dist_ma200: "distance from the 200-day MA",
  ma200_slope_20d: "200-day MA slope",
  return_60d: "60-session return",
  return_120d: "120-session return",
};

const DIRECTION_TONE = { BULLISH: "bullish", BEARISH: "bearish", NEUTRAL: "neutral" };
const RELIABILITY_TONE = { USEFUL: "positive", WEAK: "caution", NO_SKILL: "negative", INSUFFICIENT_HISTORY: "neutral" };

function TrendHorizon({ ticker, outlook, title }) {
  if (!outlook?.available) {
    return (
      <div className="trend-horizon">
        <p className="eyebrow">{title}</p>
        <p className="panel-note">{outlook?.reason || "Trend outlook unavailable."}</p>
      </div>
    );
  }
  const test = outlook.walk_forward;
  const fromModel = outlook.probability_source === "model";
  const bullish = outlook.probability_bullish;
  const callRate = (rate, count) => (count ? `${percent(rate, 0)} right (${count})` : "none made");
  return (
    <div className="trend-horizon">
      <div className="hero-head">
        <p className="eyebrow">
          {title} · next {outlook.horizon_sessions} sessions
        </p>
        <span className={`chip ${RELIABILITY_TONE[outlook.reliability]}`}>Model: {outlook.reliability_label}</span>
      </div>
      <div className={`trend-call ${DIRECTION_TONE[outlook.direction]}`}>
        <strong>{outlook.direction.toLowerCase()}</strong>
        <span>{percent(bullish, 0)} chance of a higher close</span>
      </div>
      <div
        aria-label={`${percent(bullish, 0)} bullish, ${percent(1 - bullish, 0)} bearish`}
        className="probability-bar"
        role="img"
      >
        <i className="bull" style={{ width: `${bullish * 100}%` }} />
        <i className="bear" style={{ width: `${(1 - bullish) * 100}%` }} />
        <b style={{ left: "50%" }} title="50%" />
      </div>
      <div className="probability-scale">
        <span>Bullish {percent(bullish, 0)}</span>
        <span>Bearish {percent(1 - bullish, 0)}</span>
      </div>
      <p className="summary trend-source">
        {fromModel
          ? `From the trend model, which beat ${ticker}'s base rate (${percent(outlook.base_rate_bullish, 0)}) in the walk-forward test.`
          : `${ticker}'s base rate: it closed higher ${outlook.horizon_sessions} sessions later ${percent(outlook.base_rate_bullish, 0)} of the time. The trend model did not beat this out of sample, so its reading is shown only for reference.`}
      </p>
      <div className="trend-model">
        <Row
          detail={outlook.top_drivers
            .slice(0, 2)
            .map((driver) => `${FEATURE_LABELS[driver.feature] || driver.feature} → ${driver.pushes}`)
            .join("; ")}
          label="Trend model says"
          value={`${outlook.model_direction.toLowerCase()} ${percent(outlook.model_probability_bullish, 0)}`}
        />
        <Row
          detail={`${test.sessions_scored} sessions since ${test.start}; 0.50 = coin flip`}
          label="Walk-forward AUC"
          value={fixed(test.auc, 2)}
        />
        <Row
          detail="vs. always quoting the base rate; below 0 = worse"
          label="Brier skill"
          value={signedPercent(test.brier_skill_score)}
        />
        <Row
          label="Model's bullish / bearish calls"
          value={`${callRate(test.bullish_call_hit_rate, test.bullish_calls)} / ${callRate(test.bearish_call_hit_rate, test.bearish_calls)}`}
        />
      </div>
    </div>
  );
}

export function TrendOutlook({ ticker, outlook }) {
  if (!outlook) return null;
  return (
    <section className="panel trend-panel">
      <h2>Trend outlook</h2>
      <p className="panel-note">
        Probability of a higher close at each horizon. The headline uses the trend model only when it beat the ticker's own base
        rate in a walk-forward test (trained on the past, scored on the following period); otherwise it reports the base rate.
      </p>
      <div className="trend-grid">
        <TrendHorizon outlook={outlook.short_term} ticker={ticker} title="Short-term trend" />
        <TrendHorizon outlook={outlook.long_term} ticker={ticker} title="Long-term trend" />
      </div>
    </section>
  );
}

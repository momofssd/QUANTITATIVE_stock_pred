import { useState } from "react";
import { evaluatePriceAction } from "./priceAction";

const price = (value) => Number.isFinite(value) ? `$${value.toFixed(value < 1 ? 4 : 2)}` : "—";
const pct = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
const label = (value) => value.replaceAll("_", " ");

export default function SessionOutlook({ forecast }) {
  const [selected, setSelected] = useState(5);
  const [scenarioPrice, setScenarioPrice] = useState("");
  const [entryPrice, setEntryPrice] = useState("");
  const sessions = forecast.session_forecasts || [];
  const session = sessions.find((item) => item.session === selected) || sessions.at(-1);
  if (!session) return null;
  const distribution = session.price_distribution;
  const plan = forecast.price_action_plan;
  const hypothetical = scenarioPrice.trim() ? Number(scenarioPrice) : NaN;
  const basis = entryPrice.trim() ? Number(entryPrice) : NaN;
  const result = evaluatePriceAction(plan, hypothetical, basis);
  const returnFromEntry = basis > 0 && result ? hypothetical / basis - 1 : null;

  return (
    <section className="session-outlook" aria-label="Session price probabilities and actions">
      <h3>Next five sessions</h3>
      <p className="footnote">Each row forecasts from the same current close. Gain chance comes from the direction classifier; range chances below come from historical regression errors.</p>
      <div className="session-table-scroll">
        <table className="session-table">
          <caption>Closing-price outlook by trading session</caption>
          <thead><tr><th>Session</th><th>Gain chance</th><th>Expected close</th><th>80% nominal range</th></tr></thead>
          <tbody>{sessions.map((item) => (
            <tr key={item.session} className={item.session === session.session ? "selected-session" : ""}>
              <th><button type="button" aria-pressed={item.session === session.session} onClick={() => setSelected(item.session)}>Day {item.session}</button></th>
              <td>{pct(item.probability_of_gain)}</td>
              <td>{price(item.expected_close)}</td>
              <td>{price(item.calibrated_price_range.low)} – {price(item.calibrated_price_range.high)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <h3>Day {session.session} range chances · {label(distribution.bias)}</h3>
      <p className="footnote">Ranges include the lower bound and exclude the upper bound. Sideways is −1% to +1% from the reference close.</p>
      <div className="distribution-bars">
        {distribution.bins.map((bin) => (
          <div className="distribution-bin" key={bin.label}>
            <span>{label(bin.label)}<small>{price(bin.low_price)} to {bin.high_price === null ? "unbounded" : price(bin.high_price)}</small></span>
            <div className="distribution-track"><i className={bin.label.toLowerCase()} style={{ width: `${bin.probability * 100}%` }} /></div>
            <strong>{pct(bin.probability)}</strong>
          </div>
        ))}
      </div>
      <p className="footnote">Empirical gain chance: {pct(distribution.probability_of_gain)}. Historical coverage of the nominal 80% range: {pct(session.calibrated_price_range.heldout_observed_coverage)} across {session.validation.test_rows} test rows. Calibration: {label(session.confidence_basis)}.</p>
      <p className="footnote">{distribution.note}</p>
      <details><summary>Distribution validation</summary>
        <p className="footnote">Multiclass Brier score (lower is better): {session.validation.price_distribution.multiclass_brier_score.toFixed(3)}. Estimated from untouched historical test outcomes; overlapping sessions reduce independent evidence.</p>
        <p className="footnote">{distribution.calibration_samples} live calibration samples. Quantiles: 10th {price(distribution.price_quantiles.p10)}, median {price(distribution.price_quantiles.p50)}, 90th {price(distribution.price_quantiles.p90)}.</p>
      </details>
      {plan && <>
        <h3>If price reaches a level</h3>
        <p className="footnote">Reference close {price(plan.reference_price)} · support {price(plan.levels.support)} · resistance {price(plan.levels.resistance)}. Invalidation takes priority over bounce-watch guidance.</p>
        <p className="footnote">Conditional guidance for an existing long. New-entry validation: {plan.new_entry_supported_by_validation ? "passes screening checks" : "insufficient evidence"}.</p>
        {plan.rules.map((rule) => (
          <details className="price-action-rule" key={rule.action}>
            <summary>{rule.operator === "at_or_below" ? "At/below" : "At/above"} {price(rule.trigger_price)} · {label(rule.action)}</summary>
            <p className="footnote">{rule.guidance}</p>
          </details>
        ))}
        <div className="price-scenario">
          <label>What if price reaches ($)?<input type="number" min="0" step="any" value={scenarioPrice} onChange={(event) => setScenarioPrice(event.target.value)} placeholder={String(plan.reference_price)} /></label>
          <label>Your entry price ($, optional)<input type="number" min="0" step="any" value={entryPrice} onChange={(event) => setEntryPrice(event.target.value)} /></label>
        </div>
        <div aria-live="polite">
          {scenarioPrice && !result && <p className="footnote">Enter a positive price.</p>}
          {result && <>
            <h4>{label(result.action)}</h4><p className="footnote">{result.guidance}</p>
            {returnFromEntry !== null && <p className="footnote">Return from your entry before costs: {pct(returnFromEntry)}.{returnFromEntry <= 0 ? " This would not be a profit-taking exit." : " Check fees and other costs before treating this as profit."}</p>}
          </>}
        </div>
        <p className="footnote">{plan.note}</p>
        <details><summary>How the action levels are calculated</summary><p className="footnote">{plan.basis}</p></details>
      </>}
    </section>
  );
}

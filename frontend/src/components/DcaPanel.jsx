import { dollar, isNumber, signedPercent } from "./format";

const basisPoints = (value) => (isNumber(value) ? `${value > 0 ? "+" : ""}${Math.round(value * 1e4)} bp` : "-");
const money = (value) => (isNumber(value) ? `$${Math.round(value).toLocaleString("en-US")}` : "-");
// Cost gaps are tiny; one that rounds to zero reads as 0.00%, not +0.00% or -0.00%.
const costGap = (value) => (isNumber(value) && Math.round(value * 1e4) === 0 ? "0.00%" : signedPercent(value, 2));

function dcaColumns(dca) {
  return [
    ["Avg cost", `Total invested divided by shares bought, buying ${money(dca.amount)} at each ${dca.price_basis}.`],
    ["vs average", "Average cost compared with the average of the five weekdays. Negative (green) bought more cheaply."],
    ["Final value", "What the shares are worth at the latest close."],
    ["Years cheapest", `Calendar years (with at least half a year of buys) in which that weekday had the lowest average cost, out of ${dca.years}.`],
    ["Trend removed", `${dca.nearby_definition} t is the strength of that difference; about 2 or more is unlikely to be chance.`],
    ["Holiday moves", dca.holiday_rule],
    ["Full history", "The full-history vs average for the same weekday, to compare the last year with the long run."],
  ];
}

function TrendCell({ row, highlight }) {
  return (
    <td className={highlight ? "good" : ""}>
      {basisPoints(row.vs_nearby)}{" "}
      <span className="dca-t">t {isNumber(row.vs_nearby_t) ? Math.abs(row.vs_nearby_t).toFixed(1) : "-"}</span>
    </td>
  );
}

function ColumnHeads({ columns, names }) {
  return (
    <thead>
      <tr>
        <th scope="col">Buy every</th>
        {columns
          .filter(([name]) => names.includes(name))
          .map(([name, definition]) => (
            <th key={name} scope="col" title={definition}>
              {name}
            </th>
          ))}
      </tr>
    </thead>
  );
}

export default function DcaPanel({ dca }) {
  if (!dca) return null;
  const columns = dcaColumns(dca);
  const [first, second] = dca.halves;
  const recent = dca.recent;
  const fullByDay = Object.fromEntries(dca.weekdays.map((row) => [row.weekday, row]));
  return (
    <section className="panel evidence dca-panel">
      <div className="panel-heading">
        <h2>Weekly DCA: which weekday to buy</h2>
        <span className={`chip ${dca.reliable ? "positive" : ""}`}>
          {dca.reliable ? `${dca.trend_neutral_best} edge` : "No reliable weekday edge"}
        </span>
      </div>
      <p className="panel-note">
        {money(dca.amount)} every week at the close. {dca.holiday_rule}
      </p>

      <h3>
        Full history: {dca.start} to {dca.end} ({dca.weeks} weeks, {money(dca.invested)} invested)
      </h3>
      <p className="dca-summary">{dca.summary}</p>
      <table className="evidence-table">
        <ColumnHeads columns={columns} names={["Avg cost", "vs average", "Final value", "Years cheapest", "Trend removed", "Holiday moves"]} />
        <tbody>
          {dca.weekdays.map((row) => (
            <tr className={row.best ? "best-row" : ""} key={row.weekday}>
              <th scope="row">{row.weekday}</th>
              <td>{dollar(row.average_cost)}</td>
              <td className={row.best ? "good" : ""}>{costGap(row.vs_average)}</td>
              <td>{money(row.final_value)}</td>
              <td>{row.years_cheapest}</td>
              <TrendCell highlight={dca.reliable && row.weekday === dca.trend_neutral_best} row={row} />
              <td>{row.holiday_moves}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="panel-note dca-halves">
        Cheapest once the trend is removed: {first.cheapest_vs_nearby} in {first.start.slice(0, 4)}–{first.end.slice(0, 4)},{" "}
        {second.cheapest_vs_nearby} in {second.start.slice(0, 4)}–{second.end.slice(0, 4)}.
        {dca.consistent_across_halves ? " The same weekday both times." : " A different weekday each time, so the ranking is not stable."}
      </p>

      {recent && (
        <>
          <h3>
            Last 12 months: {recent.start} to {recent.end} ({recent.weeks} weeks, {money(recent.invested)} invested)
          </h3>
          <p className="dca-summary">{recent.summary}</p>
          <table className="evidence-table">
            <ColumnHeads columns={columns} names={["Avg cost", "vs average", "Final value", "Trend removed", "Holiday moves", "Full history"]} />
            <tbody>
              {recent.weekdays.map((row) => (
                <tr className={row.best ? "best-row" : ""} key={row.weekday}>
                  <th scope="row">{row.weekday}</th>
                  <td>{dollar(row.average_cost)}</td>
                  <td className={row.best ? "good" : ""}>{costGap(row.vs_average)}</td>
                  <td>{money(row.final_value)}</td>
                  <TrendCell row={row} />
                  <td>{row.holiday_moves}</td>
                  <td className="dca-compare">{costGap(fullByDay[row.weekday]?.vs_average)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <dl className="column-definitions">
        {columns.map(([name, definition]) => (
          <div key={name}>
            <dt>{name}</dt>
            <dd>{definition}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

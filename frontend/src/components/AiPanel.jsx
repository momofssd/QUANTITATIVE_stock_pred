import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Section order and titles for the markdown fields Gemini returns.
const SECTIONS = [
  ["summary_markdown", "Summary"],
  ["short_term_markdown", "Short-term plan"],
  ["historical_replay_markdown", "Historical replay"],
  ["trend_outlook_markdown", "Trend outlook"],
  ["long_term_markdown", "Long-term plan"],
];

const VIEW_LABELS = {
  SUPPORT_BREAKDOWN: "Support breakdown",
  ENTRY_CONFIRMED: "Entry confirmed",
  ENTRY_ZONE_BOUNCE_FORMING: "In entry zone, bounce forming",
  ENTRY_ZONE: "In entry zone",
  BELOW_ENTRY_ZONE: "Below entry zone",
  APPROACHING_ENTRY: "Approaching entry",
  BOUNCE_UNDERWAY: "Bounce underway",
  WAIT_FOR_PULLBACK: "Wait for pullback",
  TAKE_PROFIT_ZONE: "Take-profit zone",
  ACCUMULATE: "Accumulate",
  VALUE_ZONE_LOW_RR: "In value zone, reward/risk under 3:1",
  BELOW_LONG_TERM_STOP: "Below long-term stop",
  DEEP_VALUE_TREND_RISK: "Below value zone",
  APPROACHING_VALUE: "Approaching value zone",
  HOLD_WAIT: "Hold / wait",
  APPROACHING_TRIM: "Approaching trim zone",
  TRIM_ZONE: "Trim zone",
};

const viewLabel = (code) => VIEW_LABELS[code] || code?.replaceAll("_", " ").toLowerCase() || "-";

function Markdown({ children }) {
  if (!children) return null;
  return (
    <div className="markdown">
      <ReactMarkdown
        components={{ a: ({ node, ...props }) => <a {...props} rel="noreferrer" target="_blank" /> }}
        remarkPlugins={[remarkGfm]}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

function PlainList({ items }) {
  if (!Array.isArray(items) || !items.length) return null;
  return (
    <ul className="ai-list">
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

export default function AiPanel({ interpretation, loading, error, disabled, onGenerate }) {
  return (
    <section className="panel ai-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Gemini Interpreter</p>
          <h2>AI explanation</h2>
        </div>
        <button disabled={loading || disabled} onClick={onGenerate} type="button">
          {loading ? "Generating..." : interpretation ? "Refresh" : "Generate"}
        </button>
      </div>
      {error && <div className="inline-error">{error}</div>}
      {!interpretation && !loading && !error && (
        <p className="summary">Optional. Sends the zone report (without chart rows) to Gemini for a plain-language explanation.</p>
      )}
      {loading && <p className="summary">Gemini is reading the zone report. This usually takes 20–40 seconds...</p>}
      {interpretation && (
        <div className="ai-content">
          <div className="ai-summary">
            <div className="ai-views">
              <span>
                Short-term: <strong>{viewLabel(interpretation.short_term_view)}</strong>
              </span>
              <span>
                Long-term: <strong>{viewLabel(interpretation.long_term_view)}</strong>
              </span>
            </div>
            {interpretation.headline && <p className="ai-headline">{interpretation.headline}</p>}
          </div>
          {SECTIONS.map(([key, title]) =>
            interpretation[key] ? (
              <section className="ai-section" key={key}>
                <h3>{title}</h3>
                <Markdown>{interpretation[key]}</Markdown>
              </section>
            ) : null,
          )}
          <section className="ai-section ai-columns">
            <div>
              <h3>Reasons</h3>
              <PlainList items={interpretation.key_reasons} />
            </div>
            <div>
              <h3>Risks</h3>
              <PlainList items={interpretation.main_risks} />
            </div>
          </section>
          {interpretation.data_cited?.length > 0 && (
            <details className="ai-data">
              <summary>Data cited</summary>
              <PlainList items={interpretation.data_cited} />
            </details>
          )}
        </div>
      )}
    </section>
  );
}

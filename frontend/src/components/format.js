export const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
export const dollar = (value) => (isNumber(value) ? `$${value.toFixed(2)}` : "-");
export const percent = (value, digits = 1) => (isNumber(value) ? `${(value * 100).toFixed(digits)}%` : "-");
export const signedPercent = (value, digits = 1) =>
  isNumber(value) ? `${value > 0 ? "+" : ""}${(value * 100).toFixed(digits)}%` : "-";
export const points = (value) => {
  if (!isNumber(value)) return "-";
  const rounded = Math.round(value * 100);
  return rounded === 0 ? "±0 pts" : `${rounded > 0 ? "+" : ""}${rounded} pts`;
};
export const signedR = (value) => (isNumber(value) ? `${value > 0 ? "+" : ""}${value.toFixed(2)}R` : "-");
export const fixed = (value, digits = 2) => (isNumber(value) ? value.toFixed(digits) : "-");
export const range = (zone) => (zone ? `${dollar(zone.low)} – ${dollar(zone.high)}` : "-");
export const ordinal = (fraction) => {
  if (!isNumber(fraction)) return "-";
  const value = Math.round(fraction * 100);
  const suffix = value % 100 >= 11 && value % 100 <= 13 ? "th" : { 1: "st", 2: "nd", 3: "rd" }[value % 10] || "th";
  return `${value}${suffix}`;
};

// Plain-language position of the current price relative to a zone.
export function zonePosition(zone, kind) {
  if (!zone) return "-";
  const distance = Math.abs(zone.distance_pct ?? 0);
  switch (zone.position) {
    case "INSIDE":
      return "Price is inside";
    case "BELOW":
      return `${percent(distance)} below the zone`;
    case "ABOVE":
      return `${percent(distance)} above the zone`;
    default:
      return kind === "buy"
        ? `${percent(distance)} above · ${fixed(zone.distance_atr, 1)} ATR away`
        : `needs +${percent(distance)} · ${fixed(zone.distance_atr, 1)} ATR away`;
  }
}

export function evaluatePriceAction(plan, price, entryPrice = null) {
  if (!plan || !Number.isFinite(price) || price <= 0) return null;
  const rule = plan.rules.find((item) => (
    item.operator === "at_or_below" ? price <= item.trigger_price : price >= item.trigger_price
  ));
  if (rule?.action.includes("PROFIT") && Number.isFinite(entryPrice) && entryPrice > 0 && price <= entryPrice) {
    return {
      ...rule,
      action: "REVIEW_RESISTANCE_OR_EXIT",
      guidance: "This reaches a profit-review level from the forecast reference, but it is at or below your entry. Consider whether to reduce risk or continue holding based on resistance and trend confirmation; selling here would not realize a gain before costs.",
    };
  }
  return rule || { action: plan.default_action, guidance: plan.default_guidance };
}

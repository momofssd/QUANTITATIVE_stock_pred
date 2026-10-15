import test from "node:test";
import assert from "node:assert/strict";
import { evaluatePriceAction } from "./priceAction.js";

const plan = {
  rules: [
    { operator: "at_or_below", trigger_price: 90, action: "EXIT" },
    { operator: "at_or_below", trigger_price: 95, action: "WATCH_BOUNCE" },
    { operator: "at_or_above", trigger_price: 120, action: "TAKE_PROFIT_2" },
    { operator: "at_or_above", trigger_price: 110, action: "TAKE_PROFIT_1" },
    { operator: "at_or_above", trigger_price: 105, action: "CONFIRM_BREAKOUT" },
  ],
  default_action: "WAIT", default_guidance: "Wait for confirmation.",
};

test("inclusive triggers choose stop before bounce and highest profit level first", () => {
  for (const [price, action] of [[80, "EXIT"], [90, "EXIT"], [95, "WATCH_BOUNCE"], [100, "WAIT"], [105, "CONFIRM_BREAKOUT"], [110, "TAKE_PROFIT_1"], [120, "TAKE_PROFIT_2"], [130, "TAKE_PROFIT_2"]]) {
    assert.equal(evaluatePriceAction(plan, price).action, action);
  }
});

test("invalid price or missing plan cannot generate guidance", () => {
  for (const price of [NaN, Infinity, 0, -1]) assert.equal(evaluatePriceAction(plan, price), null);
  assert.equal(evaluatePriceAction(null, 100), null);
});

test("profit guidance becomes an exit review when the user's entry is higher", () => {
  assert.equal(evaluatePriceAction(plan, 110, 120).action, "REVIEW_RESISTANCE_OR_EXIT");
  assert.equal(evaluatePriceAction(plan, 110, 110).action, "REVIEW_RESISTANCE_OR_EXIT");
  assert.equal(evaluatePriceAction(plan, 110, 100).action, "TAKE_PROFIT_1");
  assert.equal(evaluatePriceAction(plan, 90, 120).action, "EXIT");
});

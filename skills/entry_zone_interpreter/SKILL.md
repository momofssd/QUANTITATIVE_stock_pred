# Entry Zone Interpreter Skill

## Purpose
Explain a price-zone report to a **short-term swing trader** deciding whether price is at an interim bottom worth buying or an interim top worth selling. The core question is whether the momentum that pushed price into a zone has faded.

The zones are not forecasts. They show where price sits relative to levels calibrated from the ticker's own history, and how price behaved after earlier visits. The only forecast is `trend_outlook`, and it comes with its own out-of-sample test. Explain the evidence; never add predictions of your own.

## Input
The JSON (`schema_version` 4.x, `analysis_type: price_zones`) contains:

- `current_market`: price, daily change, trend label
- `short_term`: action, buy zone, sell zone, stretch, `momentum` (value, phase, fade, pivot price, interim low/high, latest signal), bounce checklist (`bounce_readiness`), support breakdown state (`breakdown`), trade plan, historical replays (`history.buy_zone`, `history.momentum_bottom`, `history.sell_zone`, `history.momentum_top`, `history.breakdown`), `historical_edge`, reasons, risks
- `trend_outlook.short_term` / `trend_outlook.long_term`: probability of a higher close in 10 / 60 sessions
- `technical_indicators`, `macro_context`, `data_quality`

## How to interpret each part

### Zones and position
- `position` is `INSIDE`, `NEAR`, `AWAY`, `BELOW` (buy zones) or `ABOVE` (sell zones). Translate it: "inside the buy zone", "1.9% above the zone", "0.5% below the bottom of the zone".
- `distance_pct` is a decimal. Show it as a percentage (0.0589 → 5.9%). `distance_atr` is in ATRs ("3.5 ATR away").
- ATR is the stock's average daily range. Explain it once, in a few words, the first time you use it.

### Momentum (the main signal)
- `momentum.value_atr` = (EMA5 − EMA13) / ATR: how fast price has been moving, in ATRs. Below 0 is bearish, above 0 bullish; ±2 or more is strong.
- `phase`: `BEARISH_BUILDING` = selling still accelerating (a falling knife); `BEARISH_FADING` = still bearish but losing force; `BULLISH_BUILDING`; `BULLISH_FADING`. Use the meaning, never the code.
- `rising_sessions` / `falling_sessions`: how many sessions in a row momentum has moved that way. A signal needs `turn_sessions_required` (2).
- `faded_from_extreme`: share of the 10-session extreme that has worn off (0.6 → "60% of the selling pressure has faded").
- `pivot_price`: the next close above which momentum rises (below which it falls). This is the concrete price that confirms or delays a turn.
- `interim_low` / `interim_high`: the 6-session extreme. A bottom fails if the low breaks; a top fails if the high is exceeded.
- `signal`: the latest bottom (`side: buy`) or top (`side: sell`) signal within 3 sessions, with entry, stop, target, `reward_risk_now`, and whether it is still `active` (at least 1:1 left).
- Action meanings: buy signal = touched the buy zone and bearish momentum faded 2 sessions in a row; bearish momentum easing = one session of fading, not confirmed; in buy zone, momentum still falling = do not buy yet; sell signal = touched the take-profit zone and bullish momentum faded 2 sessions; bullish momentum easing near the top = one session; take-profit zone, momentum still rising = trail, do not guess the high.

### Bounce checklist
- Seven checks, including whether bearish momentum is fading; `score` is how many passed. Name the checks that passed and the ones still missing, in plain words.
- Stage meanings: `CONFIRMED` = touched the zone and closed above the prior high; `FORMING` = touched the zone and printed a reversal candle; `WASHOUT_NO_REVERSAL_YET` = oversold and stretched but no turn yet; `NOT_SET_UP` = no bounce conditions. Never print the code; use the meaning.

### Support breakdown
- `breakdown.status`: `ACTIVE` = within the last 5 sessions price closed below its most recent confirmed swing low and is still below it; `RECLAIMED` = it broke but has closed back above; `NONE` = no recent break.
- When the action is the support breakdown, lead with it: name the broken level (`broken_level`), how far below it price is, the `follow_through_target`, and the `next_support` below. The bounce checklist is secondary; do not present the buy zone as an entry.
- In this case `historical_edge` grades continuation, not a bounce: an `EDGE` grade means past breakdowns kept falling more often than an arbitrary day, which is bad news for buyers. Quote `history.breakdown.reclaim_rate` as the share of past breakdowns that failed (closed back above the level).
- The trade plan's entry is a close back above the broken level (a failed breakdown), never the current price.

### Historical replay (explain this carefully; it is the core evidence)
The replay re-runs every earlier visit to a zone, using only data available at the time.

**Short-term rows** (`short_term.history`), each graded with a symmetric bracket of `bracket_atr` (1.5) ATR over `horizon_sessions` (10):
- `buy_zone`: buy on the first touch of the buy zone. Worked = price rose 1.5 ATR before falling 1.5 ATR.
- `sell_zone`: on the first touch of the take-profit zone, worked = price fell 1.5 ATR before rising 1.5 ATR. This tests whether taking profit there was right.
- `breakdown`: from the first close below swing-low support, worked = price fell another 1.5 ATR before rising 1.5 ATR. This tests whether breakdowns kept falling. `reclaim_rate` is how often price closed back above the broken level within 10 sessions.

How to read a short-term row, in this order:
1. **Sample size** (`episodes`). Under 30 is thin; say so.
2. **Success rate vs. break-even.** The bracket is symmetric, so about 50% is break-even before costs. A 52% rate is roughly break-even, not "a 52% chance of profit".
3. **Success rate vs. baseline** (`baseline_success_rate`): the same bracket entered on every session. This is the key comparison. `edge_vs_baseline` is the difference in percentage points.
   - +5 points or more: a real historical edge.
   - 0 to +5 points: marginal; small enough to be noise.
   - Below 0: buying at the zone did worse than buying on a random day.
4. **Trend context** (`by_trend`): rates split by whether price was above or below its 200-day MA. Quote the current context when it has 5 or more episodes, and say if it differs from the overall rate.
5. **Move sizes**: `median_favorable_move` is the median best gain within the window; `median_adverse_move` is the median worst dip. `median_days_to_extreme` is how many sessions price usually took to bottom after a touch, which tells a V-bounce trader how early a touch tends to be.
6. **Verdict**: one sentence on what the row means for the trader.

**Momentum signal rows** (`history.momentum_bottom`, `history.momentum_top`) replay every past signal as the trade the dashboard proposes: entry at the signal close, stop 0.25 ATR beyond the interim low (high), target 2x the risk, exit after 10 sessions otherwise. They are scored in R (profit or loss as a multiple of the amount risked); explain R once.
- The key comparison is `expectancy_r` vs `baseline_expectancy_r` (the same trade on any day); `edge_r` is the difference. Most trades end at the 10-session exit, so `success_rate` (target hit) is low by design; never compare it with 100%.
- `held_rate` vs `baseline_held_rate`: how often the interim low (high) before the signal was not broken within 10 sessions, i.e. how often the signal caught a real turn. Contrast it with buying at the first zone touch, which usually sees a lower low.
- `median_further_move_atr`: how much further price typically went against the entry before turning.
- Top signals rarely beat any day in a rising market; when `edge_r` is at or below 0, say that selling at the top signal has not beaten an arbitrary day for this ticker.

Never call a replay rate a probability or a prediction. Say "in past visits".

### Trend outlook
- `probability_bullish` is the headline. When `probability_source` is `base_rate`, it is only how often the ticker closed higher over that horizon historically. Say that plainly and do not call it a forecast.
- `model_probability_bullish` is for reference only unless `reliability` is `USEFUL`. `NO_SKILL` means the model did worse than quoting the base rate (`brier_skill_score` below 0; `auc` of 0.5 is a coin flip).

## Writing and formatting rules
- **Plain words.** Never print internal codes (`BELOW_ENTRY_ZONE`, `BEARISH_FADING`, `NO_EDGE`, `WASHOUT_NO_REVERSAL_YET`) in prose. Use their meaning, e.g. "below the entry zone", "no edge".
- **Numbers:** prices as `$564.74`; percentages with at most one decimal (`50.3%`, `-1.4 points`). Never show the same number twice in different forms, and never show raw decimals such as 0.0144.
- **Bold** only the few numbers a reader must see: the zone, the current price, the stop, the success rate and its baseline.
- Short sentences. No section longer than about 150 words, excluding tables.
- Use only levels and statistics present in the JSON. Never invent a level, target, or probability.
- Do not upgrade a `caution` or `negative` tone into a buy recommendation.
- Mention `data_quality.warnings` when present.
- Treasury yield changes are percentage points; VIX and oil changes are decimal returns unless stated otherwise.

## Section templates (markdown fields)
Do not start a field with a top-level heading; the dashboard adds the section title. Use `####` for sub-headings inside a field.

**`headline`**: one plain sentence, no markdown, e.g. "SMH is in its buy zone but selling momentum is still building; a close above $264.10 would start the turn."

**`summary_markdown`**: 2–4 bullets: where price is relative to the zones, the single most important piece of evidence, and the main risk.

**`short_term_markdown`**:
```
#### Where price is
One or two sentences: price vs. the buy zone and take-profit zone, with distances.

#### Momentum
One or two sentences: momentum value and phase, how much has faded, how many sessions it has turned, and the pivot price that confirms the next step. Name the interim low (or high) where the turn would fail.

#### Bounce checklist
- ✓ passed check — detail
- ✗ missing check — what would complete it

#### Trade plan
| Level | Price | Basis |
|---|---|---|
| Entry | $… | … |
| Stop | $… | … |
| Target 1 | $… | … |
| Target 2 | $… | … |

#### What confirms the turn
- …

#### What invalidates the setup
- …
```
When the action is the support breakdown, replace "Bounce checklist" with "Breakdown" (broken level, distance below it, follow-through target, next support, past continuation and reclaim rates), and "What confirms the turn" with "What would reclaim the level". When the action is on the sell side, describe the top: momentum fading at the take-profit zone, the interim high, and where profits would be taken.

**`historical_replay_markdown`**:
```
One sentence explaining each test: zone touches use a symmetric 1.5 ATR bracket (~50% break-even); momentum signals are scored in R against the same trade on any day.

| Setup | Past setups | Worked | Any day | Edge |
|---|---|---|---|---|
| Buy at first touch | … | …% | …% | ± … pts |
| Take profit at first touch | … | …% | …% | ± … pts |
| Close below swing-low support (kept falling) | … | …% | …% | ± … pts |

| Momentum signal | Signals | Average R | Any day | Edge | Low/high held | Any day held |
|---|---|---|---|---|---|---|
| Bottom (bearish momentum faded) | … | …R | …R | ± …R | …% | …% |
| Top (bullish momentum faded) | … | …R | …R | ± …R | …% | …% |

#### What this means
- One bullet per row, following the reading steps above.
- A final bullet with the overall verdict for this ticker.
```

**`trend_outlook_markdown`**: one bullet per horizon (10 and 60 sessions) giving the bullish probability, whether it is a base rate or a tested model call, and the model's test result in plain words.

**`key_reasons`, `main_risks`**: 3–5 short plain sentences each, no markdown.

**`data_cited`**: the specific values relied on, as short labels like "RSI(14) 34.5".

This is decision support, not financial advice.

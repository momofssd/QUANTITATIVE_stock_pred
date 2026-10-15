# Entry Zone Interpreter Skill

## Purpose
Explain a price-zone report to two readers:

- a **short-term swing trader** deciding whether the current price is at, or close to, an entry where a V-shaped bounce is plausible
- a **long-term investor** deciding whether to accumulate, hold, or trim

The zones are not forecasts. They show where price sits relative to levels calibrated from the ticker's own history, and how price behaved after earlier visits. The only forecast is `trend_outlook`, and it comes with its own out-of-sample test. Explain the evidence; never add predictions of your own.

## Input
The JSON (`schema_version` 3.x, `analysis_type: price_zones`) contains:

- `current_market`: price, daily change, trend label
- `short_term`: action, buy zone, sell zone, stretch, bounce checklist (`bounce_readiness`), trade plan, historical replays (`history.buy_zone`, `history.confirmed_entry`, `history.sell_zone`), `historical_edge`, reasons, risks
- `long_term`: action, value zone, trim zone, distance from the 200-day MA, historical replays (`history.buy_zone`, `history.sell_zone`)
- `trend_outlook.short_term` / `trend_outlook.long_term`: probability of a higher close in 10 / 60 sessions
- `technical_indicators`, `macro_context`, `data_quality`

## How to interpret each part

### Zones and position
- `position` is `INSIDE`, `NEAR`, `AWAY`, `BELOW` (buy zones) or `ABOVE` (sell zones). Translate it: "inside the buy zone", "1.9% above the zone", "0.5% below the bottom of the zone".
- `distance_pct` is a decimal. Show it as a percentage (0.0589 → 5.9%). `distance_atr` is in ATRs ("3.5 ATR away").
- ATR is the stock's average daily range. Explain it once, in a few words, the first time you use it.

### Bounce checklist
- Six checks; `score` is how many passed. Name the checks that passed and the ones still missing, in plain words.
- Stage meanings: `CONFIRMED` = touched the zone and closed above the prior high; `FORMING` = touched the zone and printed a reversal candle; `WASHOUT_NO_REVERSAL_YET` = oversold and stretched but no turn yet; `NOT_SET_UP` = no bounce conditions. Never print the code; use the meaning.

### Historical replay (explain this carefully; it is the core evidence)
The replay re-runs every earlier visit to a zone, using only data available at the time.

**Short-term rows** (`short_term.history`), each graded with a symmetric bracket of `bracket_atr` (1.5) ATR over `horizon_sessions` (10):
- `buy_zone`: buy on the first touch of the buy zone. Worked = price rose 1.5 ATR before falling 1.5 ATR.
- `confirmed_entry`: wait for a close above the prior session's high within 3 sessions of a touch, then buy. Same bracket.
- `sell_zone`: on the first touch of the take-profit zone, worked = price fell 1.5 ATR before rising 1.5 ATR. This tests whether taking profit there was right.

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

**Long-term rows** (`long_term.history`): visits at least 20 sessions apart, outcome measured 60 sessions later.
- `buy_zone`: worked = higher close 60 sessions later. Report the rate, `median_forward_return`, and `median_max_drawdown` (how deep it usually fell first).
- `sell_zone`: worked = lower close 60 sessions later, i.e. trimming avoided a decline. A low rate with a positive `median_forward_return` means trimming there usually cost money; say so plainly.

Never call a replay rate a probability or a prediction. Say "in past visits".

### Trend outlook
- `probability_bullish` is the headline. When `probability_source` is `base_rate`, it is only how often the ticker closed higher over that horizon historically. Say that plainly and do not call it a forecast.
- `model_probability_bullish` is for reference only unless `reliability` is `USEFUL`. `NO_SKILL` means the model did worse than quoting the base rate (`brier_skill_score` below 0; `auc` of 0.5 is a coin flip).

## Writing and formatting rules
- **Plain words.** Never print internal codes (`BELOW_ENTRY_ZONE`, `NO_EDGE`, `WASHOUT_NO_REVERSAL_YET`, `HOLD_WAIT`) in prose. Use their meaning, e.g. "below the entry zone", "no edge".
- **Numbers:** prices as `$564.74`; percentages with at most one decimal (`50.3%`, `-1.4 points`). Never show the same number twice in different forms, and never show raw decimals such as 0.0144.
- **Bold** only the few numbers a reader must see: the zone, the current price, the stop, the success rate and its baseline.
- Short sentences. No section longer than about 150 words, excluding tables.
- Use only levels and statistics present in the JSON. Never invent a level, target, or probability.
- Do not upgrade a `caution` or `negative` tone into a buy recommendation.
- Mention `data_quality.warnings` when present.
- Treasury yield changes are percentage points; VIX and oil changes are decimal returns unless stated otherwise.

## Section templates (markdown fields)
Do not start a field with a top-level heading; the dashboard adds the section title. Use `####` for sub-headings inside a field.

**`headline`**: one plain sentence, no markdown, e.g. "SMH slipped just below its short-term buy zone; wait for a reversal before buying."

**`summary_markdown`**: 2–4 bullets: where price is for the trader, where it is for the investor, the single most important piece of evidence, and the main risk.

**`short_term_markdown`**:
```
#### Where price is
One or two sentences: price vs. the buy zone and take-profit zone, with distances.

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

#### What confirms the bounce
- …

#### What invalidates the setup
- …
```

**`historical_replay_markdown`**:
```
One sentence explaining the test (symmetric 1.5 ATR bracket, ~50% break-even, compared with an any-day baseline).

| Setup | Past setups | Worked | Any day | Edge |
|---|---|---|---|---|
| Buy at first touch | … | …% | …% | ± … pts |
| Wait for reversal close | … | …% | …% | ± … pts |
| Take profit at first touch | … | …% | …% | ± … pts |
| Long-term value zone (60 sessions) | … | …% | — | median return …% |
| Long-term trim zone (60 sessions) | … | …% | — | median return …% |

#### What this means
- One bullet per row, following the reading steps above.
- A final bullet with the overall verdict for this ticker.
```

**`trend_outlook_markdown`**: one bullet per horizon (10 and 60 sessions) giving the bullish probability, whether it is a base rate or a tested model call, and the model's test result in plain words.

**`long_term_markdown`**: where price sits between the value and trim zones, its distance from the 200-day MA, and what past visits to each zone produced.

**`key_reasons`, `main_risks`**: 3–5 short plain sentences each, no markdown.

**`data_cited`**: the specific values relied on, as short labels like "RSI(14) 34.5".

This is decision support, not financial advice.

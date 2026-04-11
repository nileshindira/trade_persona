---
description: Roadmap for fixing Trade Persona Analyzer - audit fixes + new features
---

# 🗺️ Trade Persona Analyzer — Fix & Enhancement Roadmap

> **Created:** 2026-03-18  
> **Status:** In Progress  
> **Priority Key:** 🔴 Critical | 🟡 Important | 🟢 Nice-to-Have | 🔵 Futuristic

---

## Phase 1: Critical Metric Accuracy Fixes (Day 1)
*These bugs produce wrong numbers → wrong persona → wrong advice*

### 1.1 🔴 Fix Win Rate Calculation
**File:** `src/metrics_calculator.py` → `calculate_win_rate()`  
**Problem:** Rows with `pnl=0` (unmatched open legs) are counted as losses.  
**Current:** `winning/total_rows = 625/1857 = 33.66%`  
**Should be:** `winning/traded_rows ≈ 55-60%`  
**Fix:** Filter `df[df['pnl'] != 0]` before computing win rate.  
**Impact:** Cascades into persona traits, LLM narrative, hard flags, and final verdict.

### 1.2 🔴 Fix Total Trades Count
**File:** `src/metrics_calculator.py` → `calculate_all_metrics()`  
**Problem:** `total_trades = len(df)` counts CSV rows, not trade decisions.  
**Fix:** Count only rows where `pnl != 0`, OR count unique FIFO paired trades.  
**Impact:** Fixes avg_trades_per_day → fixes overtrading flag → fixes LLM context.

### 1.3 🔴 Fix Open Position PnL Sign
**File:** `src/data_processor.py` → `pair_trades()` lines 526-552  
**Problem:** Open BUY positions get `pnl = (0 - price) * qty` (always negative).  
**Fix:** Set `pnl = 0` for open positions (no mark-to-market in FIFO mode).

### 1.4 🔴 Remove Duplicate Method Definitions
**File:** `src/llm_analyzer.py`  
**Problem:** `_beautify_recommendations_html` (lines 143 vs 683) and `_compute_verdict_ceiling` (lines 100 vs 743) are defined twice.  
**Fix:** Delete the duplicate (later) definitions, keep the more complete (earlier) ones.

### 1.5 🟡 PnL Reconciliation Check
**File:** `main.py`  
**Problem:** FIFO PnL (₹6,42,491) vs MTM PnL (₹6,14,020) — ₹28,471 gap, never flagged.  
**Fix:** Add reconciliation warning when gap > 10%.

---

## Phase 2: Risk & Behavioral Metric Rework (Day 2-3)
*Fix how the system measures risk, loss patterns, and behavioral biases*

### 2.1 🔴 Rework Risk Appetite Calculation
**File:** `src/metrics_calculator.py` → `_calc_risk_appetite()`  
**Current Logic:** Based on max drawdown + avg trade value + PnL std. This implicitly rewards diversification.  
**New Logic (per user spec):**
- Should be based on **average loss size** relative to capital/avg win
- Should consider **single largest loss** as a risk tolerance indicator  
- Should NOT factor in diversification (that's an investing metric, not trading)
- Formula: `risk_appetite = f(avg_loss_size, max_single_loss, loss_frequency, position_sizing_on_losers)`

### 2.2 🔴 Sharpen Risk Appetite vs Risk Handling Separation
**Current:** Both use profit factor and loss metrics — they overlap.  
**New Definition:**
| Dimension | Measures | Inputs |
|-----------|----------|--------|
| **Risk Appetite** | *Willingness* to take risk (how much heat do you accept?) | Avg loss %, max single loss %, position size variance on losers, overnight hold frequency |
| **Risk Handling** | *Skill* at managing risk (how well do you cut losers?) | Loss-cutting speed, trailing stop effectiveness, winner/loser hold ratio, drawdown recovery time |

### 2.3 🔴 Intraday Risk vs Overnight Risk
**File:** `src/metrics_calculator.py` (NEW section)  
**Logic:**
- Classify each trade as INTRADAY (opened and closed same day) or OVERNIGHT (held across sessions)
- Compute separate metrics for each: win_rate, avg_pnl, max_loss, risk/reward
- Detect overnight gap risk: trades that lose >X% overnight but are profitable intraday
- Feed this to LLM as a key behavioral insight

### 2.4 🟡 Sectoral Profit Analysis
**File:** `src/metrics_calculator.py` (NEW section)  
**Existing data:** `industry` column from DB enrichment  
**Logic:**
- Group trades by sector/industry
- Compute per-sector: total PnL, win rate, avg holding, trade count
- Identify "money sectors" (high PnL) vs "leak sectors" (consistent losses)
- Find sector rotation patterns (does trader switch sectors after losses?)

---

## Phase 3: FOMO, Anchor Bias & Loss Averaging Rework (Day 3-4)
*Fix behavioral pattern detection to match real trading psychology*

### 3.1 🔴 Redefine FOMO Detection
**File:** `src/pattern_detector.py` → `detect_fomo_trading()`  
**Current Logic:** Buying after large price moves or at 52-week highs.  
**New Logic (per user spec):**
FOMO = entering after an **extended continuous move with speed and no consolidation**:
1. Detect extended bull/bear runs: >3 consecutive green/red days OR >5% move in <3 sessions
2. Check if there was NO consolidation (range compression) before entry
3. Check if entry was at the **tail end** of the move (price far from moving average)
4. Cross-reference with trader's entry timing — late entries into fast moves = FOMO
5. **New metric:** FOMO Score = f(move_speed, consolidation_absence, entry_lateness, position_size_inflation)

### 3.2 🔴 Add Anchor Bias Detection (NEW)
**File:** `src/pattern_detector.py` (NEW method)  
**Definition:** Trader anchors to a previous price and refuses to adapt:
- Repeated buys at/near the same strike/price even after the market has moved
- Refusing to take a loss because "it was at X price before"
- Setting targets based on historical highs rather than current structure
**Detection:**
1. For each symbol, find repeated entries at similar prices after the stock has moved >5%
2. Check if stop-losses are placed based on entry price rather than market structure
3. Detect "round number anchoring" — entries/exits clustering at round numbers

### 3.3 🟡 Improve Loss Averaging Detection
**File:** `src/metrics_calculator.py` → `_extract_evidence_packs()`  
**Current:** Only checks same-day multiple buys at declining prices.  
**Enhancement:**
- Cross-day averaging: buying the same symbol across multiple days at lower prices
- Quantity escalation: each subsequent buy is larger (doubling down)
- Compute total capital at risk from averaging down
- Flag the outcome: did the averaging pay off or compound the loss?

---

## Phase 4: Option Writer Intelligence (Day 4-5)
*Differentiate option writer behavior from buyer behavior*

### 4.1 🔴 Detect Option Writer vs Buyer
**File:** `src/metrics_calculator.py` (NEW section)  
**Logic:**
- If first transaction is SELL on an option → Option Writer
- If first transaction is BUY on an option → Option Buyer
- Track writer-specific metrics: premium collected, assignment risk, max potential loss
- Differentiate by option type: CE writer (bearish), PE writer (bullish)

### 4.2 🟡 Price Category Action Analysis
**Logic for option writers:**
- ATM options (strike ≈ spot): High premium, high risk → aggressive writing
- OTM options (strike far from spot): Low premium, low risk → conservative writing  
- ITM options (strike past spot): Rare, indicates specific strategy
- Map each trade to ATM/OTM/ITM category using DB `close` as spot
- Compute win rate and avg PnL per category

### 4.3 🟡 Time vs Price Analysis
**Time Analysis:**
- Days to expiry at entry → does trader write early or close to expiry?
- Theta decay capture → how much premium decays during the hold
- Time-of-day patterns for option trades

**Price Analysis:**
- Premium captured as % of spot price
- Strike selection relative to ATM
- Premium change during holding period

---

## Phase 5: Timing & Entry/Exit Efficiency Rework (Day 5-6)
*Precise measurement of entry/exit quality*

### 5.1 🔴 Entry Efficiency = Proximity to Low
**Current:** Uses `(high - price) / (high - low)` for BUY entries.  
**Enhancement:**
- For BUY: How close to the day's low did the entry happen? `(price - low) / (high - low)` → lower = better
- For SELL: How close to the day's high did the exit happen? `(high - price) / (high - low)` → lower = better
- **Intraday granularity:** If per-minute data is available, use minute-level high/low instead of daily

### 5.2 🟡 Post-Entry Move Analysis
- After entry, what was the maximum favorable excursion (MFE)?
- After entry, what was the maximum adverse excursion (MAE)?
- Did the trader exit at peak MFE or leave money on the table?
- Compute: Capture Ratio = actual_pnl / MFE

---

## Phase 6: Strategy Inference Engine (Day 6-8)
*Use AI to figure out what the trader is actually doing*

### 6.1 🔴 Trading Strategy Detection
**Approach:**
1. Cluster trades by: holding period, instrument type, entry pattern, P&L characteristics
2. Use the pattern clusters to infer strategy names:
   - Premium selling (option writer, short hold, small wins, rare big losses)
   - Momentum/Breakout (buys at highs, large wins on trends, whipsawed in ranges)
   - Mean reversion (buys at lows, sells at highs, works in ranges, fails in trends)
   - Event-based (trades cluster around earnings, results, news)
   - Scalping (very short hold, high frequency, small per-trade)
3. Assign confidence % to each detected strategy
4. Show which strategy is making money and which is losing

### 6.2 🟡 What-If Simulation (Trade transformation)
**User requirement:** "If we change 4-5 things, how would performance change"
**Implementation:**
- Simulate: "What if max loss was capped at X?"  → recalculate P&L with stop-loss
- Simulate: "What if holding period was extended by Y days?" → use historical prices from DB
- Simulate: "What if the trader only traded their top 3 sectors?"
- Simulate: "What if FOMO trades were avoided?"
- Present before/after metrics comparison

### 6.3 🟡 Loss Pattern Mining
**User requirement:** "Find patterns in loss-making days"
**Logic:**
- Correlate losing days with: day of week, market trend, sector, time of entry, position size
- Find "toxic combinations" — e.g., "Mondays + Options + First-hour entries" = 80% loss rate
- Present as actionable rules: "Avoid trading X on Y when Z"

---

## Phase 7: LLM & Report Improvements (Day 8-10)

### 7.1 🔴 Use Claude/GPT Instead of or Alongside Ollama
**Current:** Ollama (local) or OpenRouter.  
**Enhancement:**
- Add Claude API support (Anthropic)
- Add direct OpenAI GPT-4 support
- Config-switchable between providers
- Use Claude for deep behavioral analysis (long context)
- Use GPT-4 for structured JSON output (reliable formatting)

### 7.2 🔴 Reduce Performance Basket to 4 Categories
**Current:** 8 persona traits, 6 persona types.  
**New (per user spec):** 4 clear performance baskets:
1. **Identity**: Who are you as a trader? (type, strategy, style)
2. **Strengths**: Where are you making money and why?
3. **Mistakes**: Where are you losing money and what patterns cause it?
4. **Improvement Plan**: What specific changes would transform your results?

### 7.3 🔴 Name the Trader Based on Persona
**Examples:**
- "The Precision Sniper" — high win rate, selective entry, low frequency
- "The Premium Farmer" — option writer, steady income, rare blowups
- "The FOMO Chaser" — enters late moves, high frequency, inconsistent
- "The Grinder" — scalper, high volume, paper-cut losses accumulate
- Generate this name dynamically from the trait scores

### 7.4 🟡 Core Report Must-Haves (User Specified)
The report MUST clearly address:
1. **"Who am I as a trader?"** — identity, strategy, style, unique name
2. **"Where am I good, where am I bad?"** — with specific symbols, sectors, patterns
3. **"What are my specific mistakes?"** — with evidence from actual trades
4. **"How can I improve?"** — concrete 5-session and 30-day action plans

---

## Phase 8: Data Infrastructure (Week 2-3)

### 8.1 🟡 Per-Minute Market Data Integration
**User requirement:** "Give market data, per minute data of the last 2-3 days including T+1"
- Connect to per-minute data source (existing DB or API)
- Use for precise entry/exit efficiency calculation
- Use for MFE/MAE computation
- Enable intraday strategy pattern detection

### 8.2 🟡 Options Data (OI, Volume) Integration
**User requirement:** "If options, then options data (give OI, volume)"
- Fetch OI and volume data for option strikes from DB
- Detect: Did trader write at high-OI strikes? (Smart money alignment)
- Detect: Did OI increase/decrease during the holding period?
- Use for strategy inference (OI buildup = trend, OI unwinding = reversal)

### 8.3 🟡 1-Year Historical Data Analysis
**User requirement:** "Give one year data to figure out trading strategy and patterns"
- Extend the lookback window from current 6 months to 12+ months
- Detect seasonal patterns (month-of-year performance)
- Detect market-regime patterns (bull vs bear vs range performance)
- Build a "trader evolution" timeline — are they getting better or worse?

---

## Phase 9: Futuristic Features

### 9.1 🔵 Audio Trade Journal (Record Button)
**User requirement:** "Record button app, person spoke about the trade, audio trade journal"
- Mobile/web app with record button
- Speech-to-text transcription
- Parse trade intent from natural language
- Match audio journal entries to actual trades
- Compare stated rationale vs actual outcome

### 9.2 🔵 Indira Trade Recommendation Hook
**User requirement:** "Can portfolio existing Indira trade recommendation into it"
- Ingest Indira's trade recommendations
- Compare trader's actual trades against recommendations
- Score: compliance %, timing deviation, result comparison
- Detect if trader is following recommendations but with worse execution

---

## Implementation Order

```
Week 1 (Critical Fixes):
  Day 1:  Phase 1 (1.1-1.5) — Metric accuracy fixes
  Day 2:  Phase 2 (2.1-2.3) — Risk & behavioral rework
  Day 3:  Phase 3 (3.1-3.3) — FOMO, Anchor Bias, Loss Averaging
  Day 4:  Phase 4 (4.1-4.3) — Option writer intelligence
  Day 5:  Phase 5 (5.1-5.2) — Entry/exit timing rework

Week 2 (Intelligence Layer):
  Day 6-7: Phase 6 (6.1-6.3) — Strategy inference & simulation
  Day 8:   Phase 7 (7.1-7.4) — LLM & report improvements
  
Week 3 (Data & Infrastructure):
  Day 9-10: Phase 8 (8.1-8.3) — Enhanced data sources

Future:
  Phase 9 — Audio journal, Indira integration
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/metrics_calculator.py` | Win rate fix, total trades fix, risk appetite rework, intraday/overnight risk, sectoral analysis, option writer metrics, entry/exit rework, loss patterns, what-if simulation |
| `src/data_processor.py` | Open position PnL fix, option symbol parsing fix |
| `src/pattern_detector.py` | FOMO rework, anchor bias detection, improved loss averaging |
| `src/llm_analyzer.py` | Remove duplicates, Claude/GPT support, new prompt with 4 categories, trader naming |
| `src/report_generator.py` | 4-category report structure |
| `main.py` | PnL reconciliation, pipeline updates |
| `config.yaml` | Claude/GPT API config, new thresholds |

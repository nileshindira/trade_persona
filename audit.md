# Trading Persona Analyzer - System Audit Report

## 1. System Overview & Architecture
The **Trading Persona Analyzer** is a Python-based pipeline designed to evaluate retail trader behavior, quantify risk, detect emotional or destructive trading patterns, and generate a comprehensive LLM-powered report (HTML/JSON).

### 1.1 Wiring Architecture
The primary entry point is `main.py`, which orchestrates the following sequential pipeline:
1. **Data loading and Enrichment** (`TradingDataProcessor`): Reads flat files (CSV), connects to multiple PostgreSQL databases (`stock_data` and `market_data`), and merges technical indicators, symbol clustering, and market indices conditions.
2. **Trade Pairing & PnL** (`TradingDataProcessor`): Processes raw execution logs using a FIFO-based mechanism to pair opening and closing legs, generating realized PnL and segregating open position datasets.
3. **Metrics Calculation** (`TradingMetricsCalculator`): Derives comprehensive financial metrics (Sharpe, Max Drawdown), behavioral tags (Asset clustering), and contextual performance (News, Events, Volume). Includes an MTM (Mark-to-Market) evaluator for Open positions.
4. **Pattern Detection** (`TradingPatternDetector`): Applies deterministic heuristic algorithms to flag psychological breakdowns like FOMO, Revenge Trading, Overtrading, etc.
5. **LLM Insights** (`OllamaAnalyzer`): Consumes quantified metrics and data segments to prompt an LLM (via local Ollama or OpenRouter). It generates behavioral narratives, risk classifications, and concrete recommendations.
6. **Report Generation** (`ReportGenerator`): Marshals the outputs into a Jinja-based HTML report and a JSON data dump.

---

## 2. Pipeline & Data Architectures

### 2.1 Inputs / Sources
- **Trade Executions File:** A CSV/JSON/Excel detailing symbol, price, quantity, transaction_type, and trade_date.
- **P&L History File:** Optional secondary CSV containing daily equity curve context.
- **Local PostgreSQL DB (`stock_data` table)**: Polled for instrument specifics (`t_score`, `f_score`, `is_52week_high`, `atr`, `is_event`, `is_news`, etc.).
- **Local PostgreSQL DB (`market_data` database)**: Analyzed for context (`nse_stock`, `nse_index_ind` tables), e.g., how NIFTY50 behaved during the trade, distancing from 52-week highs, EMA alignments.
- **Static Assets:** `industry_sector_detail.xlsx` (merged for sector EMA).

### 2.2 Processing Architecture
- **Data Cleanup**: Missing fields are dropped, datetimes normalized, positional directions explicitly assigned (e.g., `LONG-OPEN`, `SHORT-OPEN`).
- **Enrichment Merges**: Highly dependent on precise date mapping & symbol parsing. Fallback routines check the first token (`symbol.split(" ")[0]`) when navigating DB queries.
- **Pairing Engine**: Implements dynamic FIFO queue lists (`buy_queue`, `sell_queue`) per symbol to resolve completed legs and calculate hold times. Remaining un-paired units are ported to `OPEN` status.

---

## 3. Metrics Dictionary & Formulas

### 3.1 Core Financial Formulas
- **Net Realized PnL**: `Sum(Trade PnL)` for closed pairs.
- **Profit Factor**: `Gross Profit / Absolute(Gross Loss)`. Infinite if no losing trades exist.
- **Win Rate**: `(Winning Trades / Total Trades) * 100`.
- **Sharpe Ratio**: `(Avg Return - Risk-Free Rate/Days) / StdDev(Returns) * Sqrt(Days)`. (Returns = PnL / Trade Value)
- **Sortino Ratio**: Uses same formula as Sharpe, but replaces overall standard deviation with the standard deviation of losing trades only.
- **Max Drawdown**: Min value of `(Cumulative PnL - Running Max of Cumulative PnL)`.

### 3.2 Advanced Analytics
- **Holding Period Buckets**: Trades split into Scalp (<30m), Intraday (30m to <1d), Swing (1d to 7d), Position (>7d).
- **Behavioral Context Segments**: Win-rate sub-groups built based on filters:
  - Event vs. Non-Event (`is_event = True`).
  - News based (`is_news = True`).
  - Good/Bad Chart Quality (`chart_charts` or `market_behaviour`).
  - Trend Alignment: Determines if action (`BUY` vs `SELL`) matched Nifty 1-week % change.

### 3.3 Open Position "Mark to Market" (MTM)
Calculated depending on instrument type and expiration date:
- Uses intrinsic formulas for Pre-Expiry (Current LTP - Entry Price).
- Applies post-expiry value collapses for out-of-the-money Options based on the target strike via hardcoded `symbol.split(' ')` parsing.

---

## 4. Algorithmic Pattern Detections

The `TradingPatternDetector` leverages strict numerical heuristics:
- **FOMO Trading:** Buy entries directly following massive intraday symbol jumps (>2.5%).
- **Revenge Trading:** Re-entries with increased quantity within 30 minutes of closing a losing position.
- **Overtrading:** >10 explicit transactions per single day natively flagging.
- **Pyramiding:** Multiple sequential BUY allocations on a symbol before a SALE.
- **Scalping:** Average holding duration natively <60 minutes, paired with >30m isolated trades.
- **Chasing Losses:** Progressively increasing positional sizing while in active consecutive drawdowns.
- **Weekend Exposure:** Friday positions held into Monday/Tuesday.
- **Hedging:** Concurrent trades on standard CALLs and PUTs within the same date constraint.

---

## 5. External APIs / LLM Consumer
- **Ollama / OpenRouter**: Environment-configured switch in `llm_provider`. `OllamaAnalyzer` bundles JSON summaries, metrics, and raw trade traces (chunked sequentially to ~100 trades to respect local LLM context limits).
- Prompts orchestrate the creation of JSON blocks to fill specific UI structures: Risk Rating, Behavioral Analysis, Recommended Actions.
- Custom structured formats output JSON dictionaries map contexts (e.g., `"event": { "verdict": "...", "reasoning": "..." }`) and convert generated markdown tables/lists into HTML divs for dashboards.

---

## 6. Identified Gaps & Audit Findings (Critical Fixes Needed)

### 6.1 Catastrophic Code Vulnerabilities
1. **Option Expiry Substring Hardcoding** 
   - *File*: `src/metrics_calculator.py` - `_compute_positions_snapshot()`
   - *Issue*: Lines assuming all open positions are options are uncommented. 
     ```python
     # if len(symbol.split(' ') >2)
     expiry_str = symbol.split(' ')[3]
     strike = symbol.split(' ')[-1]
     option_type = symbol.split(' ')[1]
     ```
     This strictly assumes all tickers have space-delimited fields (e.g., `NIFTY 23FEB 18000 CE`). If an open position is an equity like `RELIANCE EQ` or `TCS`, this throws an `IndexError` bypassing any checks, which completely terminates the pipeline. The `if len(symbol.split(' ') >2)` check was incorrectly commented out.

### 6.2 Data Rigidity & Robustness
1. **Inefficient DB Load Queries**
   - *File*: `src/data_processor.py` 
   - *Issue*: Fetching from Postgres iterates unique symbol lists dynamically into IN clauses, OR iterates rows using `df.iterrows()` line-by-line (e.g. for `getadditionaldata()`). This will cause massive bottlenecking with large trader logs. Should utilize bulk `join` techniques within Pandas or SQL temp tables.
2. **Inconsistent Transaction Typings**
   - The application expects `BUY` and `SALE` but occasionally forces fallbacks for `SELL`. Mismatched strings might bypass metrics groupings entirely and lead to bugs in FIFO matching.
3. **Database Fallbacks Masking Core Issues**
   - Code surrounds DB connects in simple `try/except` and silently drops variables or defaults values instead of actively tracking which rows failed, potentially hiding structural DB drift.

### 6.3 Metric Logical Gaps
1. **MTM Value Calculation for Puts/Calls**
   - In `metrics_calculator.py`, the expiry check relies strictly on string parses (`option_type = symbol.split(' ')[1]`). This logic is extremely fragile to broker formats (Zerodha vs Dhan vs Upstox format discrepancies).
2. **Sharpe Ratio Scaling** 
   - Return values used for Sharpe ratio calculation are based on `PnL / Trade Value`. Since standard deviation of *Return Percentage* varies vastly between Options (highly leveraged) and Equity (1x margin), Sharpe might wildly misinterpret option account risk profiles comparably against standard retail averages.

### 6.4 Missing Guardrails in LLM Integrations
- **LLM Output Strictness**: The parsing of `_analyze_context_performance` handles potential ` ```json ` blocks through manual regex, but if the LLM hallucinated the markdown schema (which smaller local models via Ollama frequently do), it unconditionally falls back to a dummy object `"Analysis Failed"`, erasing dynamic psychological outputs silently for the entire section. 
- **Missing Asset Classification**: `cluster_pct.get("EQUITY", 0)` depends on exact word structures derived from `classify_instrument()`. Unknown tokens resolve to "UNKNOWN" which are subsequently excluded from Pie Chart distributions natively.

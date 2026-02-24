
# 🏗️ Report Architecture & Data Structure

**File:** `src/templates/report.html`  
**Inherits:** `base.html`  
**Styles:** Bootstrap 5 (CDN) + Custom `style` block + `themes/light.css`  
**Scripts:** Chart.js 4.4.3 (CDN)

---

## 1. 🧱 High-Level Layout
The report is a single-page, scrollable dashboard with sticky navigation and collapsible sections. It is heavily componentized using Jinja2 `{% include %}` tags.

### **Navigation (`.section-nav`)**
Sticky tabs linking to:
- Overview (`#overview`)
- Metrics (`#metrics`)
- Charts (`#charts`)
- Persona (`#persona_scores`)
- Positions (`#positions`)
- Patterns & Buckets (`#buckets`)
- Performance (`#performance`)
- Deep Analysis (`#analysis`)
- AI Verdict (`#insights`)

---

## 2. 🧩 Section Breakdown & Key Metric Components

### **A. Overview Section (`#overview`)**
**Components:**
1.  **`components/executive_summary.html`**
    *   **Core Metrics Cards:**
        *   `report.executive_summary.total_trades`
        *   `report.executive_summary.net_pnl` (Green/Red styling)
        *   `report.executive_summary.win_rate`
        *   `report.executive_summary.sharpe_ratio`
        *   `report.executive_summary.profit_factor`
        *   `report.executive_summary.max_drawdown_pct`
        *   `report.executive_summary.risk_level` (Badged: Low/High/Very High)
        *   `report.executive_summary.open_positions_count`
    *   **Hard Risk Flags:**
        *   Iterates `report.hard_flags` (e.g., "Capital At Risk High", "Overtrading").
    *   **P&L Snapshot (Mini Grid):**
        *   Invested: `report.detailed_metrics.total_investment_value_open`
        *   Realized P&L: `report.detailed_metrics.total_realized_pnl`
        *   Unrealized P&L: `report.detailed_metrics.total_unrealized_pnl`
        *   Total P&L: `report.detailed_metrics.total_pnl_combined`
        *   ROI %: Calculated in-template.
        *   Booked SL %: `report.detailed_metrics.close_pos_booked_sl_pct`

2.  **Persona & Risk Profile (`in-line`)**
    *   **Radar Chart:** `canvas#personaRadarChart`
    *   **Top Strengths:** `report.summary_data.strengths`
    *   **Top Weaknesses:** `report.summary_data.weaknesses`
    *   **Trader Profile Text:** `report.analysis_text.trader_profile`

### **B. Metrics Dashboard (`#metrics`)**
**Component:** `components/comprehensive_metrice_dhashboard.html`
*   **Performance:**
    *   Sortino Ratio: `dm.sortino_ratio`
    *   Return on Capital: `dm.return_on_capital`
    *   Efficiency Ratio: `dm.efficiency_ratio`
*   **Risk & Volatility:**
    *   VaR (95%): `dm.value_at_risk_95`
    *   PnL Volatility: `dm.pnl_volatility`
    *   Skewness: `dm.pnl_skewness`
    *   Kurtosis: `dm.pnl_kurtosis`
*   **Win/Loss & Risk/Reward:**
    *   Avg Win/Loss: `dm.avg_win`, `dm.avg_loss`
    *   Largest Win/Loss: `dm.largest_win`, `dm.largest_loss`
    *   Reward-to-Risk: `dm.reward_to_risk_balance`
    *   Avg Gainer/Loser %: `dm.avg_gainer_pct`, `dm.avg_loser_pct`
*   **Activity:**
    *   Avg Trades/Day: `dm.avg_trades_per_day`
    *   Avg Holding (min): `dm.avg_holding_period`
    *   Streaks: `dm.consecutive_wins`, `dm.consecutive_losses`
*   **Quality Scores:**
    *   Avg F-Score: `dm.avg_f_score` (plus volatility)
    *   Avg T-Score: `dm.avg_t_score` (plus volatility)
    *   Total Score: `dm.avg_total_score`
*   **Score Signals:**
    *   High Score Win Rate: `dm.high_score_win_rate`
    *   Low Score Loss Rate: `dm.low_score_loss_rate`
    *   Events: 52w High/Low Hit Rates

### **C. Charts (`#charts`)**
**Components:**
1.  **`components/charts.html`**
    *   **P&L Timeline:** `report.web_data.charts.pnl_timeline` (Chart.js Line)
    *   **Nifty Timeline:** `report.web_data.charts.nifty_pnl_timeline` (Chart.js Line)
    *   **Instrument Dist:** `report.web_data.charts.instrument_distribution` (Chart.js Doughnut)
    *   **Segment Dist:** `report.web_data.charts.segment_distribution` (Chart.js Doughnut + Filter)
2.  **`components/time_based_performance_chart.html`** (Inferred)

### **D. Positions (`#positions`)**
**Components:**
1.  **`components/positions_table.html`**
    *   Likely iterates `report.positions_data` or similar list.
2.  **`components/gainers_n_losers.html`**
    *   Lists top/bottom performers.

### **E. Patterns & Buckets (`#buckets`)**
**Components:**
1.  **`components/patterns.html`**
    *   Analysis of specific chart patterns detected.
2.  **`components/buckets.html`**
    *   Grouped performance stats.
3.  **`components/trade_quality_distribution.html`**
4.  **`components/holdingh_period_analysis.html`**

### **F. Performance Analysis (`#performance`)**
*   **Duplicate Dashboard:** Re-includes `comprehensive_metrice_dhashboard.html`.
*   **Contextual Dashboard (`components/contextual_performance_dashboard.html`)**
    *   **Event Trading:** Count, Win Rate, Net P&L, AI Verdict (`report.analysis_text.context_performance.event...`).
    *   **News Trading:** Similar metrics.
    *   **High Volume:** Similar metrics.
    *   **Trend Alignment:** Trend Score (`report.web_data.kpis.trend_alignment_score`), Aligned Win Rate.
*   **Extra Charts:** Re-renders P&L Timeline and Instrument Chart on canvases.

### **G. Deep Dive Analysis (`#analysis`)**
*   **Tabs:** Positions Log (JS powered), Pattern Buckets, Risk Management.
*   **Content:** Re-uses bucket and risk components inside tabs.

### **H. AI Insight & Recommendations (`#insights`)**
*   **Action Plan:** `report.analysis_text.recommendations` (HTML Safe).
*   **Final Verdict:** `report.risk_score` (/100).
*   **Persona Breakdowns:** Consistency, Discipline, Emotional Control (`report.persona_scores...`).

---

## 3. 📂 Data Model (JSON Structure)

The template expects a `report` object with the following structure:

```json
{
  "executive_summary": {
    "total_trades": "int",
    "net_pnl": "float",
    "win_rate": "float",
    "sharpe_ratio": "float",
    "profit_factor": "float",
    "max_drawdown_pct": "float",
    "risk_level": "string",
    "open_positions_count": "int"
  },
  "hard_flags": { "flag_name": "True/False" },
  "detailed_metrics": {
    "total_investment_value_open": "float",
    "total_realized_pnl": "float",
    "total_unrealized_pnl": "float",
    "total_pnl_combined": "float",
    "close_pos_booked_sl_pct": "float",
    "sortino_ratio": "float",
    "return_on_capital": "float",
    "value_at_risk_95": "float",
    "pnl_volatility": "float",
    "pnl_skewness": "float",
    "avg_win": "float",
    "avg_loss": "float",
    "reward_to_risk_balance": "float",
    "avg_gainer_pct": "float",
    "avg_trades_per_day": "float",
    "avg_f_score": "float",
    "high_score_win_rate": "float",
    "...": "Other metrics listed in Dashboard section"
  },
  "summary_data": {
    "strengths": "string (bullet points)",
    "weaknesses": "string (bullet points)"
  },
  "analysis_text": {
    "trader_profile": "HTML string",
    "risk_assessment": "HTML string",
    "recommendations": "HTML string",
    "context_performance": {
      "event": { "verdict": "striing", "reasoning": "string" },
      "news": { ... },
      "volume": { ... },
      "trend": { ... }
    }
  },
  "web_data": {
    "charts": {
      "pnl_timeline": { "dates": [], "values": [], "benchmark_values": [] },
      "nifty_pnl_timeline": { "dates": [], "values": [] },
      "instrument_distribution": [ { "asset_kind": "string", "value": "float" } ],
      "segment_distribution": [ { "asset_name": "string", "value": "float" } ]
    },
    "persona_scores": {
      "discipline_score": "float (0-1)",
      "emotional_control": "float (0-1)",
      "consistency": "float (0-1)",
      "...": "other traits"
    },
    "kpis": {
      "event_trading_count": "int",
      "event_trading_win_rate": "float",
      "trend_alignment_score": "float",
      "...": "Other contextual KPIs"
    }
  },
  "risk_score": "int",
  "risk_severity": "string",
  "positions_data": [ "List of position objects" ]
}
```

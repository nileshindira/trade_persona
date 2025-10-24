"""
Professional Trading Persona Report Generator
Keeps full data fidelity, adds expand/collapse, scrollability, modern Bootstrap layout.
"""

import json
from typing import Dict, List
from datetime import datetime
from pathlib import Path
import markdown
from jinja2 import Template
import numpy as np


class ReportGenerator:
    """Generate polished HTML/JSON trading persona reports with visualization"""

    def __init__(self, config: Dict):
        self.config = config

    # ---------------------------------------------------------------------
    # MAIN REPORT BUILD
    # ---------------------------------------------------------------------
    def generate_report(self, metrics: Dict, patterns: Dict, analysis: Dict, trader_name: str = "Trader") -> Dict:
        """Compose structured report dictionary"""
        analysis_html = {
            key: markdown.markdown(analysis.get(key, ""), extensions=["tables", "fenced_code", "nl2br"])
            for key in ["trader_profile", "risk_assessment", "behavioral_insights", "performance_summary"]
        }

        return {
            "metadata": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "trader_name": trader_name,
                "analysis_period": metrics.get("date_range", "N/A"),
            },
            "executive_summary": self._create_executive_summary(metrics, analysis),
            "detailed_metrics": metrics,
            "detected_patterns": patterns,
            "ai_analysis": analysis_html,
            "recommendations": self._format_recommendations(analysis.get("recommendations", "")),
            "risk_score": self._calculate_risk_score(metrics, patterns),
        }

    # ---------------------------------------------------------------------
    # SUB-HELPERS
    # ---------------------------------------------------------------------
    def _create_executive_summary(self, metrics: Dict, analysis: Dict) -> Dict:
        return {
            "total_trades": metrics.get("total_trades", 0),
            "net_pnl": metrics.get("total_pnl_combined", metrics.get("total_pnl", 0)),
            "win_rate": metrics.get("win_rate", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "risk_level": self._get_risk_level(metrics),
            "day_mtm": metrics.get("day_mtm", 0),
            "open_positions_count": metrics.get("open_positions_count", 0),
        }

    def _get_risk_level(self, metrics: Dict) -> str:
        sharpe = metrics.get("sharpe_ratio", 0)
        drawdown = abs(metrics.get("max_drawdown_pct", 0))
        if sharpe < 0 or drawdown > 30:
            return "VERY HIGH"
        elif sharpe < 0.5 or drawdown > 20:
            return "HIGH"
        elif sharpe < 1.0 or drawdown > 10:
            return "MEDIUM"
        return "LOW"

    def _format_recommendations(self, text: str) -> List[str]:
        lines = [l.strip("-•* ").capitalize() for l in text.splitlines() if l.strip()]
        return lines[:15]

    def _calculate_risk_score(self, metrics: Dict, patterns: Dict) -> int:
        score = 50
        if metrics.get("sharpe_ratio", 0) < 0:
            score += 20
        if abs(metrics.get("max_drawdown_pct", 0)) > 20:
            score += 15
        if metrics.get("win_rate", 50) < 45:
            score += 10
        if patterns.get("overtrading", {}).get("detected"):
            score += 10
        if patterns.get("revenge_trading", {}).get("detected"):
            score += 10
        return min(100, score)

    # ---------------------------------------------------------------------
    # EXPORT HTML (with full visuals)
    # ---------------------------------------------------------------------
    def export_html(self, report: dict, filepath: str):
        """Render an interactive, psychology‑aware trading dashboard (HTML)

        Highlights
        - Bootstrap 5 layout with sticky KPI bar
        - Light/Dark theme toggle + print stylesheet
        - Chart.js visuals: P&L timeline, persona radar, winners/losers bars,
          hourly/weekday activity, drawdown curve, allocation donut (if available)
        - DataTable with search/sort for open positions
        - Recommendation carousel (auto‑scroll, dock hover)
        - Behavioral nudges with emojis & badges
        - JSON export & section deep‑links (hash routing)
        - Graceful guards when data keys are missing
        """
        import numpy as np
        from pathlib import Path
        from jinja2 import Template
        import json

        def safe(o):
            if isinstance(o, dict):
                return {k: safe(v) for k, v in o.items()}
            if isinstance(o, list):
                return [safe(v) for v in o]
            if isinstance(o, (np.generic,)):
                return o.item()
            if isinstance(o, (set, tuple)):
                return list(o)
            return o

        report = safe(report or {})

        html_template = r"""
        <!DOCTYPE html>
        <html lang="en" data-theme="light">
        <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <title>Trading Persona Report — {{ (report.metadata.trader_name or 'Trader') }}</title>

        <!-- CDN Imports -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"/>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>
        <link href="https://cdn.jsdelivr.net/npm/simple-datatables@9.0.4/dist/style.css" rel="stylesheet"/>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>
        {% raw %}
        <style>
        :root{
          /* Semantic color system tuned for clean light/dark parity */
          --primary:#0d6efd; --primary-weak:#93c5fd;
          --accent:#10b981; --accent-weak:#a7f3d0;
          --danger:#ef4444; --warn:#f59e0b; --ok:#22c55e;
          --bg-light:#f8fafc; --card-light:#ffffff; --text-light:#0f172a; --muted-light:#64748b;
          --bg-dark:#0b1120; --card-dark:#111827; --text-dark:#e5e7eb; --muted-dark:#94a3b8;
          --shadow:0 6px 24px rgba(0,0,0,.08);
        }
        [data-theme="light"]{--bg:var(--bg-light);--card:var(--card-light);--text:var(--text-light);--muted:var(--muted-light);}
        [data-theme="dark"]{--bg:var(--bg-dark);--card:var(--card-dark);--text:var(--text-dark);--muted:var(--muted-dark);}
        html,body{height:100%}
        body{background:var(--bg);color:var(--text);font-family:Inter,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
        .card{border:0;border-radius:1rem;background:var(--card);box-shadow:var(--shadow)}
        .section-title{font-weight:800;color:var(--primary);margin-top:2rem}
        .table thead th{white-space:nowrap}
        .nav-tabs .nav-link{border:0}
        .nav-tabs .nav-link.active{background:var(--primary);color:#fff;border-radius:.75rem}
        .nav-pills .nav-link{border-radius:999px}
        .kpi{min-width:160px}
        .kpi h5{font-weight:800}
        .badge-pill{border-radius:999px;padding:.4rem .75rem}
        .scroll-y{max-height:280px;overflow:auto}
        .small-muted{color:var(--muted)}
        .hr-soft{border-top:1px dashed rgba(127,127,127,.35)}
        a{color:var(--primary)}
        [data-theme="dark"] a{color:var(--primary-weak)}
        .reco-card{border-left:6px solid var(--primary);background:linear-gradient(135deg,rgba(13,110,253,.08),rgba(16,185,129,.08))}
        [data-theme="dark"] .reco-card{background:linear-gradient(135deg,rgba(96,165,250,.12),rgba(52,211,153,.10))}
        @media print{#topbar,.no-print{display:none!important}.card{box-shadow:none}}
        /* Tables: align right for numbers */
        td.text-end,th.text-end{text-align:right}
        </style>
        {% endraw %}
        </head>

        <body>
        <!-- Navbar -->
        <nav id="topbar" class="navbar navbar-expand-lg sticky-top" style="background:var(--card);box-shadow:var(--shadow)">
          <div class="container-xl py-2">
            <a class="navbar-brand fw-bold" href="#"><i class="bi bi-graph-up-arrow text-primary"></i> Trading Persona Report</a>
            <div class="d-flex gap-2">
              <button class="btn btn-outline-secondary btn-sm" id="themeToggle" title="Toggle theme"><i class="bi bi-moon"></i></button>
              <button class="btn btn-outline-primary btn-sm" onclick="window.print()" title="Print"><i class="bi bi-printer"></i></button>
              <button class="btn btn-outline-success btn-sm" id="jsonBtn" title="Download JSON"><i class="bi bi-download"></i></button>
            </div>
          </div>
        </nav>

        <div class="container-xl my-4">

          <!-- ===== TRADER OVERVIEW (INTRO WITH TAGS) ===== -->
          <section id="overview">
            <div class="card p-4 mb-4">
              <div class="d-flex flex-wrap justify-content-between align-items-center gap-3">
                <div>
                  <div class="d-flex align-items-center gap-3">
                    <div class="rounded-circle d-flex align-items-center justify-content-center" style="width:52px;height:52px;background:linear-gradient(135deg,var(--primary),var(--accent));color:#fff;font-weight:800;">
                      {{ (report.metadata.trader_name or 'T')[:1]|upper }}
                    </div>
                    <div>
                      <h3 class="fw-bold mb-1">{{ report.metadata.trader_name or 'Trader' }}</h3>
                      <div class="small-muted">
                        Period: {{ report.metadata.analysis_period or '—' }} • Generated: {{ report.metadata.generated_at or '—' }}
                      </div>
                      <div class="mt-2 d-flex flex-wrap gap-2">
                        <span class="badge bg-primary">{{ report.detailed_metrics.persona_type or 'Persona: N/A' }}</span>
                        <span class="badge bg-info text-dark">Risk: {{ report.executive_summary.risk_level or 'N/A' }}</span>
                        <span class="badge bg-success">Sharpe: {{ '%.2f'|format((report.executive_summary.sharpe_ratio or 0)) }}</span>
                        {% if report.detailed_metrics.avg_holding_period %}
                          <span class="badge bg-secondary">Avg Hold: {{ '%.1f'|format(report.detailed_metrics.avg_holding_period) }} min</span>
                        {% endif %}
                        {% if report.detailed_metrics.open_positions_count is defined %}
                          <span class="badge bg-secondary">Open Symbols: {{ report.detailed_metrics.open_positions_count }}</span>
                        {% endif %}
                      </div>
                    </div>
                  </div>
                </div>
                <div class="text-end">
                  {% set pnl_intro = report.executive_summary.net_pnl or 0 %}
                  <div class="fs-1 fw-bold {{ 'text-success' if pnl_intro>=0 else 'text-danger' }}">₹{{ '%.2f'|format(pnl_intro) }}</div>
                  <div class="small-muted">Total P&L</div>
                </div>
              </div>

              <hr class="hr-soft">
              <div class="row text-center g-3">
                <div class="col-6 col-md-2"><div class="kpi"><div class="small-muted">Trades</div><h5>{{ report.executive_summary.total_trades or 0 }}</h5></div></div>
                <div class="col-6 col-md-2"><div class="kpi"><div class="small-muted">Win Rate</div><h5>{{ '%.1f'|format(report.executive_summary.win_rate or 0) }}%</h5></div></div>
                <div class="col-6 col-md-2"><div class="kpi"><div class="small-muted">Day MTM</div><h5>{{ '%.2f'|format(report.executive_summary.day_mtm or 0) }}</h5></div></div>
                <div class="col-6 col-md-2"><div class="kpi"><div class="small-muted">Open Pos</div><h5>{{ report.executive_summary.open_positions_count or 0 }}</h5></div></div>
                <div class="col-6 col-md-2"><div class="kpi"><div class="small-muted">Sharpe</div><h5>{{ '%.2f'|format(report.executive_summary.sharpe_ratio or 0) }}</h5></div></div>
                <div class="col-6 col-md-2"><div class="kpi"><div class="small-muted">Risk</div><h5>{{ report.executive_summary.risk_level or '—' }}</h5></div></div>
              </div>
            </div>
          </section>

          <!-- ===== SNAPSHOT SUMMARY (OPEN/CLOSED/PNL/MTM) ===== -->
          <section id="snapshot">
            <h4 class="section-title"><i class="bi bi-speedometer2"></i> Snapshot Summary</h4>
            {% set dm = report.detailed_metrics %}
            <div class="row g-3 mb-3">
              <div class="col-md-2"><div class="card p-3 text-center"><div class="small-muted">Investment</div><h5>₹{{ '%.2f'|format(dm.total_investment_value_open or 0) }}</h5></div></div>
              <div class="col-md-2"><div class="card p-3 text-center"><div class="small-muted">Realized</div><h5 class="{{ 'text-success' if (dm.total_realized_pnl or 0) >=0 else 'text-danger' }}">₹{{ '%.2f'|format(dm.total_realized_pnl or 0) }}</h5></div></div>
              <div class="col-md-2"><div class="card p-3 text-center"><div class="small-muted">Unrealized</div><h5 class="{{ 'text-success' if (dm.total_unrealized_pnl or 0) >=0 else 'text-danger' }}">₹{{ '%.2f'|format(dm.total_unrealized_pnl or 0) }}</h5></div></div>
              <div class="col-md-2"><div class="card p-3 text-center"><div class="small-muted">Total P&L</div><h5 class="{{ 'text-success' if (dm.total_pnl_combined or 0) >=0 else 'text-danger' }}">₹{{ '%.2f'|format(dm.total_pnl_combined or 0) }}</h5></div></div>
              <div class="col-md-2"><div class="card p-3 text-center"><div class="small-muted">ROI %</div><h5>{{ '%.2f'|format(((dm.total_pnl_combined or 0)/((dm.total_investment_value_open or 0) if (dm.total_investment_value_open or 0)!=0 else 1)*100)) }}%</h5></div></div>
              <div class="col-md-2"><div class="card p-3 text-center"><div class="small-muted">Booked SL %</div><h5>{{ '%.1f'|format(dm.close_pos_booked_sl_pct or 0) }}%</h5></div></div>
            </div>
            {% if dm.pnl_timeline and dm.pnl_timeline.dates %}
              <div class="card p-3 mb-4">
                <small class="small-muted">Equity Curve</small>
                <canvas id="pnlTimeline" height="160"></canvas>
              </div>
            {% endif %}
          </section>

          <!-- ===== POSITIONS: OPEN / CLOSED TABS ===== -->
          <section id="positions" class="mt-4">
            <h4 class="section-title"><i class="bi bi-box-seam"></i> Positions</h4>

            <!-- Tab Snapshot above tables -->
            <div class="card p-3 mb-3">
              <div class="row g-3">
                <div class="col-md-3"><div class="small-muted">Open Positions (symbols)</div><div class="fw-bold">{{ dm.open_positions_count or 0 }}</div></div>
                <div class="col-md-3"><div class="small-muted">Avg Realized / Stock</div><div class="fw-bold">₹{{ '%.2f'|format(dm.avg_realized_pl_per_stock or 0) }}</div></div>
                <div class="col-md-3"><div class="small-muted">Avg Unrealized / Open Stock</div><div class="fw-bold">₹{{ '%.2f'|format(dm.avg_unrealized_pl_per_open_stock or 0) }}</div></div>
                <div class="col-md-3"><div class="small-muted">Day MTM (Realized)</div><div class="fw-bold">₹{{ '%.2f'|format(dm.day_mtm or 0) }}</div></div>
              </div>
            </div>

            <ul class="nav nav-tabs" id="posTabs" role="tablist">
              <li class="nav-item"><a class="nav-link active" id="open-tab" data-bs-toggle="tab" href="#openTab" role="tab">Open Positions</a></li>
              <li class="nav-item"><a class="nav-link" id="closed-tab" data-bs-toggle="tab" href="#closedTab" role="tab">Closed Positions</a></li>
            </ul>

            <div class="tab-content pt-3">
              <!-- OPEN -->
              <div class="tab-pane fade show active" id="openTab" role="tabpanel" aria-labelledby="open-tab">
                {% if dm.open_positions %}
                  <div class="table-responsive">
                    <table class="table table-sm table-striped align-middle" id="openTable">
                      <thead class="table-light">
                        <tr>
                          <th>Symbol</th>
                          <th class="text-end">Buy Rate</th>
                          <th class="text-end">Last Price</th>
                          <th class="text-end">Qty</th>
                          <th class="text-end">Value</th>
                          <th class="text-end">Unrealized</th>
                          <th class="text-end">MTM %</th>
                          <th class="text-end">Holding Days</th>
                        </tr>
                      </thead>
                      <tbody>
                        {% for p in dm.open_positions %}
                        <tr>
                          <td><a href="#buckets" class="text-decoration-none">{{ p.symbol }}</a></td>
                          <td class="text-end">₹{{ '%.2f'|format(p.avg_cost or 0) }}</td>
                          <td class="text-end">₹{{ '%.2f'|format(p.last_price or 0) }}</td>
                          <td class="text-end">{{ p.net_qty or 0 }}</td>
                          <td class="text-end">₹{{ '%.2f'|format(p.invested_value or 0) }}</td>
                          <td class="text-end {{ 'text-success' if (p.unrealized or 0) >=0 else 'text-danger' }}">₹{{ '%.2f'|format(p.unrealized or 0) }}</td>
                          <td class="text-end {{ 'text-success' if (p.pct_change or 0) >=0 else 'text-danger' }}">{{ '%.2f'|format(p.pct_change or 0) }}%</td>
                          <td class="text-end">{{ p.holding_days or '-' }}</td>
                        </tr>
                        {% endfor %}
                      </tbody>
                    </table>
                  </div>
                {% else %}
                  <div class="alert alert-secondary">No open positions available.</div>
                {% endif %}
              </div>

              <!-- CLOSED -->
              <div class="tab-pane fade" id="closedTab" role="tabpanel" aria-labelledby="closed-tab">
                {% if dm.closed_positions %}
                  <div class="table-responsive">
                    <table class="table table-sm table-striped align-middle" id="closedTable">
                      <thead class="table-light">
                        <tr>
                          <th>Symbol</th>
                          <th class="text-end">Buy Rate</th>
                          <th class="text-end">Sell Rate</th>
                          <th class="text-end">Qty</th>
                          <th class="text-end">Value</th>
                          <th class="text-end">Realized P&L</th>
                          <th class="text-end">Return %</th>
                          <th class="text-end">Holding Days</th>
                        </tr>
                      </thead>
                      <tbody>
                        {% for p in dm.closed_positions %}
                        <tr>
                          <td>{{ p.symbol }}</td>
                          <td class="text-end">₹{{ '%.2f'|format(p.buy_rate or 0) }}</td>
                          <td class="text-end">₹{{ '%.2f'|format(p.sell_rate or 0) }}</td>
                          <td class="text-end">{{ p.qty or 0 }}</td>
                          <td class="text-end">₹{{ '%.2f'|format(p.value or 0) }}</td>
                          <td class="text-end {{ 'text-success' if (p.pnl or 0) >=0 else 'text-danger' }}">₹{{ '%.2f'|format(p.pnl or 0) }}</td>
                          <td class="text-end {{ 'text-success' if (p.return_pct or 0) >=0 else 'text-danger' }}">{{ '%.2f'|format(p.return_pct or 0) }}%</td>
                          <td class="text-end">{{ p.holding_days or '-' }}</td>
                        </tr>
                        {% endfor %}
                      </tbody>
                    </table>
                  </div>
                {% else %}
                  <div class="alert alert-secondary">Closed positions summary not available in this export. (Optional: populate <code>closed_positions</code> in metrics.)</div>
                {% endif %}
              </div>
            </div>
          </section>

          <!-- ===== TRADING PERSONA (BIGGER, CLEANER) ===== -->
          <section id="persona" class="mt-4">
            <h4 class="section-title"><i class="bi bi-person-vcard"></i> Trading Persona</h4>
            {% if dm.persona_traits %}
            <div class="card p-4 mb-4">
              <div class="row g-4">
                <div class="col-xl-6">
                  <canvas id="personaRadar" height="260" aria-label="Persona Traits Radar"></canvas>
                </div>
                <div class="col-xl-6">
                  <div class="mb-3">
                    <span class="badge bg-primary me-1">{{ dm.persona_type or 'Persona' }}</span>
                    {% if dm.persona_tags %}{% for tag in dm.persona_tags %}<span class="badge bg-secondary me-1">{{ tag }}</span>{% endfor %}{% endif %}
                  </div>
                  <div class="small-muted">Summary</div>
                  <div class="card p-3 mb-3" style="background:linear-gradient(135deg,rgba(13,110,253,.08),rgba(16,185,129,.08))">
                    <div class="small">{{ (dm.trait_summary or '')|safe }}</div>
                  </div>
                  <div class="small-muted mb-1">Trait Bars</div>
                  <canvas id="personaBars" height="140"></canvas>
                </div>
              </div>
            </div>
            {% else %}
              <div class="alert alert-secondary">Persona traits not available in this export.</div>
            {% endif %}
          </section>

          <!-- ===== REALIZED & UNREALIZED OVERVIEW (GAUGE + EXPAND) ===== -->
          <section id="ru" class="mt-4">
            <h4 class="section-title"><i class="bi bi-pie-chart"></i> Realized & Unrealized Overview</h4>
            <div class="card p-4 text-center">
              <div class="row justify-content-center">
                <div class="col-md-4">
                  <canvas id="ruGauge" height="220" aria-label="Realized vs Unrealized"></canvas>
                  <div class="small-muted mt-2">Share of P&L that is realized vs unrealized</div>
                </div>
              </div>

              <button class="btn btn-sm btn-outline-primary mt-3 no-print" data-bs-toggle="collapse" data-bs-target="#ruDetails">
                Show All Trades (Realized & Unrealized)
              </button>
              <div class="collapse mt-3 text-start" id="ruDetails">
                <div class="row g-3">
                  <div class="col-lg-6">
                    <div class="card p-3 h-100">
                      <h6 class="fw-bold mb-2"><i class="bi bi-cash-coin text-success"></i> Realized Trades</h6>
                      {% if dm.realized_trades %}
                        <div class="table-responsive"><table class="table table-sm table-striped">
                          <thead><tr>
                            <th>Symbol</th><th class="text-end">Buy Rate</th><th class="text-end">Sell Rate</th>
                            <th class="text-end">Qty</th><th class="text-end">P&L</th><th class="text-end">Holding Days</th><th class="text-end">Return %</th>
                          </tr></thead>
                          <tbody>
                            {% for t in dm.realized_trades %}
                            <tr>
                              <td>{{ t.symbol }}</td><td class="text-end">₹{{ '%.2f'|format(t.buy_rate or 0) }}</td>
                              <td class="text-end">₹{{ '%.2f'|format(t.sell_rate or 0) }}</td><td class="text-end">{{ t.qty or 0 }}</td>
                              <td class="text-end {{ 'text-success' if (t.pnl or 0)>=0 else 'text-danger' }}">₹{{ '%.2f'|format(t.pnl or 0) }}</td>
                              <td class="text-end">{{ t.holding_days or '-' }}</td>
                              <td class="text-end {{ 'text-success' if (t.return_pct or 0)>=0 else 'text-danger' }}">{{ '%.2f'|format(t.return_pct or 0) }}%</td>
                            </tr>
                            {% endfor %}
                          </tbody>
                        </table></div>
                      {% else %}
                        <div class="alert alert-secondary">Detailed realized trades not attached. (Optional: populate <code>realized_trades</code> list.)</div>
                      {% endif %}
                    </div>
                  </div>

                  <div class="col-lg-6">
                    <div class="card p-3 h-100">
                      <h6 class="fw-bold mb-2"><i class="bi bi-bar-chart text-info"></i> Unrealized (Open) Trades</h6>
                      {% if dm.unrealized_trades %}
                        <div class="table-responsive"><table class="table table-sm table-striped">
                          <thead><tr>
                            <th>Symbol</th><th class="text-end">Buy Rate</th><th class="text-end">LTP</th>
                            <th class="text-end">Qty</th><th class="text-end">Unrealized</th><th class="text-end">MTM %</th><th class="text-end">Holding Days</th>
                          </tr></thead>
                          <tbody>
                            {% for t in dm.unrealized_trades %}
                            <tr>
                              <td>{{ t.symbol }}</td><td class="text-end">₹{{ '%.2f'|format(t.buy_rate or 0) }}</td>
                              <td class="text-end">₹{{ '%.2f'|format(t.ltp or 0) }}</td><td class="text-end">{{ t.qty or 0 }}</td>
                              <td class="text-end {{ 'text-success' if (t.unrealized or 0)>=0 else 'text-danger' }}">₹{{ '%.2f'|format(t.unrealized or 0) }}</td>
                              <td class="text-end {{ 'text-success' if (t.mtm_pct or 0)>=0 else 'text-danger' }}">{{ '%.2f'|format(t.mtm_pct or 0) }}%</td>
                              <td class="text-end">{{ t.holding_days or '-' }}</td>
                            </tr>
                            {% endfor %}
                          </tbody>
                        </table></div>
                      {% else %}
                        <div class="alert alert-secondary">Detailed unrealized (open) trades not attached. (Optional: populate <code>unrealized_trades</code> list.)</div>
                      {% endif %}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <!-- ===== GAINERS & LOSERS (DETAILED) ===== -->
          <section id="gainers" class="mt-4">
            <h4 class="section-title"><i class="bi bi-arrow-left-right"></i> Gainers & Losers</h4>
            <div class="accordion" id="gainLossAcc">
              {% for key,label,icon,clr in [('gainer','Top Gainers','bi-arrow-up-circle','text-success'),('loser','Top Losers','bi-arrow-down-circle','text-danger')] %}
              {% set blk = dm.get(key, {}) %}
              <div class="accordion-item mb-2">
                <h2 class="accordion-header">
                  <button class="accordion-button {{ 'collapsed' if not loop.first else '' }}" data-bs-toggle="collapse" data-bs-target="#{{key}}Collapse">
                    <i class="bi {{icon}} me-2 {{clr}}"></i> {{ label }} ({{ blk.count or 0 }})
                  </button>
                </h2>
                <div id="{{key}}Collapse" class="accordion-collapse collapse {{ 'show' if loop.first else '' }}">
                  <div class="accordion-body">
                    {% if blk.list %}
                      <div class="table-responsive">
                        <table class="table table-sm table-striped align-middle">
                          <thead>
                            <tr>
                              <th>Symbol</th><th class="text-end">Buy Rate</th><th class="text-end">Sell/LTP</th>
                              <th class="text-end">Qty</th><th class="text-end">Value</th><th class="text-end">P&L</th>
                              <th class="text-end">Holding Days</th><th class="text-end">Return %</th>
                            </tr>
                          </thead>
                          <tbody>
                            {% for s in blk.list %}
                              {% set det = (dm.symbol_details[s] if (dm.symbol_details and (s in dm.symbol_details)) else None) %}
                              <tr>
                                <td>{{ s }}</td>
                                <td class="text-end">₹{{ '%.2f'|format((det.buy_rate if det and det.buy_rate is not none else 0)) }}</td>
                                <td class="text-end">₹{{ '%.2f'|format((det.sell_rate if det and det.sell_rate is not none else (det.ltp if det and det.ltp is not none else 0))) }}</td>
                                <td class="text-end">{{ det.qty if det and det.qty is not none else '-' }}</td>
                                <td class="text-end">₹{{ '%.2f'|format((det.value if det and det.value is not none else 0)) }}</td>
                                {% set pnlv = (det.pnl if det and det.pnl is not none else (det.unrealized if det and det.unrealized is not none else 0)) %}
                                <td class="text-end {{ 'text-success' if (pnlv or 0)>=0 else 'text-danger' }}">₹{{ '%.2f'|format(pnlv or 0) }}</td>
                                <td class="text-end">{{ det.holding_days if det and det.holding_days is not none else '-' }}</td>
                                <td class="text-end {{ 'text-success' if (det.return_pct if det else 0)>=0 else 'text-danger' }}">{{ '%.2f'|format((det.return_pct if det and det.return_pct is not none else 0)) }}%</td>
                              </tr>
                            {% endfor %}
                          </tbody>
                        </table>
                      </div>
                    {% else %}
                      <div class="alert alert-secondary">No {{ label.lower() }} data present.</div>
                    {% endif %}
                  </div>
                </div>
              </div>
              {% endfor %}
            </div>
          </section>

          <!-- ===== BUCKETS BY % CHANGE (CLICK TO SEE TRADE JOURNEY) ===== -->
          <section id="buckets" class="mt-4">
            <h4 class="section-title"><i class="bi bi-percent"></i> Buckets by % Change (Open Positions)</h4>
            {% if dm.buckets %}
              <div class="row g-3">
                {% for k,v in dm.buckets.items() %}
                <div class="col-sm-6 col-lg-3">
                  <div class="card p-3 h-100">
                    <div class="d-flex justify-content-between align-items-center">
                      <div class="fw-bold">{{ k }}</div>
                      <span class="badge bg-secondary">#{{ v.count or 0 }}</span>
                    </div>
                    <div class="small-muted mt-1">{{ '%.1f'|format(v.mtm_share_pct or 0) }}% MTM • ₹{{ '%.0f'|format(v.total_value or 0) }}</div>

                    <button class="btn btn-sm btn-outline-primary mt-2 no-print" data-bs-toggle="collapse" data-bs-target="#bucket{{loop.index}}">
                      View Symbols & Trade Journey
                    </button>

                    <div class="collapse mt-2" id="bucket{{loop.index}}">
                      {% if v.list and v.list|length>0 %}
                        <ul class="list-group list-group-flush small">
                          {% set outer_index = loop.index %}
                            {% for s in v.list %}
                              <li class="list-group-item bg-transparent d-flex justify-content-between align-items-center">
                                <span><i class="bi bi-dot"></i> {{ s }}</span>
                                <button class="btn btn-xs btn-link" data-bs-toggle="collapse" data-bs-target="#journey{{outer_index}}_{{loop.index}}">details</button>
                              </li>
                              <div class="collapse" id="journey{{outer_index}}_{{loop.index}}">
                            
                              {% set J = (dm.symbol_journeys[s] if (dm.symbol_journeys and (s in dm.symbol_journeys)) else None) %}
                              {% if J %}
                                <div class="table-responsive mt-2">
                                  <table class="table table-sm table-striped">
                                    <thead><tr>
                                      <th>Leg</th><th class="text-end">Buy Rate</th><th class="text-end">Sell Rate</th>
                                      <th class="text-end">Qty</th><th class="text-end">Value</th><th class="text-end">P&L</th>
                                      <th class="text-end">Holding Days</th><th class="text-end">Return %</th>
                                    </tr></thead>
                                    <tbody>
                                      {% for leg in J %}
                                        <tr>
                                          <td>{{ loop.index }}</td>
                                          <td class="text-end">₹{{ '%.2f'|format(leg.buy_rate or 0) }}</td>
                                          <td class="text-end">₹{{ '%.2f'|format(leg.sell_rate or 0) }}</td>
                                          <td class="text-end">{{ leg.qty or 0 }}</td>
                                          <td class="text-end">₹{{ '%.2f'|format(leg.value or 0) }}</td>
                                          <td class="text-end {{ 'text-success' if (leg.pnl or 0)>=0 else 'text-danger' }}">₹{{ '%.2f'|format(leg.pnl or 0) }}</td>
                                          <td class="text-end">{{ leg.holding_days or '-' }}</td>
                                          <td class="text-end {{ 'text-success' if (leg.return_pct or 0)>=0 else 'text-danger' }}">{{ '%.2f'|format(leg.return_pct or 0) }}%</td>
                                        </tr>
                                      {% endfor %}
                                    </tbody>
                                  </table>
                                </div>
                              {% else %}
                                <div class="alert alert-secondary mt-2">Trade journey not attached for <strong>{{ s }}</strong>. (Optional: populate <code>symbol_journeys[s]</code> as a list.)</div>
                              {% endif %}
                            </div>
                          {% endfor %}
                        </ul>
                      {% else %}
                        <div class="alert alert-secondary mt-2">No symbols in this bucket.</div>
                      {% endif %}
                    </div>
                  </div>
                </div>
                {% endfor %}
              </div>
            {% else %}
              <div class="alert alert-secondary">Bucket analysis not available.</div>
            {% endif %}
          </section>

          <!-- ===== BEHAVIORAL ANALYTICS (existing charts) ===== -->
          <section id="analytics" class="mt-4">
            <h4 class="section-title"><i class="bi bi-activity"></i> Behavioral Analytics</h4>
            <div class="row g-3">
              <div class="col-md-4">
                <div class="card p-3 h-100">
                  <h6 class="fw-bold mb-2">Hour of Day ⏰</h6>
                  <canvas id="hourHist" height="180"></canvas>
                </div>
              </div>
              <div class="col-md-4">
                <div class="card p-3 h-100">
                  <h6 class="fw-bold mb-2">Day of Week 📅</h6>
                  <canvas id="weekdayHist" height="180"></canvas>
                </div>
              </div>
              <div class="col-md-4">
                <div class="card p-3 h-100">
                  <h6 class="fw-bold mb-2">Allocation 🍰</h6>
                  <canvas id="allocationDonut" height="180"></canvas>
                </div>
              </div>
            </div>
          </section>

          <!-- ===== AI INSIGHTS (existing) ===== -->
          <section id="ai" class="mt-4">
            {% if report.ai_analysis %}
            <h4 class="section-title"><i class="bi bi-robot"></i> AI Insights</h4>
            <div class="accordion" id="aiAcc">
              {% for k,v in report.ai_analysis.items() %}
              <div class="accordion-item mb-2">
                <h2 class="accordion-header">
                  <button class="accordion-button {{ '' if loop.first else 'collapsed' }}" data-bs-toggle="collapse" data-bs-target="#ai{{ loop.index }}">
                    {{ k.replace('_',' ').title() }}
                  </button>
                </h2>
                <div id="ai{{ loop.index }}" class="accordion-collapse collapse {{ 'show' if loop.first else '' }}">
                  <div class="accordion-body">{{ v|safe }}</div>
                </div>
              </div>
              {% endfor %}
            </div>
            {% endif %}
          </section>

          <!-- ===== RECOMMENDATIONS (refined dock) ===== -->
          {% if report.recommendations %}
          <section class="mt-4">
            <h4 class="section-title"><i class="bi bi-lightbulb"></i> Recommendations</h4>
            <div class="position-relative overflow-hidden py-3">
              <div id="dockCarousel" class="d-flex flex-nowrap justify-content-start align-items-stretch gap-3">
                {% for rec in report.recommendations %}
                <div class="reco-item flex-shrink-0" style="width:260px">
                  <div class="card reco-card p-3 h-100">
                    <div class="fs-2 mb-2 text-primary"><i class="bi bi-lightbulb-fill"></i></div>
                    <p class="mb-1 fw-semibold">{{ rec }}</p>
                    <div class="small-muted">#{{ loop.index }}</div>
                  </div>
                </div>
                {% endfor %}
              </div>
            </div>
          </section>
          {% endif %}

          <div class="text-center small-muted my-4 small">AI-powered report — not financial advice.<br>© {{ (report.metadata.generated_at or '0000')[:4] }} The ResolveRoom</div>
        </div>

        <!-- Toast nudges (unchanged) -->
        <div id="toastNudges" class="toast-container position-fixed bottom-0 end-0 p-3 no-print"></div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/simple-datatables@9.0.4" defer></script>
        {% raw %}
        <script>
        /* ===== Helper accessors ===== */
        const rep = {{ report|tojson }};
        const dm  = (rep && rep.detailed_metrics) ? rep.detailed_metrics : {};

        /* ===== Theme toggle & JSON download ===== */
        const root=document.documentElement;
        (function initTheme(){root.setAttribute('data-theme',localStorage.getItem('theme')||'light');})();
        document.getElementById('themeToggle')?.addEventListener('click',()=>{
          const next=(root.getAttribute('data-theme')==='dark')?'light':'dark';
          root.setAttribute('data-theme',next); localStorage.setItem('theme',next);
        });
        document.getElementById('jsonBtn')?.addEventListener('click',()=>{
          const blob=new Blob([JSON.stringify(rep,null,2)],{type:'application/json'});
          const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='trading_report.json'; a.click();
        });

        /* ===== DataTables for Open/Closed if present ===== */
        window.addEventListener('DOMContentLoaded',()=>{
          if (window.simpleDatatables){
            const ot=document.getElementById('openTable');  if(ot){ new simpleDatatables.DataTable(ot,{perPage:10,perPageSelect:[10,25,50],fixedHeight:true}); }
            const ct=document.getElementById('closedTable');if(ct){ new simpleDatatables.DataTable(ct,{perPage:10,perPageSelect:[10,25,50],fixedHeight:true}); }
          }
        });

        /* ===== Charts ===== */
        function moneyFmt(v){return '₹'+Number(v).toLocaleString(undefined,{maximumFractionDigits:2});}

        /* Equity Curve */
        (function(){
          if(!dm || !dm.pnl_timeline || !dm.pnl_timeline.dates || !dm.pnl_timeline.values) return;
          const el=document.getElementById('pnlTimeline'); if(!el) return;
          new Chart(el,{type:'line',data:{labels:dm.pnl_timeline.dates,datasets:[{label:'P&L',data:dm.pnl_timeline.values,fill:true,tension:.25}]},options:{plugins:{legend:{display:false}},scales:{y:{ticks:{callback:moneyFmt}}}});
        })();

        /* Persona Radar & Bars */
        (function(){
          const traits = dm && dm.persona_traits ? dm.persona_traits : null;
          if(!traits) return;
          const rEl=document.getElementById('personaRadar');
          if(rEl){ new Chart(rEl,{type:'radar',data:{labels:Object.keys(traits),datasets:[{data:Object.values(traits),label:'Traits',fill:true}]},options:{plugins:{legend:{display:false}},scales:{r:{min:0,max:1,grid:{color:'rgba(127,127,127,.3)'}}}}}); }
          const bEl=document.getElementById('personaBars');
          if(bEl){ new Chart(bEl,{type:'bar',data:{labels:Object.keys(traits).map(t=>t.replaceAll('_',' ').toUpperCase()),datasets:[{data:Object.values(traits)}]},options:{plugins:{legend:{display:false}},scales:{y:{min:0,max:1}}}); }
        })();

        /* Hour & Weekday histograms */
        (function(){
          const hEl=document.getElementById('hourHist'); const wEl=document.getElementById('weekdayHist');
          if(hEl && Array.isArray(dm.hour_hist)){new Chart(hEl,{type:'bar',data:{labels:[...Array(24).keys()].map(h=>h+':00'),datasets:[{data:dm.hour_hist}]},options:{plugins:{legend:{display:false}}}});}
          if(wEl && Array.isArray(dm.weekday_hist)){new Chart(wEl,{type:'bar',data:{labels:['Mon','Tue','Wed','Thu','Fri'],datasets:[{data:dm.weekday_hist}]},options:{plugins:{legend:{display:false}}}});}
        })();

        /* Allocation donut (if present) */
        (function(){
          const el=document.getElementById('allocationDonut'); if(!el || !dm || !dm.allocation) return;
          const labels=Object.keys(dm.allocation), vals=Object.values(dm.allocation);
          new Chart(el,{type:'doughnut',data:{labels,datasets:[{data:vals}]},options:{plugins:{legend:{position:'bottom'}}}});
        })();

        /* Realized vs Unrealized Gauge (doughnut style) */
        (function(){
          const el=document.getElementById('ruGauge'); if(!el) return;
          const realized = Number(dm.total_realized_pnl||0);
          const unreal   = Number(dm.total_unrealized_pnl||0);
          const tot = Math.abs(realized)+Math.abs(unreal);
          const rShare = tot ? Math.abs(realized)/tot*100 : 0;
          new Chart(el,{type:'doughnut',data:{labels:['Realized','Unrealized'],datasets:[{data:[Math.abs(realized),Math.abs(unreal)]} ]},options:{cutout:'70%',plugins:{legend:{position:'bottom'},tooltip:{callbacks:{label:ctx=>ctx.label+': '+moneyFmt(ctx.parsed)}}}});
          // Center text
          Chart.register({id:'centerText',afterDraw(c,_,opts){const {ctx,chartArea:{width,height}}=c;ctx.save();ctx.font='700 16px system-ui';ctx.fillStyle=getComputedStyle(document.body).color;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(rShare.toFixed(1)+'% Realized',c.width/2,c.height/2);ctx.restore();}});
        })();

        /* Recommendations dock animation */
        (function(){
          const container=document.getElementById('dockCarousel'); if(!container) return; let start=performance.now();
          function animate(t){const speed=20; container.scrollLeft=(t/1000*speed)%(container.scrollWidth/2); requestAnimationFrame(animate);} requestAnimationFrame(animate);
          container.addEventListener('mousemove',e=>{for(const it of container.querySelectorAll('.reco-item')){const r=it.getBoundingClientRect();const c=r.left+r.width/2;const dist=Math.abs(e.clientX-c);const scale=Math.max(1,1.6-Math.min(dist/150,1));it.style.transform=`scale(${scale})`;it.style.zIndex=scale>1.1?10:1;}});
          container.addEventListener('mouseleave',()=>{for(const it of container.querySelectorAll('.reco-item')){it.style.transform='scale(1)';it.style.zIndex=1;}});
        })();

        /* Behavioral toast nudges */
        (function(){
          const toasts=[
            {title:'Position Sizing', body:'Keep single-position risk < 2% of equity. 🧯'},
            {title:'Breaks Help', body:'Two 5-min breaks per hour improve focus. ☕'},
            {title:'Review Ritual', body:'Log 3 learnings after market close. 📓'}
          ];
          const wrap=document.getElementById('toastNudges'); if(!wrap) return;
          toasts.forEach((t,i)=>{const el=document.createElement('div'); el.className='toast align-items-center mb-2'; el.role='alert'; el.innerHTML=`<div class="d-flex"><div class="toast-body"><strong>${t.title}:</strong> ${t.body}</div><button type="button" class="btn-close me-2 m-auto" data-bs-dismiss="toast"></button></div>`; wrap.appendChild(el); new bootstrap.Toast(el,{delay:4500+i*800}).show();});
        })();
        </script>
        {% endraw %}
        </body>
        </html>
        """

        from jinja2 import Environment

        env = Environment()
        template = env.from_string(html_template)
        html = template.render(report=report)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)


    # ---------------------------------------------------------------------
    # EXPORT JSON
    # ---------------------------------------------------------------------
    def export_json(self, report: Dict, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

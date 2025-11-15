"""
Enhanced Trading Persona Report Generator
Beautiful Bootstrap + Chart.js dashboard style report
"""

import json
from typing import Dict, List
from datetime import datetime
from jinja2 import Template
import markdown


class ReportGenerator:
    """Generate visually rich trading analysis reports"""

    def __init__(self, config: Dict):
        self.config = config

    def generate_report(self, metrics: Dict, patterns: Dict, analysis: Dict, trader_name: str = "Trader") -> Dict:
        """Generate complete report"""

        analysis_html = {
            key: markdown.markdown(analysis.get(key, ''), extensions=["tables", "fenced_code", "nl2br"])
            for key in ['trader_profile', 'risk_assessment', 'behavioral_insights', 'performance_summary']
        }

        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'trader_name': trader_name,
                'analysis_period': metrics.get('date_range', 'N/A')
            },
            'executive_summary': self._create_executive_summary(metrics, analysis),
            'detailed_metrics': metrics,
            'detected_patterns': patterns,
            'ai_analysis': analysis_html,
            'recommendations': self._format_recommendations(analysis.get('recommendations', '')),
            'risk_score': self._calculate_risk_score(metrics, patterns)
        }

    def _create_executive_summary(self, metrics: Dict, analysis: Dict) -> Dict:
        return {
            'total_trades': metrics.get('total_trades', 0),
            'net_pnl': metrics.get('total_pnl', 0),
            'win_rate': metrics.get('win_rate', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'risk_level': self._get_risk_level(metrics),
        }

    def _get_risk_level(self, metrics: Dict) -> str:
        sharpe = metrics.get('sharpe_ratio', 0)
        drawdown = abs(metrics.get('max_drawdown_pct', 0))
        if sharpe < 0 or drawdown > 30:
            return "VERY HIGH"
        elif sharpe < 0.5 or drawdown > 20:
            return "HIGH"
        elif sharpe < 1.0 or drawdown > 10:
            return "MEDIUM"
        else:
            return "LOW"

    def _format_recommendations(self, recommendations_text: str) -> List[str]:
        lines = recommendations_text.split('\n')
        return [line.strip('-•* ') for line in lines if line.strip()][:10]

    def _calculate_risk_score(self, metrics: Dict, patterns: Dict) -> int:
        score = 50
        if metrics.get('sharpe_ratio', 0) < 0:
            score += 20
        if abs(metrics.get('max_drawdown_pct', 0)) > 20:
            score += 15
        if metrics.get('win_rate', 50) < 45:
            score += 10
        if patterns.get('overtrading', {}).get('detected', False):
            score += 10
        if patterns.get('revenge_trading', {}).get('detected', False):
            score += 10
        return min(100, max(0, score))

    def export_html(self, report: Dict, filepath: str):
        """Export HTML report with beautiful visuals"""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Persona Report - {{ report.metadata.trader_name }}</title>

<!-- Bootstrap + Icons + Chart.js -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body {
  background: linear-gradient(135deg, #f7f9fc, #eef3f7);
  font-family: 'Inter', sans-serif;
}
.container {
  max-width: 1100px;
  margin-top: 30px;
}
.card {
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  border: none;
}
.section-title {
  font-weight: 600;
  color: #0d6efd;
  margin-top: 40px;
  border-left: 5px solid #0d6efd;
  padding-left: 12px;
}
.metric-value {
  font-size: 28px;
  font-weight: bold;
}
.badge {
  font-size: 0.9em;
}
footer {
  text-align: center;
  margin-top: 60px;
  color: #6c757d;
  font-size: 14px;
}
canvas {
  max-height: 280px;
}
</style>
</head>

<body>
<div class="container">

  <!-- Header -->
  <div class="card p-4 mb-4 bg-white">
    <h2 class="text-primary"><i class="bi bi-graph-up-arrow"></i> Trading Persona Analysis Report</h2>
    <p class="text-muted mb-1"><strong>Trader:</strong> {{ report.metadata.trader_name }}</p>
    <p class="text-muted mb-1"><strong>Analysis Period:</strong> {{ report.metadata.analysis_period }}</p>
    <p class="text-muted"><strong>Generated:</strong> {{ report.metadata.generated_at }}</p>
  </div>

  <!-- Executive Summary -->
  <h4 class="section-title">🎯 Executive Summary</h4>
  <div class="row g-3 mb-4">
    <div class="col-md-3"><div class="card p-3 text-center"><h6>Total Trades</h6><div class="metric-value">{{ report.executive_summary.total_trades }}</div></div></div>
    <div class="col-md-3"><div class="card p-3 text-center"><h6>Net P&L</h6><div class="metric-value text-success">₹{{ "%.2f"|format(report.executive_summary.net_pnl) }}</div></div></div>
    <div class="col-md-3"><div class="card p-3 text-center"><h6>Win Rate</h6><div class="metric-value">{{ "%.1f"|format(report.executive_summary.win_rate) }}%</div></div></div>
    <div class="col-md-3"><div class="card p-3 text-center"><h6>Risk Level</h6>
      <div class="metric-value">
        <span class="badge bg-{% if report.executive_summary.risk_level == 'HIGH' %}danger{% elif report.executive_summary.risk_level == 'MEDIUM' %}warning{% else %}success{% endif %}">
          {{ report.executive_summary.risk_level }}
        </span>
      </div></div>
    </div>
  </div>

  <!-- Persona Section -->
  {% if report.detailed_metrics.persona_type %}
  <h4 class="section-title">🧠 Trading Persona</h4>
  <div class="card p-4 mb-4">
    <h5>{{ report.detailed_metrics.persona_type }}</h5>
    <p>{{ report.detailed_metrics.trait_summary | safe }}</p>

    <div class="row mt-3">
      <div class="col-md-6"><canvas id="personaRadar"></canvas></div>
      <div class="col-md-6">
        <table class="table table-sm table-striped">
          <thead><tr><th>Trait</th><th>Score (0–1)</th></tr></thead>
          <tbody>
            {% for trait, value in report.detailed_metrics.persona_traits.items() %}
            <tr><td>{{ trait.replace('_',' ').title() }}</td><td>{{ '%.2f'|format(value) }}</td></tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>
  {% endif %}

  <!-- Performance Timeline -->
  {% if report.detailed_metrics.pnl_timeline %}
  <h4 class="section-title">📈 P&L Timeline</h4>
  <div class="card p-4 mb-4">
    <canvas id="pnlTimeline"></canvas>
  </div>
  {% endif %}

  <!-- AI Insights -->
  <h4 class="section-title">🤖 AI Insights</h4>
  <div class="accordion mb-4" id="aiAccordion">
    {% for key, content in report.ai_analysis.items() %}
    <div class="accordion-item">
      <h2 class="accordion-header" id="{{ key }}">
        <button class="accordion-button {% if not loop.first %}collapsed{% endif %}" type="button" data-bs-toggle="collapse" data-bs-target="#collapse{{ loop.index }}">
          {{ key.replace('_',' ').title() }}
        </button>
      </h2>
      <div id="collapse{{ loop.index }}" class="accordion-collapse collapse {% if loop.first %}show{% endif %}">
        <div class="accordion-body">{{ content | safe }}</div>
      </div>
    </div>
    {% endfor %}
  </div>

  <!-- Recommendations -->
  <h4 class="section-title">💡 Recommendations</h4>
  <ul class="list-group mb-5">
    {% for rec in report.recommendations %}
      <li class="list-group-item"><i class="bi bi-check-circle text-success me-2"></i>{{ rec }}</li>
    {% endfor %}
  </ul>

  <footer>
    <p>AI-powered report — not financial advice.</p>
    <p>© 2025 The ResolveRoom | Built by Nilesh Tiwari</p>
  </footer>
</div>

<!-- Charts -->
<script>
const persona = {{ report.detailed_metrics.persona_traits | tojson }};
if (persona) {
  new Chart(document.getElementById('personaRadar'), {
    type: 'radar',
    data: { labels: Object.keys(persona),
      datasets: [{ label: 'Traits', data: Object.values(persona),
        backgroundColor: 'rgba(13,110,253,0.2)', borderColor: '#0d6efd', borderWidth: 2 }]
    },
    options: { scales: { r: { suggestedMin: 0, suggestedMax: 1 } } }
  });
}

const pnlData = {{ report.detailed_metrics.pnl_timeline | tojson if report.detailed_metrics.pnl_timeline else 'null' }};
if (pnlData) {
  new Chart(document.getElementById('pnlTimeline'), {
    type: 'line',
    data: { labels: pnlData.dates, datasets: [{
      label: 'Cumulative P&L',
      data: pnlData.values,
      borderColor: '#198754', fill: true, backgroundColor: 'rgba(25,135,84,0.1)', tension: 0.3 }] },
    options: { scales: { y: { beginAtZero: false } } }
  });
}
</script>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
        html = Template(html_template).render(report=report)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

    def export_json(self, report: Dict, filepath: str):
        def safe(o):
            if hasattr(o, "item"): return o.item()
            if isinstance(o, (set,)): return list(o)
            return str(o)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=safe)

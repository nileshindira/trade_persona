import json
import markdown
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict
from jinja2 import Environment, FileSystemLoader


class ReportGenerator:

    def __init__(self, base_dir=None, config: Dict = None):
        """
        base_dir = directory of the project root (optional)
        auto-detects src/templates and src/static
        """
        self.config = config or {}

        # Auto-detect paths based on this file location
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.template_dir = self.base_dir / "templates"
        self.static_dir = self.base_dir / "static"
        self.themes_dir = self.base_dir / "themes"

        self.jinja = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )

    # --------------------------------------------------------------
    # GENERATE REPORT DICT
    # --------------------------------------------------------------
    def generate_report(self, metrics, patterns, analysis, trader_name="Trader"):
        analysis_text = analysis.get("analysis_text", {})
        summary = analysis.get("summary_data", {})
        web = analysis.get("web_data", {})

        # Convert markdown → HTML
        analysis_html = {
            k: markdown.markdown(v, extensions=["tables", "nl2br"])
            for k, v in analysis_text.items()
            if isinstance(v, str)
        }

        return {
            "metadata": {
                "trader_name": trader_name,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "analysis_period": metrics.get("date_range", "N/A"),
            },
            "executive_summary": self._exec_summary(metrics, summary),
            "detailed_metrics": metrics,
            "patterns": patterns,
            "analysis_text": analysis_html,
            "summary_data": summary,
            "web_data": web,
            "risk_score": self._risk_score(metrics, patterns),
        }

    # --------------------------------------------------------------
    # SUMMARY HELPERS
    # --------------------------------------------------------------
    def _exec_summary(self, m, s):
        return {
            "total_trades": s.get("total_trades", m.get("total_trades", 0)),
            "net_pnl": m.get("total_pnl_combined", 0),
            "win_rate": s.get("win_rate", 0),
            "sharpe_ratio": s.get("sharpe_ratio", 0),
            "max_drawdown_pct": s.get("max_drawdown_pct", 0),
            "risk_level": s.get("risk_level", self._risk_level(m)),
            "profit_factor": s.get("profit_factor", 0),
            "open_positions_count": m.get("open_positions_count", 0)
        }

    def _risk_level(self, m):
        sharpe, dd = m.get("sharpe_ratio", 0), abs(m.get("max_drawdown_pct", 0))
        if sharpe < 0 or dd > 30: return "VERY HIGH"
        if sharpe < 0.5 or dd > 20: return "HIGH"
        if sharpe < 1 or dd > 10: return "MEDIUM"
        return "LOW"

    def _risk_score(self, m, patterns):
        score = 50
        if m.get("sharpe_ratio", 0) < 0: score += 20
        if abs(m.get("max_drawdown_pct", 0)) > 20: score += 10
        if patterns.get("overtrading", {}).get("detected"): score += 10
        return min(score, 100)

    # --------------------------------------------------------------
    # EXPORT HTML USING TEMPLATES
    # --------------------------------------------------------------
    def export_html(self, report: Dict, filepath: str, theme="light"):
        tpl = self.jinja.get_template("report.html")

        html = tpl.render(
            report=report,
            static_path="static",
            theme_css=f"themes/{theme}.css"
        )

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        Path(filepath).write_text(html, encoding="utf-8")

    # --------------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------------
    def export_json(self, report: Dict, filepath: str):
        Path(filepath).write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8"
        )

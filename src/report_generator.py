import json
import markdown
import numpy as np
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List
from jinja2 import Environment, FileSystemLoader
import numbers
from decimal import Decimal
import re
import math


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
        self.numeric_regex = re.compile(r'^-?\d+(\.\d+)?$')
        self.jinja = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )



    def to_number_if_possible(self, val):
        """Convert numeric-like strings to float."""
        if isinstance(val, str):
            lower = val.strip().lower()

            # special cases
            if lower in ("nan", "+nan", "-nan", "inf", "+inf", "-inf"):
                try:
                    return float(lower)
                except:
                    return val

            # regular numeric pattern
            if self.numeric_regex.match(lower):
                try:
                    return float(lower)
                except:
                    return val

        return val

    # -------------------------------------------------
    # Main conversion function
    # -------------------------------------------------
    def make_jinja_safe(self, obj):
        """Convert complex objects (NaN, numpy, datetime) to Jinja-safe types."""

        # ----------------------------------------
        # Handle Python NaN / Inf first
        # ----------------------------------------
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0  # SAFE VALUE
            return obj

        # ----------------------------------------
        # NumPy numeric (may contain nan/inf)
        # ----------------------------------------
        if isinstance(obj, np.generic):
            num = float(obj)
            if math.isnan(num) or math.isinf(num):
                return 0.0
            return num

        # ----------------------------------------
        # Dict
        # ----------------------------------------
        if isinstance(obj, dict):
            return {k: self.make_jinja_safe(v) for k, v in obj.items()}

        # ----------------------------------------
        # List / Tuple
        # ----------------------------------------
        if isinstance(obj, (list, tuple)):
            return [self.make_jinja_safe(v) for v in obj]

        # ----------------------------------------
        # Boolean → convert to string
        # ----------------------------------------
        if isinstance(obj, bool):
            return "True" if obj else "False"

        # ----------------------------------------
        # None → empty string
        # ----------------------------------------
        if obj is None:
            return ""

        # ----------------------------------------
        # Decimal → float
        # ----------------------------------------
        if isinstance(obj, Decimal):
            return float(obj)

        # ----------------------------------------
        # datetime/date → ISO string
        # ----------------------------------------
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        # ----------------------------------------
        # Strings (may be numeric-like)
        # ----------------------------------------
        if isinstance(obj, str):
            return self.to_number_if_possible(obj)

        # ----------------------------------------
        # Everything else → string
        # ----------------------------------------
        return str(obj)

    # --------------------------------------------------------------
    # GENERATE REPORT DICT
    # --------------------------------------------------------------


    # def generate_report(self, metrics, patterns, analysis, trader_name="Trader"):
    #
    #     # Extract original analysis parts
    #     analysis_raw_text = analysis.get("analysis_text", {})
    #     summary = analysis.get("summary_data", {})
    #     web = analysis.get("web_data", {})
    #
    #     # Convert Markdown → HTML only once
    #     analysis_html = {
    #         k: markdown.markdown(v, extensions=["extra", "tables", "nl2br"])
    #         for k, v in analysis_raw_text.items()
    #         if isinstance(v, str)
    #     }
    #
    #     # -----------------------------------------
    #     #  CLEANED & DEDUPLICATED OUTPUT STRUCTURE
    #     # -----------------------------------------
    #     return {
    #         "metadata": {
    #             "trader_name": trader_name,
    #             "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    #             "analysis_period": metrics.get("date_range", "N/A"),
    #         },
    #
    #         # EXEC SUMMARY
    #         "executive_summary": self._exec_summary(metrics, summary),
    #
    #         # FULL METRICS (no duplication of summary or KPIs)
    #         "metrics": metrics,
    #
    #         # PATTERNS (single reference instead of two)
    #         "patterns": patterns,
    #
    #         # MARKDOWN → HTML (used for AI insights + text sections)
    #         "analysis_html": analysis_html,
    #
    #         # RAW TEXT (kept minimal, used only if needed)
    #         "analysis_raw": analysis_raw_text,
    #
    #         # Clean web data
    #         "web": {
    #             "kpis": web.get("kpis", {}),
    #             "persona_scores": web.get("persona_scores", {}),
    #             "charts": web.get("charts", {}),
    #         },
    #
    #         # AI INSIGHTS (directly mapped to analysis_html, not duplicated)
    #         "ai_insights": analysis_html,
    #
    #         # RECOMMENDATIONS extracted from markdown
    #         "recommendations": self._format_recommendations(
    #             analysis_raw_text.get("recommendations", "")
    #         ),
    #
    #         # Risk score (functional)
    #         "risk_score": self._risk_score(metrics, patterns),
    #     }

    def generate_report(self, metrics, patterns, analysis, trader_name="Trader"):
        analysis_text = analysis.get("analysis_text", {})
        summary = analysis.get("summary_data", {})
        web = analysis.get("web_data", {})

        # Convert markdown → HTML
        analysis_html = {
            k: markdown.markdown(v, extensions=["extra", "tables", "nl2br"])
            for k, v in analysis_text.items()
            if isinstance(v, str)
        }
        print("Analysis HTML")
        print(analysis_html)

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
            "detected_patterns": patterns,
            "ai_analysis": analysis_html,
            "recommendations": self._format_recommendations(analysis_text.get("recommendations", "")),
            "all_kpis": web.get("kpis", {}),
            "persona_scores": web.get("persona_scores", {}),
            "charts_data": web.get("charts", {}),
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

    def _format_recommendations(self, text: str) -> List[str]:
        if not text:
            return []
        lines = [l.strip("-•* ").strip() for l in text.splitlines() if l.strip() and len(l.strip()) > 10]
        return lines[:30]  # Increased to show more recommendations

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
        report_safe = self.make_jinja_safe(report)
        html = tpl.render(
            report=report_safe,
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





def export_html_from_json(json_path, html_path, base_dir=None):
    gen = ReportGenerator(base_dir)

    report = json.loads(Path(json_path).read_text(encoding="utf-8"))
    gen.export_html(report, html_path)




if __name__ == "__main__":
    with open("/home/system-4/PycharmProjects/trade_persona/data/reports/Trader_report.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    gen = ReportGenerator()
    gen.export_html(data, "/home/system-4/PycharmProjects/trade_persona/data/reports/restored_report.html", theme="light")
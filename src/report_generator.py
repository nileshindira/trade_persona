import json
import os

import markdown
import numpy as np
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List
from jinja2 import Environment, FileSystemLoader
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
        self.jinja.filters['markdown'] = lambda text: markdown.markdown(str(text), extensions=['extra', 'tables', 'fenced_code'])



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


    def generate_report(self, metrics, patterns, analysis, trader_name="Trader"):
        """
        Main entry point for report generation. 
        Returns a structured dictionary optimized for clear information hierarchy.
        """
        master = analysis.get("master_persona", {})
        web = analysis.get("web_data", {})
        
        # 1. Build HERO section
        identity = master.get("identity", {})
        hero = {
            "trader_name": trader_name,
            "analysis_period": metrics.get("date_range", "N/A"),
            "persona_name": identity.get("trader_name", "Unknown Trader"),
            "headline": master.get("headline", "Trading performance analysis"),
            "topline": {
                "net_pnl": metrics.get("total_pnl_combined", metrics.get("total_pnl", 0)),
                "win_rate": metrics.get("win_rate", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "max_drawdown_pct": metrics.get("max_drawdown_pct", 0)
            }
        }

        # 2. Build DIAGNOSIS section
        def _get_val(obj, key, fallback=None):
            val = obj.get(key)
            if not val or val == "N/A" or val == "Unknown":
                return fallback
            return val

        _archetype_name = metrics.get("archetype", {}).get("name", "")
        
        diagnosis = {
            "trader_type":    _get_val(identity, "trader_type") or _get_val(identity, "primary_strategy") or _archetype_name or "Unknown",
            "inferred_style": _get_val(identity, "primary_strategy", _archetype_name or "N/A"),
            "natural_edge":   _get_val(identity, "natural_edge", _get_val(identity, "narrative", "Analyze your data for a natural edge."))[:500],
            "risk_profile":   identity.get("risk_profile", {"appetite": "N/A", "handling": "N/A"}),
            "narrative":      identity.get("narrative", "Analysis pending..."),
            "efficiency_metrics": analysis.get("efficiency_metrics", {}),
            "market_context":   analysis.get("market_context_metrics", {}),
        }



        # 3. Build improvement plan
        plan = master.get("improvement_plan", {})
        improvement_plan = {
            "next_5_sessions": plan.get("next_5_sessions", []),
            "next_30_days":    plan.get("next_30_days",    []),
            "biggest_lever":   plan.get("biggest_lever",   "N/A"),
        }

        # 4. Populate Evidence Ledger from LLM strengths + mistakes
        evidence_ledger = []
        for s in master.get("strengths", []):
            evidence_ledger.append({
                "claim":              s.get("title", ""),
                "confidence":         8,
                "supporting_metrics": [s.get("evidence", "")],
                "supporting_trades":  [],
            })
        for m in master.get("mistakes", []):
            evidence_ledger.append({
                "claim":              m.get("title", ""),
                "confidence":         7,
                "supporting_metrics": [m.get("evidence", "")],
                "supporting_trades":  [],
            })

        # 5. Construct Final Report
        archetype = metrics.get("archetype", {})
        report = {
            "hero": hero,
            "diagnosis":       diagnosis,
            "strengths":       master.get("strengths",              []),
            "mistakes":        master.get("mistakes",               []),
            "improvement_plan": improvement_plan,
            "transformation":  master.get("simulated_transformation", []),
            "final_verdict":   master.get("final_verdict",          ""),
            "evidence_ledger": evidence_ledger,
            # Persona enrichment (Phase 2)
            "archetype":       archetype,
            "trade_dna":       metrics.get("trade_dna",            {}),
            "pressure_map":    metrics.get("behavioral_pressure_map", {}),
            "consistency_score": metrics.get("behavioral_consistency_score", 0),
            "emotional_leakage": metrics.get("emotional_leakage_index",      0),
            # Pressure tendencies for the cognitive bias section
            "pressure_patterns": metrics.get("behavioral_pressure_map", {}).get("tendencies", []),
            "metadata": {
                "generated_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
                "risk_score":    self._risk_score(metrics, patterns),
                "risk_severity": web.get("risk_severity", "LOW"),
            },
            "web_data":      web,
            "analysis_text": analysis.get("analysis_text", {}),
            "hard_flags":    web.get("hard_flags", {}),
            "appendix": {
                "metrics":        metrics,
                "charts":         web.get("charts",         {}),
                "positions":      (metrics.get("positions", []) + metrics.get("closed_positions", [])),
                "persona_scores": web.get("persona_scores", {}),
                "patterns":       patterns,
            }
        }

        return report


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
            "open_positions_count": m.get("open_positions_count", 0),
            "day_mtm": m.get("day_mtm", 0),
            "risk_handling_score": m.get("persona_traits", {}).get("risk_handling", 0)
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
        
        # Read theme content to embed directly
        theme_path = self.themes_dir / f"{theme}.css"
        if theme_path.exists():
            theme_css_content = theme_path.read_text(encoding="utf-8")
        else:
            theme_css_content = ""

        html = tpl.render(
            report=report_safe,
            static_path="../../src/static",
            theme_css=f"../../src/static/css/themes/{theme}.css", # Keep as backup
            theme_css_html=f"<style>\n{theme_css_content}\n</style>" if theme_css_content else "" # New embedded content
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

    with open(r"/home/system-4/PycharmProjects/trade_persona/data/reports/Anish_report.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Patch for missing context_performance in older JSONs
    if "analysis_text" in data and "context_performance" not in data["analysis_text"]:
        data["analysis_text"]["context_performance"] = {
            "event": {"verdict": "N/A", "reasoning": "Not analyzed"},
            "news": {"verdict": "N/A", "reasoning": "Not analyzed"},
            "volume": {"verdict": "N/A", "reasoning": "Not analyzed"},
            "trend": {"verdict": "N/A", "reasoning": "Not analyzed"}
        }

    gen = ReportGenerator()
    gen.export_html(data, "/home/system-4/PycharmProjects/trade_persona/data/reports/restored1_report.html", theme="light")
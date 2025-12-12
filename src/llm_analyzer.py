"""
LLM Analyzer Module
Integrates with Ollama for AI-powered analysis
"""

import requests
import json
from typing import Dict, List
import logging
import re
import pandas as pd
import markdown

class OllamaAnalyzer:
    """LLM-based analysis using Ollama"""

    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config['ollama']['base_url']
        self.model = config['ollama']['model']
        self.logger = logging.getLogger(__name__)

    # --- (Placeholder for existing Ollama methods like _call_ollama, _prepare_context, etc.) ---
    # NOTE: These methods are omitted here for brevity, assume they exist.
    # The crucial addition is in generate_analysis and the new preparation methods.

    # =========================================================
    # New: Core Data Structure Preparation for Web Display
    # =========================================================

    def _prepare_dashboard_kpis(self, metrics: Dict, patterns: Dict) -> Dict:
        """
        Prepares high-level KPIs and behavioral scores for the dashboard.
        """
        kpis = {
            # 1. Financial Health & Risk
            'pnl_total': metrics.get('total_pnl', 0),
            'roc_pct': metrics.get('return_on_capital', 0),
            'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'sortino_ratio': metrics.get('sortino_ratio', 0),
            'value_at_risk_95': metrics.get('value_at_risk_95', 0),

            # 2. Efficiency & Activity
            'win_rate_pct': metrics.get('win_rate', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'efficiency_ratio': metrics.get('efficiency_ratio', 0),
            'total_trades': metrics.get('total_trades', 0),
            'avg_holding_period_min': metrics.get('avg_holding_period', 0),

            # 3. Behavioral Scores & Patterns
            'persona_type': metrics.get('persona_type', 'N/A'),
            'discipline_score': metrics.get('persona_traits', {}).get('discipline_score', 0),
            'emotional_control': metrics.get('persona_traits', {}).get('emotional_control', 0),
            'risk_appetite': metrics.get('persona_traits', {}).get('risk_appetite', 0),
            'overtrading_detected': patterns.get('overtrading', {}).get('detected', False),
            'revenge_trading_detected': patterns.get('revenge_trading', {}).get('detected', False),
            'fomo_trading_detected': patterns.get('fomo_trading', {}).get('detected', False),

            # 4. Technical Alignment
            'avg_total_score': metrics.get('avg_total_score', 0),
            'high_score_win_rate_pct': metrics.get('high_score_win_rate', 0),
        }
        return kpis

    def _prepare_chart_data(self, metrics: Dict,patterns: Dict, df: pd.DataFrame) -> Dict:
        """
        Formats data arrays for charting libraries (e.g., P&L timeline, distribution).
        """
        chart_data = {}

        # P&L Timeline (For line/area chart)
        # Assuming 'pnl_timeline' is a list of cumulative PnL points over time
        # NOTE: If pnl_timeline isn't structured as required, you'd need to re-index the trades df.
        chart_data['pnl_timeline'] = metrics.get('pnl_timeline', [])

        # Day-by-Day MTM (For bar chart)
        # Assuming 'day_mtm' is a list of daily PnL values
        chart_data['day_mtm'] = metrics.get('day_mtm', [])

        # Instrument Clustering (For pie chart)
        # Format as list of {'name': 'Instrument', 'value': 12.3}
        inst_cluster = patterns.get('instrument_clustering', {})
        chart_data['instrument_distribution'] = metrics.get("chart_data")["asset_clusters"]
        chart_data['segment_distribution'] = metrics.get("symbol_cluster")
        chart_data['symbol_distribution_raw'] = metrics["symbol_cluster"]
        # Add other instruments as needed

        # Win/Loss Distribution (For simple bar chart)
        chart_data['win_loss_amounts'] = {
            'avg_win': metrics.get('avg_win', 0),
            'avg_loss': metrics.get('avg_loss', 0),
            'largest_win': metrics.get('largest_win', 0),
            'largest_loss': metrics.get('largest_loss', 0),
        }

        return chart_data

    def _beautify_recommendations_html(self, md_text: str) -> str:
        """
        Takes LLM Markdown output and converts it into
        BEAUTIFULLY STYLED HTML COMPONENTS for your UI.
        """



        # Convert markdown → basic HTML first
        html = markdown.markdown(md_text, extensions=["extra", "tables"])

        # --- Beautify Tables ---
        html = re.sub(
            r"<table>",
            "<div class='table-responsive'><table class='table table-bordered table-sm shadow-sm'>",
            html
        )
        html = html.replace("</table>", "</table></div>")

        # --- Beautify Bullet Sections (wrap in rec-box) ---
        html = re.sub(
            r"<ul>",
            "<div class='rec-box'><ul>",
            html
        )
        html = html.replace("</ul>", "</ul></div>")

        # --- Beautify Checkboxes ---
        html = html.replace("[ ]", "<span class='checkbox'>☐</span>")
        html = html.replace("[x]", "<span class='checkbox checked'>☑</span>")

        # --- Optional section title detection ---
        html = re.sub(
            r"<h[1-6]>(.*?)</h[1-6]>",
            r"<h4 class='rec-title'>\1</h4>",
            html
        )

        return html

    def generate_analysis(self, metrics: Dict, patterns: Dict, df: pd.DataFrame) -> Dict:
        context = self._prepare_context(metrics, patterns, df)
        prompt = self._build_unified_prompt(context)

        llm_json = self._call_ollama(prompt)
        llm_output = self._safe_json_load(llm_json)  # <-- FIXED HERE

        web_kpis = self._prepare_dashboard_kpis(metrics, patterns)
        web_charts = self._prepare_chart_data(metrics, patterns, df)

        return {
            "analysis_text": llm_output,
            "summary_data": llm_output,
            "web_data": {
                "kpis": web_kpis,
                "charts": web_charts,
                "persona_scores": metrics.get("persona_traits", {}),
                "raw_patterns": patterns
            }
        }

    # =========================================================
    # Context Preparation
    # =========================================================
    def _prepare_context(self, metrics: Dict, patterns: Dict, df: pd.DataFrame) -> str:
        """Prepare context for LLM with JSON-safe conversion"""
        import numpy as np
        import pandas as pd

        def make_json_safe(obj):
            """Convert non-serializable types to safe Python types"""
            if isinstance(obj, (np.bool_, np.bool)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (pd.Timestamp,)):
                return obj.isoformat()
            if isinstance(obj, (set,)):
                return list(obj)
            return str(obj)

        # safely serialize complex objects
        try:
            safe_metrics = json.dumps(metrics, indent=2, default=make_json_safe)
        except Exception as e:
            safe_metrics = f"Error serializing metrics: {e}\n{str(metrics)}"

        try:
            safe_patterns = json.dumps(patterns, indent=2, default=make_json_safe)
        except Exception as e:
            safe_patterns = f"Error serializing patterns: {e}\n{str(patterns)}"

        # === Base Context ===
        context = f"""
    TRADING METRICS:
    - Total Trades: {metrics.get('total_trades', 0)}
    - Total P&L: ₹{metrics.get('total_pnl', 0):,.2f}
    - Win Rate: {metrics.get('win_rate', 0):.2f}%
    - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
    - Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
    - Profit Factor: {metrics.get('profit_factor', 0):.2f}
    - Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%
    - Average Trade Value: ₹{metrics.get('avg_trade_value', 0):,.2f}
    - Return on Capital: {metrics.get('return_on_capital', 0):.2f}%
    - Efficiency Ratio: {metrics.get('efficiency_ratio', 0):.2f}
    - Avg F-Score: {metrics.get('avg_f_score', 0):.2f}
    - Avg T-Score: {metrics.get('avg_t_score', 0):.2f}
    - Avg Total Score: {metrics.get('avg_total_score', 0):.2f}
    
    DETECTED PATTERNS:
    - Overtrading: {patterns.get('overtrading', {}).get('detected', False)}
    - Revenge Trading: {patterns.get('revenge_trading', {}).get('detected', False)}
    - Pyramiding: {patterns.get('pyramiding', {}).get('detected', False)}
    - Scalping: {patterns.get('scalping', {}).get('detected', False)}
    - Hedging: {patterns.get('hedging', {}).get('detected', False)}
    - FOMO Trading: {patterns.get('fomo_trading', {}).get('detected', False)}
    - Overconfidence: {patterns.get('overconfidence', {}).get('detected', False)}
    - Weekend Exposure: {patterns.get('weekend_exposure', {}).get('detected', False)}
    """

        # === NEW SECTION: Include Persona Metrics ===
        persona_traits = metrics.get("persona_traits", {})
        context += f"""
    TRADING PERSONA:
    - Persona Type: {metrics.get('persona_type', 'N/A')}
    - Discipline Score: {persona_traits.get('discipline_score', 0):.2f}
    - Emotional Control: {persona_traits.get('emotional_control', 0):.2f}
    - Risk Appetite: {persona_traits.get('risk_appetite', 0):.2f}
    - Patience: {persona_traits.get('patience', 0):.2f}
    - Adaptability: {persona_traits.get('adaptability', 0):.2f}
    - Consistency: {persona_traits.get('consistency', 0):.2f}
    - Confidence: {persona_traits.get('confidence', 0):.2f}

    Summary:
    {metrics.get('trait_summary', 'No persona data available.')}
    """

        # === ✅ NEW SECTION: Include Trades DataFrame ===
        context += self._format_trades_data(df)

        # === Additional Context (Raw JSON) ===
        context += f"""
    Additional Context:
    {safe_metrics}
    {safe_patterns}
    """

        return context

    def _format_trades_data(self, df: pd.DataFrame) -> str:
        """Send RAW trades DataFrame to LLM - no preprocessing"""

        if df is None or df.empty:
            return "\nTRADES DATA: No trades available\n"

        # Simple header
        trades_context = f"""
    ================================================================================
    RAW TRADES DATASET ({len(df)} trades)
    ================================================================================
    
    """

        try:
            # Convert DataFrame to JSON with proper handling of datetime and NaN values
            import numpy as np

            # Replace NaN with None for proper JSON serialization
            df_clean = df.replace({np.nan: None})

            # Convert to JSON - RAW data only
            all_trades_json = df_clean.to_json(orient='records', date_format='iso', indent=2)
            trades_context += all_trades_json

        except Exception as e:
            # Fallback: convert to dict if JSON serialization fails
            self.logger.warning(f"JSON serialization failed, using dict format: {e}")
            trades_context += json.dumps(df.to_dict('records'), indent=2, default=str)

        trades_context += "\n\n================================================================================"

        return trades_context

    def _safe_json_load(self, text: str) -> dict:
        """Safe JSON loader that prevents LLM noise from breaking code."""
        try:
            return json.loads(text)
        except Exception:
            # Attempt to extract JSON substring
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass

            self.logger.error("Unable to decode JSON from LLM output")
            return {}

    def _call_ollama(self, prompt: str) -> str:
        """
        FIXED:
        - Forces non-streaming output
        - Forces Ollama to return ONE final JSON blob
        - Strips noise before JSON extraction
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False  # <-- FIXED: Prevent token streaming
        }

        try:
            res = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300
            )
            res.raise_for_status()

            raw = res.json().get("response", "")

            # CLEAN unwanted markdown, text, spacing before JSON
            raw = raw.strip()

            # remove backticks ```
            raw = raw.replace("```json", "").replace("```", "")

            # Find JSON block even if model talks
            json_match = re.search(r"\{.*\}$", raw, re.S)
            if json_match:
                cleaned = json_match.group(0)
            else:
                cleaned = raw

            return cleaned

        except Exception as e:
            self.logger.error(f"Ollama call failed: {e}")
            return "{}"

    def _build_unified_prompt(self, context: str) -> str:
        return f"""
    You are an expert trading psychologist, quant analyst, and portfolio risk evaluator.

    Below is the FULL trading dataset and computed metrics.  
    Analyze everything deeply and return **ONLY a valid JSON object** in the structure shown.

    =====================================================================
    CONTEXT (metrics + patterns + persona + trades)
    =====================================================================
    {context}

    =====================================================================
    STRICT OUTPUT FORMAT (MANDATORY)
    =====================================================================
    Return ONLY a JSON object with this exact structure:

    {{
      "trader_profile": {{
          "style": "...",
          "strengths": ["...", "..."],
          "weaknesses": ["...", "..."],
          "persona_summary": "..."
      }},
      "risk_assessment": {{
          "risk_level": "Low | Medium | High",
          "key_risks": ["...", "..."],
          "max_drawdown_comment": "...",
          "position_sizing_comment": "..."
      }},
      "behavioral_insights": {{
          "biases_detected": ["...", "..."],
          "emotional_patterns": ["...", "..."],
          "discipline_score_comment": "...",
          "psychology_summary": "..."
      }},
      "recommendations": {{
          "top_improvements": ["...", "...", "..."],
          "risk_controls": ["...", "..."],
          "trading_rules": ["...", "..."],
          "habit_changes": ["...", "..."]
      }},
      "performance_summary": {{
          "pnl_quality": "...",
          "score_interpretation": "...",
          "overall_verdict": "..."
      }}
    }}

    =====================================================================
    RULES:
    - The JSON must be syntactically perfect.
    - No markdown.
    - No explanation.
    - No extra keys.
    - No commentary outside the JSON.
    - Keep text concise and factual, based on CONTEXT.
    =====================================================================
    """

    # =========================================================
    # NEW: Structured Summary Extraction
    # =========================================================
    def _extract_structured_summary(self, sections: Dict, metrics: Dict, patterns: Dict) -> Dict:
        """Extract structured summary (risk level, verdict, key strengths/weaknesses)"""

        text = " ".join(sections.values())

        summary = {}

        # Extract Risk Level
        risk_match = re.search(r"(LOW|MEDIUM|HIGH|VERY HIGH)", text, re.IGNORECASE)
        summary["risk_level"] = risk_match.group(1).upper() if risk_match else "UNKNOWN"

        # Extract Verdict
        verdict_match = re.search(r"(EXCELLENT|GOOD|AVERAGE|POOR|CRITICAL)", text, re.IGNORECASE)
        summary["performance_verdict"] = verdict_match.group(1).upper() if verdict_match else "N/A"

        # Extract Strengths & Weaknesses (basic bullet detection)
        # Extract Strengths & Weaknesses (compatible with Python 3.13+)
        strengths_match = re.search(r"[Ss]trengths?:\s*(.*?)(?:[Ww]eaknesses?:|$)", text, re.DOTALL)
        weaknesses_match = re.search(r"[Ww]eaknesses?:\s*(.*)", text, re.DOTALL)

        summary["strengths"] = strengths_match.group(1).strip() if strengths_match else ""
        summary["weaknesses"] = weaknesses_match.group(1).strip() if weaknesses_match else ""

        # Add Derived Metrics Highlights
        summary["total_trades"] = metrics.get("total_trades", 0)
        summary["win_rate"] = metrics.get("win_rate", 0)
        summary["profit_factor"] = metrics.get("profit_factor", 0)
        summary["max_drawdown_pct"] = metrics.get("max_drawdown_pct", 0)
        summary["return_on_capital"] = metrics.get("return_on_capital", 0)
        summary["avg_total_score"] = metrics.get("avg_total_score", 0)
        summary["avg_t_score"] = metrics.get("avg_t_score", 0)
        summary["avg_f_score"] = metrics.get("avg_f_score", 0)
        summary["sharpe_ratio"] = metrics.get("sharpe_ratio", 0)
        summary["sortino_ratio"] = metrics.get("sortino_ratio", 0)

        # Add detected patterns summary
        summary["detected_patterns"] = [p for p, v in patterns.items() if v.get("detected", False)]

        # Simple performance score for dashboards
        score_map = {"EXCELLENT": 90, "GOOD": 75, "AVERAGE": 60, "POOR": 40, "CRITICAL": 25}
        summary["performance_score"] = score_map.get(summary["performance_verdict"], 50)

        return summary

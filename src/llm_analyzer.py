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
import numpy as np
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
        chart_data['segment_distribution']=metrics.get("symbol_cluster")
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

        import markdown
        import re

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


    # =========================================================
    # Main Integration Function
    # =========================================================
    def generate_analysis(self, metrics: Dict, patterns: Dict, df: pd.DataFrame) -> Dict:
        """Generate comprehensive analysis using LLM"""
        df_small = df[['symbol', 'trade_date','transaction_type','quantity',
                       'price','trade_value','t_score', 'f_score', 'total_score',
                       'is_52week_high', 'is_52week_low', 'is_alltime_high',
                        'is_alltime_low', 'is_event', 'atr', 'is_high_volume',
                        'market_behaviour', 'chart_charts']].copy()

        # Prepare context (NOW includes DataFrame)
        context = self._prepare_context(metrics, patterns, df_small)

        self.logger.info("1. Generate text analysis from LLM")
        analysis_text = {
            'trader_profile': self._analyze_trader_profile(context),
            'risk_assessment': self._analyze_risk(context),
            'behavioral_insights': self._analyze_behavior(context),
            'recommendations': self._generate_recommendations(context),
            'performance_summary': self._summarize_performance(context),
        }

        self.logger.info("2. Extract structured LLM summary")
        structured_summary = self._extract_structured_summary(analysis_text, metrics, patterns)

        self.logger.info("3. NEW: Prepare dedicated data structures for the Web Page UI")
        web_kpis = self._prepare_dashboard_kpis(metrics, patterns)
        web_charts = self._prepare_chart_data(metrics,patterns, df)


        self.logger.info("Final Output for API/Web Service")
        analysis = {
            "analysis_text": analysis_text,        # LLM generated text blocks (for display)
            "summary_data": structured_summary,   # Extracted LLM verdict/score (for dashboard)
            "web_data": {                         # **NEW BLOCK: Clean, structured data for the UI**
                "kpis": web_kpis,                 # Key single-value metrics (Section 1 & 2)
                "charts": web_charts,             # Arrays for charts (Section 3)
                "persona_scores": metrics.get("persona_traits", {}), # (Section 4)
                "raw_patterns": patterns          # Full pattern detail
            }
        }
        return analysis

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

    # =========================================================
    # Ollama API Interaction
    # =========================================================
    def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config['ollama']['temperature'],
                        "top_p": self.config['ollama']['top_p']
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                self.logger.error(f"Ollama API error: {response.text}")
                return "Error generating analysis"

        except Exception as e:
            self.logger.error(f"Error calling Ollama: {str(e)}")
            return "Error generating analysis - Ollama may not be running"

    # =========================================================
    # Analysis Sections
    # =========================================================
    def _analyze_trader_profile(self, context: str) -> str:
        """Analyze trader profile"""
        prompt = f"""
Based on the following trading data, provide a detailed trader profile classification.
Include:
1. Trader type (scalper, day trader, swing trader, etc.)
2. Risk appetite (conservative, moderate, aggressive)
3. Trading style characteristics
4. How the persona traits reflect in actual trade behavior
Ensure to keep response in simple interactive english for layman trader in india. 

{context}

Provide a concise but comprehensive trader profile (200-300 words):
"""
        system_prompt = "You are an expert financial analyst specializing in trading behavior analysis."
        self.logger.info("1.1 Generate _analyze_trader_profile from LLM")
        raw = self._call_ollama(prompt, system_prompt)
        return self._beautify_recommendations_html(raw)

    def _analyze_risk(self, context: str) -> str:
        """Analyze risk profile"""
        prompt = f"""
Based on the following trading metrics and persona traits, provide a risk assessment.

{context}

Analyze:
1. Overall risk level (LOW/MEDIUM/HIGH/VERY HIGH)
2. Key risk factors
3. Risk-adjusted performance
4. Potential vulnerabilities based on persona behavior
Ensure to keep response in simple interactive english for layman trader in india
Provide detailed risk analysis (200-300 words):
"""
        system_prompt = "You are a risk management expert analyzing trading portfolios."
        self.logger.info("1.2 Generate _analyze_risk from LLM")
        return self._call_ollama(prompt, system_prompt)

    def _analyze_behavior(self, context: str) -> str:
        """Analyze behavioral patterns"""
        prompt = f"""
Based on the detected patterns and persona traits, provide behavioral insights.

{context}

Focus on:
1. Psychological tendencies
2. Emotional trading signs
3. Discipline issues
4. Positive behaviors and consistency traits
Ensure to keep response in simple interactive english for layman trader in india
Provide behavioral analysis (200-300 words):
"""
        system_prompt = "You are a trading psychology expert analyzing trader behavior."
        self.logger.info("1.3 Generate _analyze_behavior from LLM")
        return self._call_ollama(prompt, system_prompt)

    def _generate_recommendations(self, context: str) -> str:
        """Generate actionable recommendations"""
        prompt = f"""
Based on the trading metrics, patterns, and persona traits, provide specific, actionable recommendations.

{context}

Provide:
1. Immediate actions (next 1-2 weeks)
2. Short-term improvements (1-3 months)
3. Long-term strategy changes
4. Specific metrics or traits to improve
Ensure to keep response in simple interactive english for layman trader in india
Format as bullet points with clear action items:
"""
        system_prompt = "You are a professional trading coach providing improvement strategies."
        self.logger.info("1.4 Generate _generate_recommendations from LLM")
        return self._call_ollama(prompt, system_prompt)

    def _summarize_performance(self, context: str) -> str:
        """Summarize overall performance"""
        prompt = f"""
Provide an executive summary of the trading performance.

{context}

Include:
1. Overall verdict (Excellent/Good/Average/Poor/Critical)
2. Key strengths
3. Major weaknesses
4. Bottom-line assessment integrating persona analysis
Ensure to keep response in simple interactive english for layman trader in india
Be direct and honest in assessment (150-200 words):
"""
        system_prompt = "You are a senior financial advisor providing performance reviews."
        self.logger.info("1.5 Generate _summarize_performance from LLM")

        return self._call_ollama(prompt, system_prompt)

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

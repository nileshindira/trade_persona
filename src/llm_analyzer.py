"""
LLM Analyzer Module
Integrates with Ollama for AI-powered analysis
"""
import os

import requests
import json
from typing import Dict, List
import logging
import re
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


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

            # 5. NEW: Contextual Performance Metrics
            'trend_alignment_score': metrics.get('trend_alignment_score', 0),
            'trend_aligned_win_rate': metrics.get('trend_aligned_win_rate', 0),
            'event_trading_count': metrics.get('event_trading_count', 0),
            'event_trading_win_rate': metrics.get('event_trading_win_rate', 0),
            'event_trading_pnl': metrics.get('event_trading_pnl', 0),
            'news_trading_count': metrics.get('news_trading_count', 0),
            'news_trading_win_rate': metrics.get('news_trading_win_rate', 0),
            'news_trading_pnl': metrics.get('news_trading_pnl', 0),
            'high_volume_trading_count': metrics.get('high_volume_trading_count', 0),
            'high_volume_trading_win_rate': metrics.get('high_volume_trading_win_rate', 0),
            'high_volume_trading_pnl': metrics.get('high_volume_trading_pnl', 0),

            # 6. Price Extremes & Chart Quality (NEW)
            'ath_trading_count': metrics.get('ath_trading_count', 0),
            'ath_trading_win_rate': metrics.get('ath_trading_win_rate', 0),
            'ath_trading_pnl': metrics.get('ath_trading_pnl', 0),
            'atl_trading_count': metrics.get('atl_trading_count', 0),
            'atl_trading_win_rate': metrics.get('atl_trading_win_rate', 0),
            'atl_trading_pnl': metrics.get('atl_trading_pnl', 0),
            'good_chart_trading_count': metrics.get('good_chart_trading_count', 0),
            'good_chart_trading_win_rate': metrics.get('good_chart_trading_win_rate', 0),
            'good_chart_trading_pnl': metrics.get('good_chart_trading_pnl', 0),
            'bad_chart_trading_count': metrics.get('bad_chart_trading_count', 0),
            'bad_chart_trading_win_rate': metrics.get('bad_chart_trading_win_rate', 0),
            'bad_chart_trading_pnl': metrics.get('bad_chart_trading_pnl', 0),
            'chart_behavior_breakdown': metrics.get('chart_behavior_breakdown', []),
        }
        return kpis

    def _compute_verdict_ceiling(self, hard_flags: Dict) -> str:
        if hard_flags.get("negative_expectancy"):
            return "AVERAGE"
        if hard_flags.get("capital_at_risk_high"):
            return "GOOD"
        if hard_flags.get("severe_emotional_trading"):
            return "AVERAGE"
        return "EXCELLENT"

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

        chart_data['chart_quality_distribution'] = metrics.get('chart_behavior_breakdown', [])

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
        Path("chatgpt_context.json").write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
        Path("df_context.json").write_text(
            json.dumps(df.to_dict(orient="records"), indent=2, default=str),
            encoding="utf-8"
        )
        self.logger.info("1. Generate text analysis from LLM")
        analysis_text = {
            'trader_profile': self._analyze_trader_profile(context),
            'risk_assessment': self._analyze_risk(context),
            'behavioral_insights': self._analyze_behavior(context),
            'context_performance': self._analyze_context_performance(context),  # NEW
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
            "analysis_text": analysis_text,
            "summary_data": structured_summary,
            "web_data": {
                "kpis": web_kpis,
                "charts": web_charts,
                "persona_scores": metrics.get("persona_traits", {}),
                "hard_flags": metrics.get("_hard_flags", {}),  # 👈 STEP 6
                "raw_patterns": patterns
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

    BEHAVIORAL CONTEXT (NEW):
    - Trend Alignment Score: {metrics.get('trend_alignment_score', 0):.1f}% (Win Rate: {metrics.get('trend_aligned_win_rate', 0):.1f}%)
    - Event Trading: {metrics.get('event_trading_count', 0)} trades (Win Rate: {metrics.get('event_trading_win_rate', 0):.1f}%, PnL: ₹{metrics.get('event_trading_pnl', 0):,.2f})
    - News Trading: {metrics.get('news_trading_count', 0)} trades (Win Rate: {metrics.get('news_trading_win_rate', 0):.1f}%, PnL: ₹{metrics.get('news_trading_pnl', 0):,.2f})
    - News Categories: {metrics.get('news_category_breakdown', 'N/A')}
    - High Volume Trading: {metrics.get('high_volume_trading_count', 0)} trades (Win Rate: {metrics.get('high_volume_trading_win_rate', 0):.1f}%, PnL: ₹{metrics.get('high_volume_trading_pnl', 0):,.2f})
    - Volume Volatility: {metrics.get('volume_volatility', 0):.2f} (High = Erratic Sizing)
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
        hard_flags = metrics.get("_hard_flags", {})

        context += f"""
        HARD RISK FLAGS (NON-NEGOTIABLE):
        - Capital at Risk High: {hard_flags.get('capital_at_risk_high')}
        - Overtrading: {hard_flags.get('overtrading')}
        - Negative Expectancy: {hard_flags.get('negative_expectancy')}
        - Emotional Instability: {hard_flags.get('emotional_instability')}

        IMPORTANT:
        Do NOT override these facts in narrative.
        Align conclusions strictly with these flags.
        """

        return context

    def _format_trades_data(self, df: pd.DataFrame) -> str:
        """Send RAW trades DataFrame to LLM - limited to avoid context overflow"""

        if df is None or df.empty:
            return "\nTRADES DATA: No trades available\n"

        # --- CRITICAL FIX: LIMIT TRADES FOR LLM CONTEXT ---
        max_trades = 100
        if len(df) > max_trades:
            self.logger.info(f"Trimming trades dataset from {len(df)} to {max_trades} for LLM context")
            # Sample: First 40, Last 40, and 20 Random items to give a sense of the whole period
            df_sampled = pd.concat([
                df.head(40),
                df.sample(n=min(20, len(df)-80)) if len(df) > 80 else pd.DataFrame(),
                df.tail(40)
            ]).drop_duplicates().sort_index()
            df_to_use = df_sampled
            truncated_note = f"\n[NOTE: Showing {len(df_to_use)} of {len(df)} trades only to stay within context limits]\n"
        else:
            df_to_use = df
            truncated_note = ""

        # Simple header
        trades_context = f"""
    ================================================================================
    RAW TRADES DATASET ({len(df)} trades total)
    {truncated_note}
    ================================================================================
    
    """

        try:
            # Convert DataFrame to JSON with proper handling of datetime and NaN values
            import numpy as np

            # Replace NaN with None for proper JSON serialization
            df_clean = df_to_use.replace({np.nan: None})

            # Convert to JSON - RAW data only
            all_trades_json = df_clean.to_json(orient='records', date_format='iso', indent=2)
            trades_context += all_trades_json

        except Exception as e:
            # Fallback: convert to dict if JSON serialization fails
            self.logger.warning(f"JSON serialization failed, using dict format: {e}")
            trades_context += json.dumps(df_to_use.to_dict('records'), indent=2, default=str)

        trades_context += "\n\n================================================================================"

        return trades_context

    # =========================================================
    # Ollama API Interaction
    # =========================================================
    def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """Call Ollama or OpenRouter based on config switch"""
        try:
            use_ollama = self.config.get("llm_provider", {}).get("use_ollama", 1)
            print(use_ollama)
            # ======================================================
            # OPTION 1: OLLAMA (existing behavior – untouched logic)
            # ======================================================
            if use_ollama == 1:
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
                    timeout=1200
                )

                if response.status_code == 200:
                    return response.json().get("response", "")
                else:
                    self.logger.error(f"Ollama API error: {response.text}")
                    return "Error generating analysis"

            # ======================================================
            # OPTION 2: OPENROUTER
            # ======================================================
            # ======================================================
            # OPENROUTER PATH (Python equivalent of your JS code)
            # ======================================================
            headers = {
                "Authorization": f"Bearer {self.config['openrouter']['api_key']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Trade Persona Analyzer"
            }

            payload = {
                "model": self.config["openrouter"]["model"],  # e.g. openai/gpt-oss-20b
                "messages": [
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.config["openrouter"]["temperature"],
                "top_p": self.config["openrouter"]["top_p"],
                "stream": False
            }

            payload["messages"] = [m for m in payload["messages"] if m]

            response = requests.post(
                self.config["openrouter"]["base_url"],
                headers=headers,
                json=payload,
                timeout=1200
            )

            if response.status_code == 200:
                data = response.json()

                # Optional: log reasoning tokens (JS equivalent)
                usage = data.get("usage", {})
                if usage:
                    self.logger.info(
                        f"OpenRouter usage | prompt={usage.get('prompt_tokens')} "
                        f"completion={usage.get('completion_tokens')} "
                        f"reasoning={usage.get('reasoning_tokens')}"
                    )

                return data["choices"][0]["message"]["content"]

            else:
                self.logger.error(f"OpenRouter API error: {response.text}")
                return "Error generating analysis"

        except Exception as e:
            self.logger.error(f"Error calling LLM provider: {str(e)}")
        return "Error generating analysis - LLM provider unavailable"

    # =========================================================
    # Analysis Sections
    # =========================================================
    def _analyze_trader_profile(self, context: str) -> str:
        """Analyze trader profile with evidence-based classification"""
        # Extract metrics for prompt enhancement
        metrics = {}
        try:
            # Parse context to extract key metrics (basic extraction)
            import re
            metrics['discipline_score'] = re.search(r'Discipline Score: ([\d.]+)', context)
            metrics['emotional_control'] = re.search(r'Emotional Control: ([\d.]+)', context)
            metrics['risk_appetite'] = re.search(r'Risk Appetite: ([\d.]+)', context)
            
            discipline = float(metrics['discipline_score'].group(1)) if metrics['discipline_score'] else 0
            emotional = float(metrics['emotional_control'].group(1)) if metrics['emotional_control'] else 0
            risk_app = float(metrics['risk_appetite'].group(1)) if metrics['risk_appetite'] else 0
        except:
            discipline = emotional = risk_app = 0
        
        prompt = f"""
You are analyzing a trader's profile based on comprehensive trading data.

{context}

TASK: Classify this trader's profile with EVIDENCE-BASED reasoning.

CLASSIFICATION CRITERIA:
- **Scalper**: Avg holding < 30 min, 20+ trades/day
- **Day Trader**: Avg holding 30min-6hrs, 5-20 trades/day
- **Swing Trader**: Avg holding 1-5 days, 1-5 trades/day
- **Position Trader**: Avg holding > 5 days, < 1 trade/day

RISK APPETITE CRITERIA:
- **Conservative**: Max drawdown < 10%, position size < 5% capital
- **Moderate**: Max drawdown 10-20%, position size 5-10% capital
- **Aggressive**: Max drawdown > 20%, position size > 10% capital

OUTPUT FORMAT:
1. **Primary Classification**: [Type] with [Risk Appetite]
   - Evidence: [Cite specific metrics from the data]
   - Confidence: [High/Medium/Low]

2. **Persona Trait Alignment**:
   - Discipline Score ({discipline:.1f}/100): [How it manifests in trades]
   - Emotional Control ({emotional:.1f}/100): [Specific examples from patterns]
   - Risk Appetite ({risk_app:.1f}/100): [Position sizing patterns]

3. **Trading Style Characteristics** (3-5 bullet points with numbers):
   - Example: "Prefers options trading (78% of trades) over equity (22%)"

4. **Behavioral Signature** (1-2 sentences):
   - What makes this trader unique based on the data?

CONSTRAINTS:
- Use simple English for Indian retail traders
- Cite specific numbers from the data
- Total length: 250-350 words
- Be direct and honest

Provide the trader profile:
"""
        system_prompt = "You are an expert financial analyst specializing in trading behavior analysis. Provide evidence-based classifications with specific metrics."
        self.logger.info("1.1 Generate _analyze_trader_profile from LLM")
        raw = self._call_ollama(prompt, system_prompt)
        return self._beautify_recommendations_html(raw)

    def _analyze_risk(self, context: str) -> str:
        """Analyze risk profile with quantified assessment"""
        # Extract metrics for prompt enhancement
        try:
            import re
            sharpe_match = re.search(r'Sharpe Ratio: ([-\d.]+)', context)
            sortino_match = re.search(r'Sortino Ratio: ([-\d.]+)', context)
            dd_match = re.search(r'Max Drawdown: ([\d.]+)%', context)
            
            sharpe = float(sharpe_match.group(1)) if sharpe_match else 0
            sortino = float(sortino_match.group(1)) if sortino_match else 0
            max_dd = float(dd_match.group(1)) if dd_match else 0
        except:
            sharpe = sortino = max_dd = 0
        
        prompt = f"""
You are conducting a risk assessment for a trader.

{context}

TASK: Provide a QUANTIFIED risk assessment with actionable insights.

RISK LEVEL CRITERIA:
- **LOW**: Max DD < 10%, Sharpe > 1.5, Win Rate > 60%, No emotional patterns
- **MEDIUM**: Max DD 10-20%, Sharpe 0.5-1.5, Win Rate 50-60%, Minor emotional patterns
- **HIGH**: Max DD 20-35%, Sharpe 0-0.5, Win Rate 40-50%, Moderate emotional patterns
- **VERY HIGH**: Max DD > 35%, Sharpe < 0, Win Rate < 40%, Severe emotional patterns

OUTPUT FORMAT:

1. **Overall Risk Level**: [LOW/MEDIUM/HIGH/VERY HIGH]
   - Justification: [Which criteria triggered this level?]
   - Current Metrics: Sharpe={sharpe:.2f}, Sortino={sortino:.2f}, Max DD={max_dd:.1f}%

2. **Top 3 Risk Factors** (ranked by severity):
   🔴 **CRITICAL**: [Factor] - [Impact in ₹ or %]
   🟡 **HIGH**: [Factor] - [Impact in ₹ or %]
   🟢 **MEDIUM**: [Factor] - [Impact in ₹ or %]

3. **Risk-Adjusted Performance Analysis**:
   - Sharpe Ratio interpretation: [What does {sharpe:.2f} mean for this trader?]
   - Sortino Ratio interpretation: [What does {sortino:.2f} mean?]
   - Max Drawdown impact: [How long to recover from {max_dd:.1f}% loss?]

4. **Persona-Based Vulnerabilities**:
   - Link emotional control and discipline scores to specific risks
   - Cite examples from detected patterns

5. **Risk Mitigation Priority**:
   - **Do First**: [Most urgent action with expected risk reduction]
   - **Do Next**: [Second priority]

CONSTRAINTS:
- Use simple English for Indian retail traders
- Quantify every risk (₹ or %)
- Total length: 250-350 words
- Be direct about vulnerabilities

Provide the risk assessment:
"""
        system_prompt = "You are a risk management expert analyzing trading portfolios. Focus on quantified risk metrics and actionable mitigation strategies."
        self.logger.info("1.2 Generate _analyze_risk from LLM")
        return self._call_ollama(prompt, system_prompt)

    def _analyze_behavior(self, context: str) -> str:
        """Analyze behavioral patterns with quantified impact"""
        prompt = f"""
You are analyzing trading behavior patterns.

{context}

TASK: Identify behavioral patterns with FREQUENCY, SEVERITY, and P&L IMPACT.

OUTPUT FORMAT:

1. **Detected Negative Patterns** (for each pattern detected in the data):
   - **Pattern Name**: [e.g., Revenge Trading]
   - **Frequency**: [X times out of Y trades = Z%]
   - **P&L Impact**: [Estimated ₹ lost due to this pattern]
   - **Severity**: [Low/Medium/High/Critical]
   - **Example Trade**: [Cite specific date, symbol, loss amount from data]
   - **Persona Link**: [Which trait score explains this? Cite the score]

2. **Detected Positive Patterns** (if any):
   - **Pattern Name**: [e.g., Disciplined Exits]
   - **Frequency**: [X times out of Y trades = Z%]
   - **P&L Impact**: [Estimated ₹ gained due to this pattern]
   - **Example Trade**: [Cite specific date, symbol, profit amount]

3. **Contextual Trading Performance** (NEW):
   - **Market Trend Alignment**: [Are they fighting the Nifty trend or riding it? Cite 'Trend Alignment Score']
   - **Event/News Efficiency**: [Do they profit from volatility (events/news) or get trapped? Cite Win Rates for Event/News]
   - **Volume Handling**: [Do they size up correctly on high volume days? Cite High Volume PnL]

4. **Psychological Profile**:
   - **Primary Tendency**: [Based on most frequent pattern]
   - **Emotional Triggers**: [What causes bad trades? Cite examples from data]
   - **Discipline Breakdown Points**: [When does discipline fail? Cite times/situations]

5. **Consistency Analysis**:
   - Link consistency score to actual trading behavior
   - Evidence from win/loss streaks in the data

CONSTRAINTS:
- Every claim must have a number (frequency, amount, percentage)
- Cite at least 2 specific trade examples with dates
- Link patterns to persona trait scores
- Use simple English for Indian retail traders
- Total length: 300-400 words

Provide the behavioral analysis:
"""
        system_prompt = "You are a trading psychology expert analyzing trader behavior. Focus on data-driven insights with specific examples."
        self.logger.info("1.3 Generate _analyze_behavior from LLM")
        return self._call_ollama(prompt, system_prompt)

    def _analyze_context_performance(self, context: str) -> Dict:
        # """Analyze contextual performance (Events, News, Trend, Volume, Charts, ATH/ATL) returning STRUCTURED JSON"""
        #
        # # --- CRITICAL FIX: STRIP RAW TRADES FROM CONTEXT TO AVOID OVERLOAD ---
        # short_context = context
        # if "RAW TRADES DATASET" in context:
        #     short_context = context.split("RAW TRADES DATASET")[0]
        #
        prompt = f"""
        You are a high-performance trading psychologist analyzing a trader's behavior in specific contexts.

        {context}

        TASK: Interpret the 'Contextual Trading Performance' metrics.
        
        CRITICAL INSTRUCTIONS:
        1. **Do NOT just repeat the numbers** (The user can see the Win Rate/PnL). 
        2. **Explain the BEHAVIOR**: Why are they winning/losing in this context?
        3. **Map to PERSONA**: Link the observation to the identified 'Persona Type' and 'Traits'.

        Return a STRICT JSON object with no markdown formatting.

        JSON STRUCTURE:
        {{
            "event": {{
                "verdict": "Psychological/Behavioral Title (2-3 words)",
                "reasoning": "Insight linking to Persona traits."
            }},
            "news": {{
                "verdict": "Verdict Title",
                "reasoning": "Insight linking to Persona."
            }},
            "volume": {{
                "verdict": "Verdict Title",
                "reasoning": "Insight linking to Persona."
            }},
            "trend": {{
                "verdict": "Verdict Title",
                "reasoning": "Insight linking to Persona."
            }},
            "chart_quality": {{
                "verdict": "Verdict Title (e.g. 'Setup Discipline')",
                "reasoning": "Analyze performance on Good vs Bad charts. Do they respect technical setups?"
            }},
            "ath": {{
                "verdict": "Verdict Title (e.g. 'Breakout Confidence')",
                "reasoning": "Analyze behavior at All-Time Highs (fear of heights or momentum capture?)."
            }},
            "atl": {{
                "verdict": "Verdict Title (e.g. 'Value Hunting')",
                "reasoning": "Analyze behavior at All-Time Lows (catching falling knives or value signs?)."
            }}
        }}

        JSON OUTPUT:
        """
        system_prompt = "You are a data-driven trading analyst. Output only valid JSON."
        self.logger.info("1.3.5 Generate _analyze_context_performance schema from LLM")
        
        raw_response = self._call_ollama(prompt, system_prompt)
        
        # Parse JSON
        import json
        import re
        try:
            # Clean potential markdown code blocks
            clean_json = re.sub(r"```json|```", "", raw_response).strip()
            # Sanitization
            clean_json = clean_json.replace("‑", "-").replace("“", '"').replace("”", '"')
            
            return json.loads(clean_json)
        except Exception as e:
            self.logger.error(f"Failed to parse context JSON: {e}")
            # Fallback structure
            return {
                "event": {"verdict": "Analysis Failed", "reasoning": "Could not generate insight."},
                "news": {"verdict": "Analysis Failed", "reasoning": "Could not generate insight."},
                "volume": {"verdict": "Analysis Failed", "reasoning": "Could not generate insight."},
                "trend": {"verdict": "Analysis Failed", "reasoning": "Could not generate insight."},
                "chart_quality": {"verdict": "Analysis Failed", "reasoning": "Could not generate insight."},
                "ath": {"verdict": "Analysis Failed", "reasoning": "Could not generate insight."},
                "atl": {"verdict": "Analysis Failed", "reasoning": "Could not generate insight."}
            }

    def _generate_recommendations(self, context: str) -> str:
        """Generate actionable recommendations with numerical targets"""
        prompt = f"""
You are a professional trading coach creating an ACTION PLAN.

{context}

TASK: Provide SPECIFIC, MEASURABLE, ACHIEVABLE recommendations.

CRITICAL RULES:
1. Every recommendation MUST have a NUMBER (from X to Y)
2. Every recommendation MUST have a SUCCESS METRIC
3. Prioritize by IMPACT (which will improve P&L most?)
4. Personalize based on PERSONA TRAITS

OUTPUT FORMAT:

## 🔴 CRITICAL ACTIONS (Do in Next 1-2 Weeks)
**Priority 1: [Action Title]**
- **Current State**: [Metric] = [Current Value]
- **Target**: [Metric] = [Target Value]
- **Why**: [Expected P&L impact or risk reduction in ₹ or %]
- **How**: [Specific steps: 1, 2, 3]
- **Success Metric**: [How to measure progress]
- **Persona Link**: [Which trait needs improvement? Current score: X/100]

**Priority 2: [Action Title]**
[Same format]

## 🟡 IMPORTANT ACTIONS (Do in Next 1-3 Months)
**Action 1: [Title]**
- **Current → Target**: [X → Y]
- **Expected Impact**: [₹ or % improvement]
- **Steps**: [1, 2, 3]

**Action 2: [Title]**
[Same format]

## 🟢 LONG-TERM STRATEGY (3-6 Months)
**Strategic Change 1: [Title]**
- **Current Approach**: [Description]
- **Recommended Approach**: [Description]
- **Why**: [Rationale based on persona analysis]

## 📊 PERSONA DEVELOPMENT PLAN
Based on persona scores from the data:
- **Discipline**: [Specific exercise to improve this trait]
- **Emotional Control**: [Specific exercise to improve this trait]
- **Patience**: [Specific exercise to improve this trait]

## ✅ QUICK WINS (Easiest to Implement)
1. [Action with immediate impact - cite expected ₹ benefit]
2. [Action with immediate impact - cite expected ₹ benefit]

CONSTRAINTS:
- EVERY recommendation must have "From X to Y" format
- Prioritize by P&L impact (cite expected ₹ improvement)
- Use simple English for Indian retail traders
- Total length: 400-500 words
- Be brutally honest about what needs to change

Provide the action plan:
"""
        system_prompt = "You are a professional trading coach providing improvement strategies. Focus on specific, measurable actions with clear numerical targets."
        self.logger.info("1.4 Generate _generate_recommendations from LLM")
        return self._call_ollama(prompt, system_prompt)

    def _summarize_performance(self, context: str) -> str:
        """Summarize overall performance with improvement path"""
        prompt = f"""
You are providing an executive performance review.

{context}

TASK: Deliver a DIRECT, HONEST verdict with clear improvement path.

VERDICT CRITERIA:
- **EXCELLENT**: Sharpe > 1.5, Win Rate > 60%, Max DD < 10%, No critical patterns
- **GOOD**: Sharpe 1-1.5, Win Rate 55-60%, Max DD 10-15%, Minor patterns
- **AVERAGE**: Sharpe 0.5-1, Win Rate 50-55%, Max DD 15-25%, Moderate patterns
- **POOR**: Sharpe 0-0.5, Win Rate 40-50%, Max DD 25-35%, Severe patterns
- **CRITICAL**: Sharpe < 0, Win Rate < 40%, Max DD > 35%, Critical patterns

OUTPUT FORMAT:

## 📊 OVERALL VERDICT: [EXCELLENT/GOOD/AVERAGE/POOR/CRITICAL]

**Justification**: [Which criteria triggered this verdict? Cite specific metrics]

**Hard Flags Consideration**: [If any hard flags exist in the data, explain how they influenced the verdict]

## ✅ TOP 3 STRENGTHS
1. **[Strength]**: [Quantified evidence from data]
2. **[Strength]**: [Quantified evidence from data]
3. **[Strength]**: [Quantified evidence from data]

## ❌ TOP 3 WEAKNESSES
1. **[Weakness]**: [Quantified impact on P&L in ₹ or %]
2. **[Weakness]**: [Quantified impact on P&L in ₹ or %]
3. **[Weakness]**: [Quantified impact on P&L in ₹ or %]

## 🎯 PATH TO NEXT LEVEL
**To move from [current verdict] to [next level], you need to:**
1. [Specific metric improvement: From X to Y]
2. [Specific metric improvement: From X to Y]
3. [Specific behavioral change with measurable target]

## 🧠 PERSONA ASSESSMENT
**Your Trading Personality**: [Based on persona type from data]
- **Alignment**: [Are your traits helping or hurting performance?]
- **Key Insight**: [1-2 sentences on persona-performance relationship]

## 💰 BOTTOM LINE
[2-3 sentences: Direct, honest assessment. Would you invest with this trader? Why or why not?]

CONSTRAINTS:
- Be brutally honest
- Use simple English for Indian retail traders
- Cite specific numbers from the data
- Total length: 200-300 words

Provide the executive summary:
"""
        system_prompt = "You are a senior financial advisor providing performance reviews. Be direct and honest in your assessment."
        self.logger.info("1.5 Generate _summarize_performance from LLM")

        return self._call_ollama(prompt, system_prompt)

    # =========================================================
    # NEW: Structured Summary Extraction
    # =========================================================
    def _extract_structured_summary(self, sections: Dict, metrics: Dict, patterns: Dict) -> Dict:
        """Extract structured summary (risk level, verdict, key strengths/weaknesses)"""

        # Only join string sections to avoid TypeError with dicts (like context_performance)
        text_parts = [v for k, v in sections.items() if isinstance(v, str)]
        text = " ".join(text_parts)

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
        ceiling = self._compute_verdict_ceiling(metrics.get("_hard_flags", {}))

        order = ["CRITICAL", "POOR", "AVERAGE", "GOOD", "EXCELLENT"]
        if summary["performance_verdict"] in order:
            if order.index(summary["performance_verdict"]) > order.index(ceiling):
                summary["performance_verdict"] = ceiling

        return summary

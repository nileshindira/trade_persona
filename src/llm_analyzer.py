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


class LLMAnalyzer:
    """LLM-based analysis using multiple providers (Ollama, Claude, OpenRouter)"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

        self.ollama_base_url = config.get("ollama", {}).get("base_url", "http://localhost:11434")
        self.ollama_model = config.get("ollama", {}).get("model", "llama3.1")

        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")

        self.openrouter_model = config.get("openrouter", {}).get("model", "openai/gpt-3.5-turbo")

        if self.llm_provider == "claude":
            self.model = self.anthropic_model
        elif self.llm_provider == "openrouter":
            self.model = self.openrouter_model
        else:
            self.model = self.ollama_model
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
            'emotional_leakage_index': metrics.get('emotional_leakage_index', 0),
            'behavioral_severities': metrics.get('behavioral_severities', {}),

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
            # 7. Additional Financial Snapshot Metrics (Requested)
            'overall_win_rate': metrics.get('overall_win_rate', 0),
            'avg_win_pct_of_all_wins': metrics.get('avg_win_pct_of_all_wins', 0),
            'avg_loss_pct_of_all_losses': metrics.get('avg_loss_pct_of_all_losses', 0),
            'consecutive_wins': metrics.get('consecutive_wins', 0),
            'consecutive_losses': metrics.get('consecutive_losses', 0),
            'avg_win': metrics.get('avg_win', 0),
            'avg_loss': metrics.get('avg_loss', 0),
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
        chart_data['pnl_timeline'] = {
            'dates': metrics.get('pnl_timeline', {}).get('dates', []),
            'values': metrics.get('pnl_timeline', {}).get('values', []),
            'benchmark_values': metrics.get('pnl_timeline', {}).get('benchmark_values', []),
            'benchmark_original_values': metrics.get('pnl_timeline', {}).get('benchmark_original_values', [])
        }

        # Day-by-Day MTM (For bar chart)
        # Assuming 'day_mtm' is a list of daily PnL values
        chart_data['day_mtm'] = metrics.get('day_mtm', [])

        # Instrument Clustering (For pie chart)
        # Format as list of {'name': 'Instrument', 'value': 12.3}
        chart_data['instrument_distribution'] = metrics.get("chart_data", {}).get("asset_clusters", [])
        chart_data['segment_distribution'] = metrics.get("symbol_cluster", [])
        chart_data['symbol_distribution_raw'] = metrics.get("symbol_cluster", [])
        chart_data['industry_distribution'] = metrics.get("industry_distribution", [])
        chart_data['sectoral_analysis'] = metrics.get("sectoral_analysis", [])
        chart_data['option_strategy'] = metrics.get("option_strategy", {})

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
        required_cols = [
            'symbol', 'trade_date', 'transaction_type', 'quantity',
            'price', 'trade_value', 't_score', 'f_score', 'total_score',
            'is_52week_high', 'is_52week_low', 'is_alltime_high',
            'is_alltime_low', 'is_event', 'atr', 'is_high_volume',
            'market_behaviour', 'chart_charts'
        ]
        available_cols = [c for c in required_cols if c in df.columns]
        df_small = df[available_cols].copy()

        # Prepare context (NOW includes DataFrame)
        context = self._prepare_context(metrics, patterns, df_small)
        Path("chatgpt_context.json").write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
        Path("df_context.json").write_text(
            json.dumps(df.to_dict(orient="records"), indent=2, default=str),
            encoding="utf-8"
        )
        self.logger.info("1. Generate Stockk Persona Analysis from LLM")
        master_analysis = self._analyze_master_persona(metrics, patterns, context)

        # Backward compatibility for existing report structure
        analysis_text = {
            'trader_profile': master_analysis.get("identity", {}).get("narrative", master_analysis.get("fallback_narrative", "Analysis Pending")),
            'risk_assessment': master_analysis.get("identity", {}).get("risk_profile", {}).get("handling", "Analysis Pending"),
            'behavioral_insights': self._format_json_to_markdown_sections(master_analysis),
            'context_performance': self._build_context_performance(metrics),
            'simulated_transformation': master_analysis.get("simulated_transformation", []),
            'recommendations': self._format_recommendations_from_json(master_analysis.get("improvement_plan", {})),
            'performance_summary': master_analysis.get("final_verdict", "Analysis Pending"),
            'efficiency_metrics': {
                'entry_efficiency': metrics.get('avg_entry_efficiency', 0),
                'exit_efficiency': metrics.get('avg_exit_efficiency', 0),
                'profit_giveback': metrics.get('avg_profit_giveback', 0),
                'reward_to_risk': metrics.get('reward_to_risk_balance', 0),
                'consistency_score': metrics.get('behavioral_consistency_score', 0)
            },
            'market_context_metrics': {
                'avg_daily_range': metrics.get('avg_daily_range', 0),
                'volatility_index': metrics.get('volatility_index', 0),
                'volume_volatility': metrics.get('volume_volatility', 0)
            }
        }

        self.logger.info("2. Extract structured LLM summary")
        structured_summary = self._extract_structured_summary(analysis_text, metrics, patterns)
        
        # Override fields with Master Analysis data where appropriate
        structured_summary["trader_name"] = master_analysis.get("identity", {}).get("trader_name", "N/A")
        structured_summary["risk_appetite"] = master_analysis.get("identity", {}).get("risk_profile", {}).get("appetite", "N/A")
        structured_summary["risk_handling"] = master_analysis.get("identity", {}).get("risk_profile", {}).get("handling", "N/A")

        self.logger.info("3. NEW: Prepare dedicated data structures for the Web Page UI")
        web_kpis = self._prepare_dashboard_kpis(metrics, patterns)
        web_charts = self._prepare_chart_data(metrics,patterns, df)

        self.logger.info("Final Output for API/Web Service")
        analysis = {
            "analysis_text": analysis_text,
            "summary_data": structured_summary,
            "master_persona": master_analysis, # Pass through the full rich JSON
            "web_data": {
                "kpis": web_kpis,
                "charts": web_charts,
                "persona_scores": metrics.get("persona_traits", {}),
                "hard_flags": metrics.get("_hard_flags", {}),
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
            if isinstance(obj, (np.bool_, bool)):
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
        max_trades = 10000
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
    # LLM Provider Dispatcher
    # =========================================================
    def _call_llm(self, prompt: str, system_prompt: str = "") -> str:
        provider = self.llm_provider
        self.logger.info(f"Using LLM Provider: {provider} | model={self.model}")

        try:
            if provider == "claude":
                return self._call_claude(prompt, system_prompt)
            elif provider == "openrouter":
                return self._call_openrouter(prompt, system_prompt)
            else:
                return self._call_ollama(prompt, system_prompt)

        except Exception as e:
            self.logger.exception("Primary LLM provider failed: %s", e)

            fallback_enabled = os.getenv("LLM_FALLBACK_TO_OLLAMA", "true").lower() == "true"
            if provider != "ollama" and fallback_enabled:
                self.logger.warning("Falling back to Ollama")
                return self._call_ollama(prompt, system_prompt)

            raise
    def _call_claude(self, prompt: str, system_prompt: str = "") -> str:
        """Call Anthropic Claude API directly"""
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.anthropic_model,  # e.g. claude-opus-4-6
            "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "8192")),
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            self.logger.info(f"Calling Anthropic model: {self.anthropic_model}")
            response = requests.post(url, headers=headers, json=payload, timeout=600)

            if response.status_code != 200:
                self.logger.error("Anthropic API error %s: %s", response.status_code, response.text)
                raise RuntimeError(f"Anthropic API error {response.status_code}: {response.text}")

            data = response.json()

            stop_reason = data.get("stop_reason")
            self.logger.info("Anthropic stop_reason=%s", stop_reason)



            # Claude returns a content list; gather text blocks safely
            text_parts = []
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            result = "\n".join(part for part in text_parts if part).strip()
            if not result:
                raise RuntimeError("Anthropic returned no text content")

            usage = data.get("usage", {})
            self.logger.info(
                "Anthropic usage | input_tokens=%s output_tokens=%s",
                usage.get("input_tokens"),
                usage.get("output_tokens"),
            )

            return result

        except requests.Timeout as e:
            self.logger.error("Anthropic timeout: %s", e)
            raise RuntimeError("Anthropic request timed out") from e
        except requests.RequestException as e:
            self.logger.error("Anthropic request failed: %s", e)
            raise RuntimeError("Anthropic request failed") from e

    def _call_openrouter(self, prompt: str, system_prompt: str = "") -> str:
        """Call OpenRouter API"""
        try:
            api_key = self.config.get('openrouter', {}).get('api_key') or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OpenRouter API key not configured")


            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Trade Persona Analyzer"
            }

            payload = {
                "model": self.config["openrouter"].get("model", "openai/gpt-3.5-turbo"),
                "messages": [
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.config["openrouter"].get("temperature", 0.7),
                "top_p": self.config["openrouter"].get("top_p", 0.9),
                "stream": False
            }
            payload["messages"] = [m for m in payload["messages"] if m]

            response = requests.post(
                self.config["openrouter"].get("base_url", "https://openrouter.ai/api/v1/chat/completions"),
                headers=headers,
                json=payload,
                timeout=600
            )

            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                self.logger.info(
                    f"OpenRouter usage | prompt={usage.get('prompt_tokens')} "
                    f"completion={usage.get('completion_tokens')} "
                )
                return data["choices"][0]["message"]["content"]
            else:
                self.logger.error(f"OpenRouter API error: {response.text}")
                raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")
        except requests.Timeout as e:
            self.logger.error(f"OpenRouter timeout: {e}")
            raise RuntimeError("OpenRouter request timed out") from e
        except requests.RequestException as e:
            self.logger.error(f"OpenRouter request failed: {e}")
            raise RuntimeError("OpenRouter request failed") from e


    def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """Call Ollama API directly (moved logic from former _call_ollama)"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.get('ollama', {}).get('temperature', 0.7),
                        "top_p": self.config.get('ollama', {}).get('top_p', 0.9)
                    }
                },
                timeout=1200
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                self.logger.error(f"Ollama API error: {response.text}")
                raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")

        except Exception as e:
            self.logger.error(f"Error calling Ollama: {str(e)}")
            raise RuntimeError("Ollama provider unavailable") from e

    # =========================================================
    # ANTIGRAVITY MASTER PROMPT
    # =========================================================
    ANTIGRAVITY_MASTER_PROMPT = """
*** CRITICAL INSTRUCTION: YOUR ENTIRE RESPONSE MUST BE A SINGLE VALID JSON OBJECT. ***
*** DO NOT write any text, markdown code fences (```json), or commentary. Only the JSON starting with {{ and ending with }}. ***
*** IF YOU NEED TO EXPLAIN SOMETHING, DO IT INSIDE THE "narrative" OR "improvement_plan" FIELD AS MARKDOWN. ***

You are Antigravity, a surgery-grade Trading Psychologist and Quant Performance Coach.
Generate a deep-dive trader diagnosis report by answering these 3 major questions:

1. WHO IS THIS TRADER? (Diagnostic Identity & Signature)
   - Define a unique, memorable persona name.
   - Craft a beautiful "Diagnostic Verdict" narrative. Use rich markdown (bullets, bolding) to explain the trader's core psychology, strategy alignment, and behavioral fingerprint.
   - Contrast Risk Appetite vs. Risk Handling with specific scores and data-backed reasons.

2. WHERE ARE YOU GOOD? WHERE ARE YOU LOSING MONEY? (Edge vs. Leakage)
   - Strengths: List at least 3 distinct edges. Specify the proof/metric, why it works (causality), and how to leverage it further.
   - Mistakes: List at least 3 distinct leaks. Show the behavioral pattern, the metric evidence, the impact (High/Med/Low), and a "Surgical Fix".

3. DETAILED IMPROVEMENT ROADMAP (Transformation Strategy)
   - Incorporate the ACTUAL COMPUTED METRICS from 'what_if_analysis' to show the impact of behavioral change.
   - Provide a "Surgical Transformation Forecast" mapping specific adjustments (e.g., "Avoid Friday trading") to their PROJECTED BENEFIT (e.g., "₹X improvement, +Y% ROI").
   - Provide a prioritized action plan (Next 5 sessions, Next 30 days) and identify the "One Big Lever".

BEAUTY & PRESENTATION RULES:
- Use Markdown within the string values (e.g., * for bullets, ** for emphasis).
- Keep descriptions crisp, professional, and high-agency.
- DO NOT summarize; DIAGNOSE.

MANDATORY JSON SCHEMA:
{{
  "headline": "A one-line punchy surgical diagnosis.",
  "identity": {{
    "trader_name": "Unique persona name (e.g., The Precision Sniper)",
    "trader_type": "e.g., Scalper / Option Writer",
    "narrative": "A beautiful, structured markdown narrative answering 'Who Is This Trader?'.",
    "risk_profile": {{
      "appetite": "Score (1-10) and reason with data evidence.",
      "handling": "Score (1-10) and reason with data evidence."
    }}
  }},
  "strengths": [
    {{
      "title": "Edge Name",
      "evidence": "Data-backed proof",
      "why_it_works": "Causal explanation",
      "sectors": "Best performing sectors/instruments",
      "to_leverage": "Actionable advice"
    }}
  ],
  "mistakes": [
    {{
      "title": "Leak Name",
      "pattern": "Behavioral pattern",
      "evidence": "Data-backed proof",
      "impact": "High/Medium/Low",
      "correction": "Surgical Fix action"
    }}
  ],
  "simulated_transformation": [
    {{
      "adjustment": "Specific change based on what-if data",
      "projected_benefit": "Actual computed improvement (e.g., '₹X profit increase / +Y% ROI')",
      "rationale": "Why this specific change matters based on data"
    }}
  ],
  "improvement_plan": {{
    "next_5_sessions": ["Step 1", "Step 2", "Step 3"],
    "next_30_days": ["Process 1", "Process 2", "Process 3"],
    "biggest_lever": "The single most impactful move"
  }},
  "final_verdict": "One summary sentence of the trader's potential."
}}

--------------------------------------------------
INPUT DATA FOR ANALYSIS:
--------------------------------------------------
TRADING METRICS: {metrics}
BEHAVIORAL PATTERNS: {patterns}
WHAT-IF ANALYSIS: {what_if_analysis}
EVIDENCE PACKS: {evidence_packs}
RECENT MARKET CONTEXT (T+1/News/Options): {market_context}
FULL CONTEXT:
{context}
"""

    def _analyze_master_persona(self, metrics: Dict, patterns: Dict, context: str) -> Dict:
        """Call LLM with refined 3-question diagnostics prompt"""
        
        # Prepare context by stripping unnecessary deep nesting for the prompt
        include_keys = [
            "total_trades", "win_rate", "total_pnl", "avg_win", "avg_loss", 
            "profit_factor", "risk_appetite", "risk_handling", "trader_type",
            "timing_skill_score", "exit_quality_score", "pnl_reconciliation",
            "sectoral_analysis", "option_strategy", "strategy_inference",
            "what_if_analysis", "loss_patterns", "intraday_vs_overnight",
            "overall_win_rate", "avg_win_pct_of_all_wins", "avg_loss_pct_of_all_losses",
            "multivariate_pattern_analysis", "emotional_leakage_index", "behavioral_severities", "evidence_packs"
        ]
        prompt_metrics = {k: v for k, v in metrics.items() if k in include_keys}
        evidence_packs = metrics.get("evidence_packs", {})
        market_context = metrics.get("recent_market_context", {})
        what_if = metrics.get("what_if_analysis", {})

        full_prompt = self.ANTIGRAVITY_MASTER_PROMPT.format(
            metrics=json.dumps(prompt_metrics, default=str),
            patterns=json.dumps(patterns, default=str),
            what_if_analysis=json.dumps(what_if, default=str),
            evidence_packs=json.dumps(evidence_packs, default=str),
            market_context=json.dumps(market_context, default=str),
            context=context
        )
        system_prompt = (
            "You are Antigravity, a surgery-grade trading psychologist and performance coach. "
            "You MUST respond with ONLY a single valid JSON object. "
            "Do NOT use markdown code fences. Do NOT add any text before or after the JSON. "
            "Your response must start with { and end with }. "
            "Every field in the requested schema is mandatory - do not skip any."
        )
        raw = self._call_llm(full_prompt, system_prompt)
        
        # Comprehensive JSON Extraction Logic
        try:
            # 1. Strip Markdown fences if present
            clean_raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            
            # 2. Find JSON block using braces
            start_idx = clean_raw.find('{')
            end_idx = clean_raw.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                clean_json = clean_raw[start_idx:end_idx+1]
            else:
                clean_json = clean_raw

            # 3. Clean common LLM hallucinations in JSON
            for old, new in [("\u2011", "-"), ("\u201c", '"'), ("\u201d", '"'), ("\u2018", "'"), ("\u2019", "'")]:
                clean_json = clean_json.replace(old, new)
            
            # Remove trailing commas before closing braces/brackets
            clean_json = re.sub(r',\s*([\]}])', r'\1', clean_json)

            if not clean_json:
                raise ValueError("Extracted JSON string is empty")

            parsed = json.loads(clean_json)
            
            # Impactful check: if we got a list instead of a dict, wrap it or handle it
            if isinstance(parsed, list) and len(parsed) > 0:
                parsed = parsed[0] if isinstance(parsed[0], dict) else {"error": "Unexpected list format"}

            # Validate essential fields are present
            if not parsed.get("strengths") and not parsed.get("identity", {}).get("narrative"):
                 self.logger.warning("LLM returned JSON but essential fields are empty")
                 if len(raw) > 200:
                     return self._parse_markdown_fallback(raw)
            
            return parsed
            
        except Exception as e:
            self.logger.error(f"Master Persona Analysis JSON parse failed: {e}")
            snippet = raw[:500] + "..." if len(raw) > 500 else raw
            self.logger.error(f"Raw Response Snippet: {snippet}")
            
            # ROBUST FALLBACK: Parse markdown text into structured fields
            if len(raw) > 100:
                self.logger.info("Attempting markdown fallback parse...")
                return self._parse_markdown_fallback(raw)
            return {"error": "JSON parse failed", "fallback_narrative": raw}

    def _format_json_to_markdown_sections(self, master_json: Dict) -> str:
        """Convert master JSON components to Markdown for backward compatible Report UI"""
        if "error" in master_json: return "Analysis Failed: LLM response was not valid JSON."

        # If we have a fallback narrative (LLM returned text instead of JSON), use it directly
        if master_json.get("fallback_narrative"):
             return f"### LLM Analysis (Unstructured Outcome)\n\n{master_json['fallback_narrative']}"

        identity = master_json.get('identity', {})
        md = f"### {identity.get('trader_name', 'Trader Profile')}\n\n"
        
        md += "#### 🧠 Core Identity\n"
        md += f"- **Strategy Type**: {identity.get('strategy_type')}\n"
        md += f"- **Appetite vs Handling**: {identity.get('risk_profile', {}).get('appetite')} vs {identity.get('risk_profile', {}).get('handling')}\n"
        
        md += "\n#### 🔍 Execution Strengths\n"
        for s in master_json.get("strengths", []):
            md += f"- **{s.get('title')}**: {s.get('why_it_works')} (Evidence: {s.get('evidence')})\n"
            
        md += "\n#### 📉 Critical Mistakes\n"
        for m in master_json.get("mistakes", []):
            md += f"- **{m.get('title')}**: {m.get('pattern')}. Fix: {m.get('correction')}\n"
            
        return md

    def _format_recommendations_from_json(self, coaching: Dict) -> str:
        """Format the coaching plan into the HTML structure expected by the UI"""
        if not coaching: return "No coaching data."
        
        res = "## 🔴 IMMEDIATE ACTIONS (Next 5 Sessions)\n"
        for act in coaching.get("next_5_sessions", []):
            res += f"- {act}\n"
            
        res += "\n## 🟡 STRUCTURAL CHANGES (Next 30 Days)\n"
        for act in coaching.get("next_30_days", []):
            res += f"- {act}\n"
            
        res += "\n## 🟢 STRATEGIC GROWTH (Long-term)\n"
        for act in coaching.get("long_term", []):
            res += f"- {act}\n"
            
        return res

    # =========================================================
    # NEW: Build context_performance from computed metrics
    # =========================================================
    def _build_context_performance(self, metrics: Dict) -> Dict:
        """Build contextual performance verdicts from computed metrics data."""
        def _verdict(win_rate, count, pnl):
            if count == 0:
                return {"verdict": "No Data", "reasoning": "No trades detected in this category."}
            if win_rate >= 60 and pnl > 0:
                return {"verdict": "Strong Edge", "reasoning": f"{win_rate:.1f}% win rate across {count} trades with ₹{pnl:,.0f} net P&L. This is a profitable pattern."}
            elif win_rate >= 50 and pnl > 0:
                return {"verdict": "Modest Edge", "reasoning": f"{win_rate:.1f}% win rate across {count} trades with ₹{pnl:,.0f} net P&L. Slight positive edge."}
            elif pnl < 0:
                return {"verdict": "Leaking Capital", "reasoning": f"{win_rate:.1f}% win rate across {count} trades but losing ₹{abs(pnl):,.0f}. This pattern is a capital drain."}
            else:
                return {"verdict": "Neutral", "reasoning": f"{win_rate:.1f}% win rate across {count} trades. No clear edge or leak."}

        return {
            "event": _verdict(
                metrics.get("event_trading_win_rate", 0),
                metrics.get("event_trading_count", 0),
                metrics.get("event_trading_pnl", 0)
            ),
            "news": _verdict(
                metrics.get("news_trading_win_rate", 0),
                metrics.get("news_trading_count", 0),
                metrics.get("news_trading_pnl", 0)
            ),
            "volume": _verdict(
                metrics.get("high_volume_trading_win_rate", 0),
                metrics.get("high_volume_trading_count", 0),
                metrics.get("high_volume_trading_pnl", 0)
            ),
            "trend": _verdict(
                metrics.get("trend_aligned_win_rate", 0),
                metrics.get("total_trades", 0),  # trend applies to all trades
                metrics.get("total_pnl", 0)
            ),
            "chart_quality": _verdict(
                metrics.get("good_chart_trading_win_rate", 0),
                metrics.get("good_chart_trading_count", 0),
                metrics.get("good_chart_trading_pnl", 0)
            ),
            "ath": _verdict(
                metrics.get("ath_trading_win_rate", 0),
                metrics.get("ath_trading_count", 0),
                metrics.get("ath_trading_pnl", 0)
            ),
            "atl": _verdict(
                metrics.get("atl_trading_win_rate", 0),
                metrics.get("atl_trading_count", 0),
                metrics.get("atl_trading_pnl", 0)
            ),
        }

    # =========================================================
    # NEW: Markdown Fallback Parser
    # =========================================================
    def _parse_markdown_fallback(self, raw_text: str) -> Dict:
        """
        When LLM returns markdown text instead of JSON, parse it into 
        the expected structured format so the report is still populated.
        """
        self.logger.info("Parsing LLM markdown output into structured fields...")
        
        # Extract persona name from markdown headers or bold text
        persona_match = re.search(r'["\u201c](The [^"\u201d]+)["\u201d]', raw_text)
        if not persona_match:
            persona_match = re.search(r'\*\*(?:Trader Profile|Persona).*?["\u201c]?(The [^"\u201d*]+)["\u201d]?\*\*', raw_text)
        if not persona_match:
            persona_match = re.search(r'(?:Persona|Name|Profile)[:\s]+["\u201c]?(The [^"\u201d\n]+)["\u201d]?', raw_text, re.IGNORECASE)
        persona_name = persona_match.group(1).strip() if persona_match else "The Trader"
        
        # Extract strategy type
        strategy_match = re.search(r'(?:strategy|style|type)[:\s]+["\u201c]?([^"\u201d\n,.]+)["\u201d]?', raw_text, re.IGNORECASE)
        strategy_type = strategy_match.group(1).strip() if strategy_match else "Mixed Strategy"
        
        # Extract strengths from bullet points after strength-related headers
        strengths = []
        strength_section = re.search(r'(?:strength|good|edge|profit|where.*good).*?\n((?:[-*•]\s+.+\n?)+)', raw_text, re.IGNORECASE)
        if strength_section:
            bullets = re.findall(r'[-*•]\s+\*\*(.+?)\*\*[:\s]*(.+)', strength_section.group(1))
            for title, desc in bullets[:5]:
                strengths.append({
                    "title": title.strip(),
                    "evidence": desc.strip()[:200],
                    "why_it_works": desc.strip()[:200],
                    "sectors": "See evidence",
                    "to_leverage": "Scale this pattern during high-confidence setups."
                })
        
        # If no structured strengths found, extract from any bold items
        if not strengths:
            bold_items = re.findall(r'\*\*(.{5,60}?)\*\*[:\s]*(.{10,300}?)(?=\n|$)', raw_text)
            for title, desc in bold_items[:5]:
                if any(kw in title.lower() for kw in ['win', 'profit', 'strong', 'edge', 'reward', 'scalp', 'momentum']):
                    strengths.append({
                        "title": title.strip(),
                        "evidence": desc.strip()[:200],
                        "why_it_works": desc.strip()[:200],
                        "sectors": "See evidence",
                        "to_leverage": "Continue to refine this entry/exit trigger."
                    })
        
        # Ensure at least one strength
        if not strengths:
            strengths = [{
                "title": "Positive Expectancy System",
                "evidence": "Overall P&L is positive, indicating the core system has edge.",
                "why_it_works": "The trading strategy generates more profits than losses over time.",
                "sectors": "Primary trading instruments",
                "to_leverage": "Increase size gradually as confidence builds."
            }]
        
        # Extract mistakes from bullet points
        mistakes = []
        mistake_section = re.search(r'(?:mistake|bad|weakness|leak|loss|risk flag|where.*bad|where.*los).*?\n((?:[-*•]\s+.+\n?)+)', raw_text, re.IGNORECASE)
        if mistake_section:
            bullets = re.findall(r'[-*•]\s+\*\*(.+?)\*\*[:\s]*(.+)', mistake_section.group(1))
            for title, desc in bullets[:5]:
                mistakes.append({
                    "title": title.strip(),
                    "pattern": desc.strip()[:200],
                    "evidence": desc.strip()[:200],
                    "leakage_sectors": "See evidence",
                    "impact": "High",
                    "correction": "Review and address this pattern."
                })
        
        if not mistakes:
            bold_items = re.findall(r'\*\*(.{5,60}?)\*\*[:\s]*(.{10,300}?)(?=\n|$)', raw_text)
            for title, desc in bold_items[:5]:
                if any(kw in title.lower() for kw in ['loss', 'risk', 'over', 'revenge', 'fomo', 'leak', 'drawdown', 'weak']):
                    mistakes.append({
                        "title": title.strip(),
                        "pattern": desc.strip()[:200],
                        "evidence": desc.strip()[:200],
                        "leakage_sectors": "See evidence",
                        "impact": "Medium",
                        "correction": "Implement systematic rules to prevent this."
                    })
        
        if not mistakes:
            mistakes = [{
                "title": "Behavioral Pattern Under Review",
                "pattern": "LLM analysis detected patterns that need further investigation.",
                "evidence": "See full narrative in the diagnosis section.",
                "leakage_sectors": "To be identified",
                "impact": "Unknown",
                "correction": "Follow the improvement roadmap."
            }]
        
        # Extract improvement recommendations
        improvements = re.findall(r'[-*•]\s+(.{15,200}?)(?=\n|$)', raw_text)
        improvement_items = [item.strip().strip('*').strip() for item in improvements if len(item.strip()) > 15][:9]
        
        next_5 = improvement_items[:3] if len(improvement_items) >= 3 else improvement_items + ["Review trading journal daily"] * (3 - len(improvement_items))
        next_30 = improvement_items[3:6] if len(improvement_items) >= 6 else ["Build systematic rules for identified patterns", "Track emotional state during trading sessions", "Review and optimize position sizing"]
        long_term = improvement_items[6:9] if len(improvement_items) >= 9 else ["Develop a comprehensive trading plan", "Build consistency in execution", "Master risk management fundamentals"]
        
        # Extract a verdict-like sentence
        verdict_match = re.search(r'(?:verdict|conclusion|bottom line|summary)[:\s]*(.{20,200}?)(?=\n|$)', raw_text, re.IGNORECASE)
        final_verdict = verdict_match.group(1).strip() if verdict_match else f"{persona_name}: A trader with identifiable edge patterns but behavioral leaks that need systematic correction."
        
        # Build the biggest lever
        lever_match = re.search(r'(?:biggest|single|most important|key).*?(?:change|fix|improvement|lever)[:\s]*(.{20,200}?)(?=\n|$)', raw_text, re.IGNORECASE)
        biggest_lever = lever_match.group(1).strip() if lever_match else "Eliminate the #1 behavioral leak to unlock latent profitability."
        
        # Clean the narrative text for the display
        # 1. Remove markdown tables and separator lines
        clean_text = re.sub(r'\|.*\|', '', raw_text)
        clean_text = re.sub(r'[-]{3,}', '', clean_text)
        # 2. Remove markdown headers
        clean_text = re.sub(r'#{1,6}\s+.*', '', clean_text)
        # 3. Remove excessive newlines
        clean_text = re.sub(r'\n{2,}', '\n', clean_text).strip()
        
        # Extract a better narrative (first 5-7 meaningful sentences)
        sentences = re.split(r'(?<=[.!?])\s+', clean_text.replace('\n', ' '))
        narrative = ' '.join(s.strip() for s in sentences[:7] if len(s.strip()) > 25)
        if not narrative:
            narrative = clean_text[:600]
        
        # Try to extract strategy name from first bold items if strategy_type is weak
        if len(strategy_type) < 5 or "Analysis" in strategy_type:
            strat_found = re.search(r'\*\*(?:Strategy|Type|Profile)\*\*[:\s]+(.*?)(?=\n|$)', raw_text, re.IGNORECASE)
            if strat_found:
                strategy_type = strat_found.group(1).strip().strip('*')

        # Try to extract risk info from tables specifically
        appetite_val = "Requires detailed analysis — see narrative"
        handling_val = "Requires detailed analysis — see narrative"
        
        table_rows = re.findall(r'\|(.*?)\|(.*?)\|', raw_text)
        for col1, col2 in table_rows:
            c1, c2 = col1.strip().lower(), col2.strip()
            if 'risk' in c1 or 'capital' in c1:
                appetite_val = f"{col1.strip()}: {c2}"
            if 'overtrading' in c1 or 'handling' in c1 or 'instability' in c1:
                handling_val = f"{col1.strip()}: {c2}"
        
        # Try to extract natural edge
        edge_match = re.search(r'(?:natural edge|core edge|competitive advantage)[:\s]*(.{20,200}?)(?=\n|$)', raw_text, re.IGNORECASE)
        natural_edge = edge_match.group(1).strip() if edge_match else "Momentum capture during high-volume periods."

        result = {
            "headline": f"{persona_name} — Surgical Trading Diagnosis",
            "identity": {
                "trader_name": persona_name,
                "strategy_type": strategy_type,
                "primary_strategy": strategy_type,
                "natural_edge": natural_edge,
                "risk_profile": {
                    "appetite": "Requires detailed analysis — see narrative",
                    "handling": "Requires detailed analysis — see narrative"
                },
                "narrative": narrative
            },
            "strengths": strengths,
            "mistakes": mistakes,
            "simulated_transformation": [
                {"adjustment": "Eliminate revenge trading patterns", "projected_benefit": "Potential 10-15% P&L improvement"},
                {"adjustment": "Optimize position sizing on high-conviction setups", "projected_benefit": "Better risk-adjusted returns"},
                {"adjustment": "Implement strict stop-loss discipline", "projected_benefit": "Reduced drawdown by 20-30%"}
            ],
            "improvement_plan": {
                "next_5_sessions": next_5,
                "next_30_days": next_30,
                "long_term": long_term,
                "biggest_lever": biggest_lever
            },
            "final_verdict": final_verdict,
            "fallback_narrative": raw_text
        }
        
        # Try to enrich with risk flag info from the text
        risk_section = re.search(r'(?:Risk Flag|Risk Profile|HARD RISK).*?\n((?:[-*•]\s+.+\n?)+)', raw_text, re.IGNORECASE)
        if risk_section:
            risk_bullets = re.findall(r'[-*•]\s+\*\*(.+?)\*\*[:\s–-]+(.+?)(?=\n|$)', risk_section.group(0))
            if risk_bullets:
                appetite_text = next((desc for title, desc in risk_bullets if 'appetite' in title.lower() or 'capital' in title.lower()), None)
                handling_text = next((desc for title, desc in risk_bullets if 'handling' in title.lower() or 'emotion' in title.lower() or 'overtrading' in title.lower()), None)
                if appetite_text:
                    appetite_val = appetite_text.strip()[:150]
                if handling_text:
                    handling_val = handling_text.strip()[:150]
        
        result["identity"]["risk_profile"]["appetite"] = appetite_val
        result["identity"]["risk_profile"]["handling"] = handling_val
        result["identity"]["strategy_type"] = strategy_type
        result["identity"]["primary_strategy"] = strategy_type
        
        self.logger.info(f"Fallback parse extracted: {len(strengths)} strengths, {len(mistakes)} mistakes, persona='{persona_name}'")
        return result

    # =========================================================
    # Legacy Analysis Sections (now wrappers or deleted)
    # =========================================================
    def _analyze_trader_profile(self, context: str) -> str:
        return "Legacy Section - Replaced by Master Analysis"

    def _analyze_risk(self, context: str) -> str:
        return "Legacy Section - Replaced by Master Analysis"

    def _analyze_behavior(self, context: str) -> str:
        return "Legacy Section - Replaced by Master Analysis"

    def _analyze_context_performance(self, context: str) -> Dict:
        return {}

    def _generate_recommendations(self, context: str) -> str:
        return "Legacy Section - Replaced by Master Analysis"

    def _summarize_performance(self, context: str) -> str:
        return "Legacy Section - Replaced by Master Analysis"

    # NOTE: _beautify_recommendations_html defined earlier at line 143 with full
    # HTML beautification logic. Duplicate removed to prevent silent override.

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

    # NOTE: _compute_verdict_ceiling defined earlier at line 100. Duplicate removed.

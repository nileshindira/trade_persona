"""
LLM Analyzer Module
Integrates with Ollama for AI-powered analysis
"""
import os
import requests
import json
import hashlib
from typing import Dict, List, Optional, Any
import logging
import re
import pandas as pd
import numpy as np
from pathlib import Path
import datetime
from datetime import date
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

        # Setup caching for resumability
        self.cache_dir = Path("data/.cache/llm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = os.getenv("LLM_ENABLE_CACHE", "true").lower() == "true"
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
    # What-If Simulation Engine (Factual Metrics)
    # =========================================================
    def _compute_whatif_simulations(self, metrics: Dict, df: pd.DataFrame) -> Dict:
        """
        Compute factual what-if scenarios from real trade data.
        These are CALCULATED numbers, not LLM-generated — providing
        grounded evidence for improvement recommendations.
        """
        simulations = {}
        
        total_pnl = metrics.get("total_pnl", 0)
        avg_loss = abs(metrics.get("avg_loss", 0))
        avg_win = metrics.get("avg_win", 0)
        total_trades = metrics.get("total_trades", 1)
        win_rate = metrics.get("win_rate", 50)
        profit_factor = metrics.get("profit_factor", 1)
        
        # Identify losing trades from positions data
        positions = metrics.get("closed_positions", [])
        losing_trades = [p for p in positions if isinstance(p, dict) and (p.get("realized_pnl", 0) or 0) < 0]
        winning_trades = [p for p in positions if isinstance(p, dict) and (p.get("realized_pnl", 0) or 0) > 0]
        
        total_loss_amount = sum(abs(p.get("realized_pnl", 0)) for p in losing_trades) if losing_trades else avg_loss * (total_trades * (1 - win_rate/100))
        total_win_amount = sum(p.get("realized_pnl", 0) for p in winning_trades) if winning_trades else avg_win * (total_trades * win_rate/100)
        
        # Simulation 1: If avg loss reduced by 20%
        loss_reduction_pct = 20
        saved_amount = total_loss_amount * (loss_reduction_pct / 100)
        new_pnl_1 = total_pnl + saved_amount
        pnl_improvement_1 = (saved_amount / max(abs(total_pnl), 1)) * 100
        simulations["reduce_avg_loss"] = {
            "scenario": f"If average stop-loss was tightened by {loss_reduction_pct}%",
            "action": f"Tighten Stops by {loss_reduction_pct}%",
            "delta": round(saved_amount, 2),
            "impact_pct": round(pnl_improvement_1, 1),
            "metric_label": f"₹{saved_amount:,.0f} saved ({pnl_improvement_1:.1f}% P&L boost)"
        }
        
        # Simulation 2: If worst 5% of losers were eliminated 
        if losing_trades:
            sorted_losers = sorted(losing_trades, key=lambda p: p.get("realized_pnl", 0))
            bottom_5pct_count = max(1, int(len(sorted_losers) * 0.05))
            worst_losers = sorted_losers[:bottom_5pct_count]
            worst_loss_total = sum(abs(p.get("realized_pnl", 0)) for p in worst_losers)
            new_pnl_2 = total_pnl + worst_loss_total
            pnl_improvement_2 = (worst_loss_total / max(abs(total_pnl), 1)) * 100
            simulations["eliminate_worst_losers"] = {
                "scenario": f"If worst {bottom_5pct_count} trades (bottom 5% losers) were avoided",
                "action": f"Avoid {bottom_5pct_count} Outlier Losers",
                "delta": round(worst_loss_total, 2),
                "impact_pct": round(pnl_improvement_2, 1),
                "metric_label": f"Remove {bottom_5pct_count} worst trades → +₹{worst_loss_total:,.0f} ({pnl_improvement_2:.1f}% uplift)"
            }
        
        # Simulation 3: If win rate improved by 5 percentage points
        win_rate_boost = 5
        new_win_rate = min(100, win_rate + win_rate_boost)
        additional_wins = total_trades * (win_rate_boost / 100)
        total_sim3_gain = additional_wins * (avg_win + avg_loss)
        new_pnl_3 = total_pnl + total_sim3_gain
        pnl_improvement_3 = (total_sim3_gain / max(abs(total_pnl), 1)) * 100
        simulations["improve_win_rate"] = {
            "scenario": f"If win rate improved from {win_rate:.1f}% to {new_win_rate:.1f}%",
            "action": f"Boost Win Rate by {win_rate_boost}%",
            "delta": round(total_sim3_gain, 2),
            "impact_pct": round(pnl_improvement_3, 1),
            "metric_label": f"+{win_rate_boost}% win rate → +₹{total_sim3_gain:,.0f} ({pnl_improvement_3:.1f}% uplift)"
        }
        
        # Simulation 4: If position sizing was normalized (remove outlier sizes)
        if 'quantity' in df.columns:
            qty_series = df['quantity'].dropna().abs()
            if len(qty_series) > 10:
                q75 = qty_series.quantile(0.75)
                q25 = qty_series.quantile(0.25)
                iqr = q75 - q25
                oversized_mask = qty_series > (q75 + 1.5 * iqr)
                oversized_count = oversized_mask.sum()
                if oversized_count > 0:
                    median_qty = qty_series.median()
                    simulations["normalize_sizing"] = {
                        "scenario": f"If {int(oversized_count)} oversized positions were capped to median size ({median_qty:.0f})",
                        "action": "Normalize Position Sizing",
                        "delta": 0,
                        "impact_pct": 0,
                        "metric_label": f"Cap {int(oversized_count)} oversized trades to median ({median_qty:.0f}) for reduced volatility"
                    }
        
        # Simulation 5: Revenge trading impact
        revenge_pnl = metrics.get("behavioral_pressure_map", {}).get("revenge_trade_index", 0)
        if revenge_pnl and revenge_pnl < 0:
            estimated_revenge_count = max(1, int(total_trades * 0.08))  # estimate ~8% are revenge trades
            estimated_revenge_loss = abs(revenge_pnl) * estimated_revenge_count
            simulations["eliminate_revenge"] = {
                "scenario": f"If revenge trades after losses were eliminated (~{estimated_revenge_count} est. trades)",
                "action": "Eliminate Revenge Trading",
                "delta": round(estimated_revenge_loss, 2),
                "impact_pct": round((estimated_revenge_loss / max(abs(total_pnl), 1)) * 100, 1),
                "metric_label": f"No revenge trading → +₹{estimated_revenge_loss:,.0f} estimated recovery"
            }
        
        # Build compact summary for LLM context
        simulation_summary = "WHAT-IF SIMULATIONS (Computed from actual data):\n"
        for key, sim in simulations.items():
            simulation_summary += f"- {sim['scenario']}: {sim['metric_label']}\n"
        
        return {
            "simulations": simulations,
            "llm_context": simulation_summary
        }

    def _prepare_merged_trades(self, df: pd.DataFrame) -> list:
        """
        Serialize the full paired trades DataFrame for the merged trade table.
        Returns a list of dicts with all columns matching the Proposed_Trade_File schema.
        """
        if df is None or df.empty:
            return []
        
        # Select key columns available in the df, matching the Proposed schema
        proposed_cols = [
            'trade_date', 'symbol', 'transaction_type', 'quantity', 'price', 'trade_value',
            'open', 'high', 'low', 'close', 'volume',
            't_score', 'f_score', 'total_score',
            'is_52week_high', 'is_52week_low', 'is_alltime_high', 'is_alltime_low',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'atr', 'adx', 'cci', 'mfi',
            'bb_middle', 'bb_upper', 'bb_lower',
            'market_behaviour', 'is_news', 'is_event', 'is_high_volume',
            'news_sentiment', 'news_category',
            # Paired trade fields
            'realized_pnl', 'pnl',
        ]
        
        available_cols = [c for c in proposed_cols if c in df.columns]
        
        # Add any remaining columns not in the proposed list
        extra_cols = [c for c in df.columns if c not in available_cols and c not in ['__merged']]
        all_cols = available_cols + extra_cols
        
        try:
            import numpy as np
            df_export = df[all_cols].copy()
            # Clean NaN/NaT for JSON
            df_export = df_export.replace({np.nan: None, np.inf: None, -np.inf: None})
            # Convert timestamps
            for col in df_export.columns:
                if df_export[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                    df_export[col] = df_export[col].apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x) if x is not None else None)
            
            records = df_export.to_dict(orient='records')
            # Truncate to a reasonable amount for web view to prevent browser crashes
            # The full trace is exported as CSV separately
            return records[:500] 
        except Exception as e:
            self.logger.warning(f"Failed to prepare merged trades: {e}")
            return []

    # =========================================================
    # Main Integration Function
    # =========================================================
    def generate_analysis(self, metrics: Dict, patterns: Dict, df: pd.DataFrame, trader_name: str = "Trader") -> Dict:
        """Generate comprehensive analysis using LLM with 4-stage grounded flow."""
        self.trader_name = trader_name
        # Step 1: Orchestrate Stage Analysis (Chunks -> Aggregate -> Synthesis -> Roadmap)
        self.logger.info(f"Starting Grounded 4-Stage Diagnostic Flow for {trader_name}")
        master_analysis = self._analyze_master_persona(metrics, patterns, df)

        # 2. Extract values from the new FLAT schema with fallbacks
        # Schema: persona_name, persona_summary, core_identity, trader_type, execution_profile, risk_profile, etc.
        persona_name = master_analysis.get("persona_name") or \
                       master_analysis.get("identity", {}).get("persona_name") or \
                       master_analysis.get("identity", {}).get("trader_name") or \
                       "The Market Participant"
                       
        trader_type = master_analysis.get("trader_type", metrics.get("archetype", {}).get("name", "N/A"))
        headline = master_analysis.get("persona_summary", master_analysis.get("headline", "Trading performance analysis"))
        
        # Combine profiles into a cohesive narrative for the UI
        # Fallback to identity.narrative if core_identity is missing (case for fallback parser)
        raw_core = master_analysis.get('core_identity')
        if not raw_core:
            # Maybe the LLM returned it in 'identity' block (fallback parser does this)
            raw_core = master_analysis.get('identity', {}).get('narrative')
            
        narrative = f"### Core Identity\n{raw_core or 'Analysis Pending...'}\n\n"
        narrative += f"### Execution & Discipline\n{master_analysis.get('execution_profile', master_analysis.get('identity', {}).get('strategy_type', ''))}\n\n"
        narrative += f"### Consistency & Emotion\n{master_analysis.get('consistency_profile', '')}"

        # Map strengths and weaknesses (Mistakes)
        strengths = master_analysis.get("strengths", master_analysis.get("top_strengths", []))
        mistakes = master_analysis.get("mistakes", master_analysis.get("top_weaknesses", []))

        # Build high-fidelity improvement roadmap from the LLM
        # Priority: dedicated roadmap (Stage 4) -> flat roadmap (Stage 3) -> fallbacks
        dedicated_roadmap = master_analysis.get("improvement_roadmap_dedicated", {})
        
        biggest_lever_raw = dedicated_roadmap.get("one_move_to_change_everything") or \
                            master_analysis.get("one_move_to_change_everything") or \
                            "Consistency in execution."
        
        # Flatten object to string if LLM returned an object by mistake
        if isinstance(biggest_lever_raw, dict):
            biggest_lever_raw = biggest_lever_raw.get("action", biggest_lever_raw.get("text", str(biggest_lever_raw)))

        improvement_roadmap = {
            "biggest_lever": biggest_lever_raw,
            "next_5_sessions": dedicated_roadmap.get("roadmap_5_days") or 
                               master_analysis.get("roadmap_5_days") or [],
            "next_30_days": dedicated_roadmap.get("roadmap_30_days") or 
                            master_analysis.get("roadmap_30_days") or []
        }

        # Extra context for the "biggest lever" if rationale exists
        rationale = dedicated_roadmap.get("one_move_rationale")
        if rationale and improvement_roadmap["biggest_lever"]:
            improvement_roadmap["biggest_lever"] = f"**{improvement_roadmap['biggest_lever']}** | {rationale}"

        # Ensure roadmap has at least some entries (Fallbacks)
        if not improvement_roadmap["next_5_sessions"]:
            improvement_roadmap["next_5_sessions"] = [m.get("fix", "") for m in mistakes if isinstance(m, dict) and m.get("fix")][:3]
        if not improvement_roadmap["next_5_sessions"]:
            improvement_roadmap["next_5_sessions"] = ["Review trading journal daily", "Verify trend alignment on all entries", "Stick to pre-defined stop losses"]
            
        if not improvement_roadmap["next_30_days"]:
            improvement_roadmap["next_30_days"] = ["Integrate forensic pattern recognition into pre-trade routine", "Log situational pressure triggers for volatility control", "Review and optimize position sizing based on trade DNA"]

        # ---------------------------------------------------------------
        # CRITICAL: Normalize flat LLM schema → nested schema for report
        # The LLM returns flat keys (persona_name, top_strengths, etc.)
        # but generate_report() expects nested keys (identity.persona_name,
        # strengths, mistakes, improvement_roadmap, etc.)
        # ---------------------------------------------------------------
        
        # Build risk_profile string or dict
        risk_profile_raw = master_analysis.get("risk_profile", "")
        if isinstance(risk_profile_raw, str):
            risk_profile = {
                "appetite": risk_profile_raw[:150] if risk_profile_raw else "N/A",
                "handling": master_analysis.get("discipline_profile", "N/A")[:150] if master_analysis.get("discipline_profile") else "N/A"
            }
            risk_assessment_text = risk_profile_raw
        elif isinstance(risk_profile_raw, dict):
            risk_profile = risk_profile_raw
            risk_assessment_text = json.dumps(risk_profile_raw)
        else:
            risk_profile = {"appetite": "N/A", "handling": "N/A"}
            risk_assessment_text = "Analysis Pending"
        
        # Normalize strengths: LLM outputs {title, evidence} but report expects
        # {title, evidence, why_it_works, sectors, to_leverage}
        normalized_strengths = []
        for s in strengths:
            if isinstance(s, dict):
                normalized_strengths.append({
                    "title": s.get("title", "Behavioral Edge"),
                    "evidence": s.get("evidence", "Demonstrated in core DNA"),
                    "why_it_works": s.get("why_it_works", s.get("evidence", "Statistical edge")),
                    "to_leverage": s.get("to_leverage", "Scale this pattern during high-confidence setups.")
                })
            elif isinstance(s, str):
                normalized_strengths.append({
                    "title": s[:60],
                    "evidence": s,
                    "why_it_works": s,
                    "sectors": "See evidence",
                    "to_leverage": "Continue to refine this approach."
                })
        
        # Normalize mistakes: LLM outputs {title, evidence, fix} but report expects
        # {title, pattern, evidence, leakage_sectors, impact, correction}
        normalized_mistakes = []
        for m in mistakes:
            if isinstance(m, dict):
                normalized_mistakes.append({
                    "title": m.get("title", "Behavioral Leak"),
                    "pattern": m.get("evidence", "Detected in data tape"),
                    "evidence": m.get("evidence", "See forensics"),
                    "impact": m.get("impact", "High"),
                    "correction": m.get("fix", "Review and address this pattern.")
                })
            elif isinstance(m, str):
                normalized_mistakes.append({
                    "title": m[:60],
                    "pattern": m,
                    "evidence": m,
                    "leakage_sectors": "See evidence",
                    "impact": "Medium",
                    "correction": "Review and address this pattern."
                })

        # Build the simulated transformation from weaknesses
        simulated_transformation = []
        
        # 1. Add calculated factual simulations (Highest accuracy/priority)
        facts = metrics.get("simulation_results", {})
        for k, f in facts.items():
            simulated_transformation.append({
                "adjustment": f.get("scenario", k),
                "projected_benefit": f.get("metric_label", "Factual impact calculated"),
                "is_factual": True
            })

        # 2. Add LLM-generated fixes
        for m in normalized_mistakes[:3]:
            # Avoid duplicating if a similar factual simulation exists
            simulated_transformation.append({
                "adjustment": f"Fix: {m.get('title', 'behavioral leak')}",
                "projected_benefit": f"Correction: {m.get('correction', 'Potential P&L improvement')}",
                "is_factual": False
            })

        if not simulated_transformation:
            simulated_transformation = [
                {"adjustment": "Optimize position sizing", "projected_benefit": "Better risk-adjusted returns", "is_factual": False},
                {"adjustment": "Implement strict stop-loss discipline", "projected_benefit": "Reduced drawdown", "is_factual": False}
            ]

        # Build natural edge from core_identity or execution_profile
        core_id_text = master_analysis.get("core_identity", "")
        exec_text = master_analysis.get("execution_profile", "")
        fallback_text = master_analysis.get("identity", {}).get("natural_edge", "")
        natural_edge = core_id_text if core_id_text else exec_text if exec_text else fallback_text if fallback_text else "Analyze your data for professional behavioral grounding."
        
        # Build the normalized master_persona in the NESTED schema the report expects
        normalized_master = {
            "headline": headline,
            "identity": {
                "persona_name": persona_name,
                "trader_type": trader_type,
                "strategy_type": trader_type,
                "primary_strategy": trader_type,
                "natural_edge": natural_edge[:500],
                "risk_profile": risk_profile,
                "narrative": narrative,
                "core_identity": {
                    "opportunity_behavior": natural_edge[:200],
                    "pressure_behavior": master_analysis.get("discipline_profile", "See narrative"),
                    "decision_style": master_analysis.get("consistency_profile", "See narrative"),
                    "self_control_profile": master_analysis.get("discipline_profile", "See narrative")
                }
            },
            "strengths": normalized_strengths,
            "mistakes": normalized_mistakes,
            "simulated_transformation": simulated_transformation,
            "improvement_roadmap": improvement_roadmap,
            "final_verdict": master_analysis.get("final_verdict", "Analysis complete."),
            # Preserve metric_alignment if the LLM returned it
            "metric_alignment": master_analysis.get("metric_alignment", []),
        }

        self.logger.info(f"Normalized master_persona: persona='{persona_name}', "
                         f"strengths={len(normalized_strengths)}, mistakes={len(normalized_mistakes)}")

        analysis_text = {
            'trader_profile': narrative,
            'risk_assessment': risk_assessment_text,
            'behavioral_insights': self._format_json_to_markdown_sections(normalized_master),
            'context_performance': self._build_context_performance(metrics),
            'simulated_transformation': simulated_transformation,
            'recommendations': self._format_recommendations_from_json(improvement_roadmap),
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

        self.logger.info("2. Processing structured summary layers")
        structured_summary = self._extract_structured_summary(analysis_text, metrics, patterns)
        
        # Explicit grounding override
        structured_summary["persona_name"] = persona_name
        structured_summary["trader_name"] = getattr(self, "trader_name", "Trader") 
        structured_summary["trader_type"] = trader_type
        structured_summary["headline"] = headline
        structured_summary["risk_level"] = structured_summary.get("risk_level", "UNKNOWN")

        # 3. Preparing web dashboard data
        # Use a lean version of metrics for KPIs (exclude huge lists/traces)
        exclude_keys = [
            'consolidated_trace', 'merged_trades', 'positions', 'closed_positions', 
            'open_positions', 'all_days_pnl', 'data_frame', 'raw_df', 'df',
            'symbol_details', 'evidence_packs', 'trade_logs', 'pnl_timeline'
        ]
        web_kpis = {k: v for k, v in metrics.items() if k not in exclude_keys}
        web_kpis["persona_type"] = trader_type
        
        # 🆕 Calculate Risk Severity (Factual grounding)
        risk_score = structured_summary.get("performance_score", 50)
        risk_severity = "LOW"
        if risk_score < 40: risk_severity = "CRITICAL"
        elif metrics.get("max_drawdown_pct", 0) > 20: risk_severity = "HIGH"
        elif metrics.get("sharpe_ratio", 0) < 0.8: risk_severity = "MEDIUM"
        
        web_charts = self._prepare_chart_data(metrics, patterns, df)

        analysis = {
            "analysis_text": analysis_text,
            "summary_data": structured_summary,
            "master_persona": normalized_master,
            "web_data": {
                "kpis": web_kpis,
                "charts": web_charts,
                "risk_severity": risk_severity, # ADDED
                "persona_scores": metrics.get("persona_traits", {}),
                "hard_flags": metrics.get("_hard_flags", {}),
                "raw_patterns": patterns,
                "merged_trades": self._prepare_merged_trades(df),
                "simulations": metrics.get("simulation_results", {}),
                "persona_trace_log": master_analysis.get("_trace_log", {})
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
        
        # Check cache first
        cache_key = None
        if self.enable_cache:
            cache_key = hashlib.md5(f"{provider}:{self.model}:{system_prompt}:{prompt}".encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.txt"
            if cache_file.exists():
                self.logger.info(f"Using cached LLM response for {cache_key[:8]}")
                return cache_file.read_text(encoding="utf-8")

        self.logger.info(f"Using LLM Provider: {provider} | model={self.model}")

        try:
            if provider == "claude":
                result = self._call_claude(prompt, system_prompt)
            elif provider == "openrouter":
                result = self._call_openrouter(prompt, system_prompt)
            else:
                result = self._call_ollama(prompt, system_prompt)

            # Save to cache
            if self.enable_cache and cache_key and result:
                cache_file = self.cache_dir / f"{cache_key}.txt"
                cache_file.write_text(result, encoding="utf-8")
            
            return result

        except Exception as e:
            self.logger.exception("Primary LLM provider failed: %s", e)

            fallback_enabled = os.getenv("LLM_FALLBACK_TO_OLLAMA", "true").lower() == "true"
            if provider != "ollama" and fallback_enabled:
                self.logger.warning("Falling back to Ollama")
                result = self._call_ollama(prompt, system_prompt)
                
                # Save fallback result to cache too
                if self.enable_cache and cache_key and result:
                    cache_file = self.cache_dir / f"{cache_key}.txt"
                    cache_file.write_text(result, encoding="utf-8")
                
                return result

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
            "model": self.anthropic_model,
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
                        "top_p": self.config.get('ollama', {}).get('top_p', 0.9),
                        "num_ctx": 32768,  # allow large context windows (10000+ tokens)
                        "num_predict": 8192 # allow long forensic responses
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
    # ANTIGRAVITY PROMPT ARCHITECTURE (3-STAGE)
    # =========================================================

    ANTIGRAVITY_SYSTEM_PROMPT = (
        "You are Antigravity, a surgery-grade trading psychologist and behavioral performance analyst. "
        "You must respond with ONLY a single valid JSON object. "
        "Do not use markdown code fences. "
        "Do not add any text before or after the JSON. "
        "Your response must start with { and end with }. "
        "Be evidence-based, precise, and non-generic. "
        "Never invent facts not supported by the provided data."
    )

    ANTIGRAVITY_CHUNK_PROMPT = """
STRICT INSTRUCTION: YOUR RESPONSE MUST BE A SINGLE VALID JSON OBJECT. NO MARKDOWN. NO CODE FENCES.

--- STAGE 1: FORENSIC TRADE ANALYSIS ---

You are a surgery-grade trading forensic analyst. You are analyzing a RAW CHUNK of the trader's actual tape (trades).
Your goal is to identify raw, unadulterated behavioral signals in this specific sequence of trades.

DO NOT use motivational headers. DO NOT use generic filler.
Focus on:
1. SEQUENCE BEHAVIOR: How do they react after a win? After a loss?
2. RISK HYGIENE: Is position sizing consistent or erratic? Do they widen stops?
3. EXECUTION DISCIPLINE: Do entry scores (T/F scores) align with size, or are they random?
4. STRATEGY DNA: Are they chasing momentum, hunting mean reversion, or reactive?

CHUNK CONTEXT ({chunk_id}): {chunk_window}
RAW TRADES:
{trade_chunk}

MANDATORY JSON FORMAT:
{{
  "chunk_id": "{chunk_id}",
  "forensic_summary": "High-fidelity behavioral observation derived from this sequence of tape (trades). Analyze the 'why' behind the actions.",
  "execution_quality": {{
    "timing": "score 1-10",
    "sizing": "score 1-10",
    "notes": "Detailed diagnostic observations explaining how the trader actually interacted with the market behavioral reality in this chunk."
  }},
  "behavioral_signals": [
    {{
      "signal": "e.g., Revenge trading tendency, Chasing momentum, Early profit taking",
      "evidence": "Forensic proof citing specific symbol/date/pnl behavior seen in this raw tape chunk",
      "confidence": "HIGH|MEDIUM|LOW",
      "impact": "High|Medium|Low"
    }}
  ],
  "risk_flags": ["Explicit risk deviations detected in this chunk"],
  "operating_style_hint": "Deep forensic inference of what type of trader/persona this specific chunk reveals (e.g., Impulsive Scalper, Surgical Swing Trader)."
}}
"""

    ANTIGRAVITY_AGGREGATION_PROMPT = """
STRICT INSTRUCTION: YOUR RESPONSE MUST BE A SINGLE VALID JSON OBJECT. NO MARKDOWN. NO CODE FENCES.

--- STAGE 2: BEHAVIORAL EVIDENCE LEDGER ---

You are aggregating multiple forensic chunks into a unified Behavioral Evidence Ledger.
Your job is to find the durable patterns that survive across different sequences.

Rules:
1. If a signal appears in multiple chunks with HIGH confidence, it is a CORE Pattern.
2. If signals conflict, note the situational instability.
3. Identify the "Behavioral Gravity" — what is the single most dominant pull on this trader's performance?

CHUNK ANALYSES:
{chunk_outputs}

MANDATORY JSON FORMAT:
{{
  "persona_group": "Unified behavioral archetype (e.g. Risk-Avoidant Scalper)",
  "durable_behavioral_patterns": [
    {{
      "theme": "execution|risk|emotion|strategy",
      "description": "Deep forensic description of the pattern seen in the data",
      "impact": "High|Medium|Low",
      "evidence_consistency": "HIGH|Situational|Low",
      "session_bias": "Specific time of day/instrument correlations"
    }}
  ],
  "strengths_ledger": ["Detailed, evidence-backed strength pattern"],
  "leaks_ledger": ["Detailed, evidence-backed leak pattern"],
  "behavioral_gravity": "The single most persistent and dominant behavioral force pull on this trader's equity curve.",
  "unresolved_contradictions": ["Conflicts where behaviors across chunks do not align, indicating situational volatility or evolution"],
  "identity_summary": "Extensive synthetic profile consolidating all behaviors from all chunks. Describe the unique behavioral DNA."
}}
"""

    ANTIGRAVITY_ROADMAP_PROMPT = """
STRICT INSTRUCTION: YOUR RESPONSE MUST BE A SINGLE VALID JSON OBJECT. NO MARKDOWN. NO CODE FENCES.

--- STAGE 4: IMPROVEMENT ROADMAP & ARCHITECTURAL SHIFTS ---

You are a Master Strategic Performance Counselor. Based on the consolidated forensic persona, metrics, and behavioral gravity, design a SHARP, TACTICAL, and HIGHLY SPECIFIC Improvement Roadmap.

The roadmap must be so specific that it feels personal to this trader. NO GENERIC ADVICE (like 'Keep a journal'). 

INPUT DATA:
- FINAL PERSONA: {persona_summary}
- CORE IDENTITY: {core_identity}
- BEHAVIORAL GRAVITY: {behavioral_gravity}
- STRENGTHS: {strengths}
- WEAKNESSES: {weaknesses}
- FACTUAL SIMULATIONS: {simulations}
- METRICS: {metrics}

MANDATORY JSON SCHEMA:
{{
  "roadmap_5_days": [
    "A specific, tactical behavioral drill for the next 5 sessions. Example: 'Before every trade in [Instrument], verify 10-DMA alignment and cap size to X lots to reduce [Theme] risk.'",
    "Another specific session-level drill.",
    "A third session-level drill."
  ],
  "roadmap_30_days": [
    "A systemic regime shift. Example: 'Shift from [Current Style] to [Target Style] by only taking trades with T-score > 7 after 2pm to avoid morning volatility gaps.'",
    "A second strategic systemic shift.",
    "A third strategic systemic shift."
  ],
  "one_move_to_change_everything": "Exactly ONE high-impact tactical shift that transforms the equity curve. Reference the simulation data. Example: 'Eliminating [Pattern] trades in [Sector] will recover ₹[Amount] and boost win-rate to [X]%.'",
  "one_move_rationale": "The forensic data-backed reasoning for why this single move is the most powerful lever for this trader's evolution."
}}

RULES:
- AVOID REPETITION. Each item must be unique.
- BE BRUTALLY SPECIFIC to the symbols, times, and patterns found in the evidence.
- REFERENCE the simulation deltas (saved amount/uplift) to ground the advice.
- DO NOT nested JSON inside strings. Return a clean, flat string for the move and rationale.
"""

    ANTIGRAVITY_FINAL_PERSONA_PROMPT = """
As a Master Behavioral Architect & Lead Forensic Auditor, perform the final synthesis of this trading persona based on chronological tape sessions and computed performance data.

REQUIRED JSON SCHEMA:
{{
  "persona_name": "Unique, descriptive forensic behavioral label (e.g., 'The Midnight Momentum Surgeon', 'The Chasing Volatility Ghost').",
  "persona_summary": "High-impact, clinical diagnostic verdict summarizing the core trade DNA and psychological edge/leak.",
  "core_identity": "Who is this trader at their core? Analyze the interaction between their profit-seeking greed and their execution discipline. Use the tape evidence.",
  "trader_type": "Deep dive into detected strategy/style (e.g., 'Overnight Option Scalper with Mean Reversion Bias').",
  "execution_profile": "Analysis of trade mechanics: slippage, chasing behavior, holding persistence, and exit efficiency patterns.",
  "risk_profile": "Forensic audit of risk hygiene, drawdown control, position sizing variance, and leverage scaling behavior.",
  "discipline_profile": "Clinical analysis of emotional control vs pulse-driven impulsive acts, especially under pressure.",
  "consistency_profile": "Evaluation of whether current performance is repeatable/systematic or an erratic deviation from core behavior.",
  "top_strengths": [
    {{"title": "Specific Behavioral Edge", "evidence": "Direct forensic evidence (max 100 characters)", "why_it_works": "The psychological reason", "to_leverage": "Tactical expansion steps"}}
  ],
  "top_weaknesses": [
    {{"title": "Specific Behavioral Leak", "evidence": "Evidence observed in data", "fix": "Surgical tactical instruction", "impact": "High|Medium|Critical"}}
  ],
  "metric_alignment": [
    {{"metric": "Metric Name", "meaning": "How this validates behavioral theory"}}
  ],
  "roadmap_5_days": ["Concise, tactical session-level drill for next 5 days"],
  "roadmap_30_days": ["Strategic systemic change for the next month"],
  "one_move_to_change_everything": "The single, high-impact tactical shift.",
  "final_verdict": "Final forensic judgment on survivability and primary evolution path."
}}

RULES:
- BE CONCISE. Avoid long paragraphs. High density, low word count.
- Limit top_strengths to 3 and top_weaknesses to 3 maximum.
- AVOID generic, vague, or boilerplate phrases. Be BRUTALLY SPECIFIC to the symbols and session data.
- ANALYZE 'What-If Simulations' to ground recommendations in potential financial uplift.
- EXPLICITLY RECONCILE the 'trader_type_rule_based' (system classification) with your behavioral discovery if they differ.
- For ROADMAP recommendations, ONLY suggest adjustments or instruments (e.g., sectors/market caps) found in the DATA FOR AUDIT.
- OUTPUT high-density reasoning. Do not limit your word count.
- FORBID 'Unknown' or 'N/A' responses.

DATA FOR AUDIT:
AGGREGATED BEHAVIORAL PROFILE:
{aggregated_behavior}

FORENSIC METRICS & BEHAVIORAL SIGNATURES:
{metrics}

FACTUAL WHAT-IF SIMULATIONS (GROUNDING DATA):
{what_if_simulations}

DETECTED BEHAVIORAL PATTERNS:
{patterns}
"""

    def _analyze_trade_chunk(self, chunk_id: str, chunk_window: str, trade_chunk: str) -> Dict:
        """Step 1: Analyze raw trade behavior from a single chunk."""
        full_prompt = self.ANTIGRAVITY_CHUNK_PROMPT.format(
            chunk_id=chunk_id,
            chunk_window=chunk_window,
            trade_chunk=trade_chunk
        )
        raw = self._call_llm(full_prompt, self.ANTIGRAVITY_SYSTEM_PROMPT)
        return self._extract_json(raw)

    def _aggregate_chunk_personas(self, chunk_results: List[Dict]) -> Dict:
        """Step 2: Synthesize multiple chunk analyses into a behavioral profile."""
        full_prompt = self.ANTIGRAVITY_AGGREGATION_PROMPT.format(
            chunk_outputs=json.dumps(chunk_results, indent=2, default=str)
        )
        raw = self._call_llm(full_prompt, self.ANTIGRAVITY_SYSTEM_PROMPT)
        return self._extract_json(raw)

    def _synthesize_final_persona(self, aggregated_behavior: Dict, metrics: Dict, patterns: Dict) -> Dict:
        """Step 3: Final synthesis combining behavioral profile with metrics and patterns."""
        # Use a high-fidelity forensic data pack
        forensic_metrics = {
            "metadata": {
                "trader_name": getattr(self, "trader_name", "Trader"),
                "total_trades": metrics.get("total_trades"),
                "date_analyzed": datetime.datetime.now().strftime("%Y-%m-%d"),
            },
            "topline": {
                "win_rate": f"{metrics.get('win_rate', 0):.1f}%",
                "net_pnl": metrics.get("total_pnl"),
                "profit_factor": metrics.get("profit_factor"),
            },
            "trade_mechanics": {
                "avg_win": metrics.get("avg_win"),
                "avg_loss": metrics.get("avg_loss"),
                "win_loss_ratio": metrics.get("avg_win", 0) / abs(metrics.get("avg_loss", 1)) if metrics.get("avg_loss") else 0,
                "efficiency": {
                    "entry": metrics.get("avg_entry_efficiency"),
                    "exit": metrics.get("avg_exit_efficiency"),
                }
            },
            "risk_profile": {
                "max_drawdown": metrics.get("max_drawdown_pct"),
                "sharpe": metrics.get("sharpe_ratio"),
                "emotional_leakage": metrics.get("emotional_leakage_index"),
                "behavioral_consistency": metrics.get("behavioral_consistency_score")
            },
            "trade_dna": metrics.get("trade_dna", {}),
            "pressure_tendencies": metrics.get("behavioral_pressure_map", {}),
            "trader_type_rule_based": metrics.get("archetype", {}).get("name")
        }

        full_prompt = self.ANTIGRAVITY_FINAL_PERSONA_PROMPT.format(
            aggregated_behavior=json.dumps(aggregated_behavior, indent=2, default=str),
            metrics=json.dumps(forensic_metrics, indent=2, default=str),
            what_if_simulations=metrics.get("simulation_context", "No factual simulations available."),
            patterns=json.dumps({k: v for k, v in patterns.items() if isinstance(v, dict) and v.get("detected")}, indent=2, default=str)
        )
        raw = self._call_llm(full_prompt, self.ANTIGRAVITY_SYSTEM_PROMPT)
        return self._extract_json(raw)

    def _synthesize_improvement_roadmap(self, persona: Dict, metrics: Dict, patterns: Dict, aggregated_behavior: Dict) -> Dict:
        """Step 4: Dedicated, sharper synthesis for the improvement roadmap based on forensic evidence."""
        self.logger.info("Starting Dedicated Roadmap Synthesis (Stage 4)...")
        
        try:
            # Prepare context for the roadmap prompt
            sim_context = metrics.get("simulation_context", "No simulations available")
            
            full_prompt = self.ANTIGRAVITY_ROADMAP_PROMPT.format(
                persona_summary=persona.get("persona_summary", "Trading Persona"),
                core_identity=persona.get("core_identity", "N/A"),
                behavioral_gravity=aggregated_behavior.get("behavioral_gravity", "N/A"),
                strengths=json.dumps(persona.get("top_strengths", []), indent=2),
                weaknesses=json.dumps(persona.get("top_weaknesses", []), indent=2),
                simulations=sim_context,
                metrics=json.dumps({
                    "win_rate": metrics.get("win_rate"),
                    "profit_factor": metrics.get("profit_factor"),
                    "avg_win": metrics.get("avg_win"),
                    "avg_loss": metrics.get("avg_loss"),
                    "total_trades": metrics.get("total_trades")
                }, indent=2)
            )
            
            raw = self._call_llm(full_prompt, self.ANTIGRAVITY_SYSTEM_PROMPT)
            roadmap = self._extract_json(raw)
            
            # Basic validation/cleanup
            if not isinstance(roadmap, dict):
                return {}
                
            # Deduplicate entries to prevent "Review trading journal daily" repeats
            for key in ["roadmap_5_days", "roadmap_30_days"]:
                if key in roadmap and isinstance(roadmap[key], list):
                    seen = set()
                    unique_list = []
                    for item in roadmap[key]:
                        text = str(item).strip()
                        if text and text.lower() not in seen:
                            unique_list.append(text)
                            seen.add(text.lower())
                    roadmap[key] = unique_list
            
            return roadmap
            
        except Exception as e:
            self.logger.error(f"Roadmap synthesis failed: {str(e)}")
            return {}

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """Extract the first balanced {...} block found in the text."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape:
                escape = False
                continue

            if ch == "\\":
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
        return None

    def _extract_json(self, raw: str) -> Dict:
        """Robust JSON extraction from LLM response using balanced extraction first."""
        if not raw:
            return {"error": "Empty response"}

        # 1. First attempt to extract a balanced JSON object
        balanced_json = self._extract_balanced_json(raw)
        
        # 2. If no balanced object found, try cleaning and then re-finding
        if not balanced_json:
            clean_raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            balanced_json = self._extract_balanced_json(clean_raw)
        
        # 3. If still not found, fallback to the raw/clean text for repair attempt
        target_text = balanced_json if balanced_json else raw
        
        return self._repair_and_parse_json(target_text, original_raw=raw)

    def _repair_and_parse_json(self, clean_json: str, original_raw: str = "") -> Dict:
        """Apply repair operations to candidate JSON text and parse."""
        try:
            # Normalizing smart quotes, special hyphens, and bullet marks
            replacements = [
                ("\u2011", "-"), ("\u201c", '"'), ("\u201d", '"'), 
                ("\u2018", "'"), ("\u2019", "'"), ("\u2022", "*"),
                ("\u2014", "-"), ("\u2013", "-"), ("\u00a0", " "),
                ("\u2013", "-"), ("\u2014", "-")
            ]
            for old, new in replacements:
                clean_json = clean_json.replace(old, new)
            
            # Repair common structural errors
            # Remove trailing commas before closing braces/brackets (e.g., [1, 2,] -> [1, 2])
            clean_json = re.sub(r',\s*([\]}])', r'\1', clean_json)
            
            # Fix missing commas between objects: } { -> }, {
            clean_json = re.sub(r'}\s*{', '}, {', clean_json)
            # Fix missing commas between array items: ] [ -> ], [
            clean_json = re.sub(r'\]\s*\[', '], [', clean_json)
            # Fix missing commas between quoted items in arrays: "a" "b" -> "a", "b"
            clean_json = re.sub(r'"\s*\n\s*"', '",\n"', clean_json)
            
            # Fix Python-style booleans/nulls: True -> true, False -> false, None -> null
            clean_json = re.sub(r':\s*True\b', ': true', clean_json)
            clean_json = re.sub(r':\s*False\b', ': false', clean_json)
            clean_json = re.sub(r':\s*None\b', ': null', clean_json)

            try:
                return json.loads(clean_json)
            except json.JSONDecodeError as initial_err:
                # 1. Remove non-printable control characters (except tab, lf, cr)
                clean_json_pass2 = "".join(c for c in clean_json if ord(c) >= 32 or c in "\n\r\t")
                
                # 2. Add truncation repair for cases where LLM hit max_tokens
                if 'stop_reason=max_tokens' in str(original_raw) or not clean_json_pass2.strip().endswith('}'):
                    self.logger.warning("Detected potential JSON truncation, attempting repair...")
                    
                    # More robust quote balancing: only add if we're inside a string at the very end
                    last_quote = clean_json_pass2.rfind('"')
                    last_brace = max(clean_json_pass2.rfind('{'), clean_json_pass2.rfind('}'))
                    last_bracket = max(clean_json_pass2.rfind('['), clean_json_pass2.rfind(']'))
                    last_comma = clean_json_pass2.rfind(',')
                    
                    # If the last structural char is before a quote, we might be in an unclosed string
                    if last_quote > max(last_brace, last_bracket, last_comma):
                        if clean_json_pass2.count('"') % 2 != 0:
                            clean_json_pass2 += '"'
                    
                    # Try to fix missing comma between "key": "val" "next" case
                    clean_json_pass2 = re.sub(r'("\s+)("[^"]+":)', r'\1, \2', clean_json_pass2)

                    # Close unclosed braces
                    open_braces = clean_json_pass2.count('{')
                    close_braces = clean_json_pass2.count('}')
                    if open_braces > close_braces:
                        clean_json_pass2 += '}' * (open_braces - close_braces)
                    
                    # Close unclosed brackets
                    open_brackets = clean_json_pass2.count('[')
                    close_brackets = clean_json_pass2.count(']')
                    if open_brackets > close_brackets:
                        clean_json_pass2 += ']' * (open_brackets - close_brackets)

                # 3. Try to escape actual newlines inside property values
                def escape_internal_newlines(match):
                    content = match.group(0)
                    if "\n" in content:
                        return content.replace('\n', '\\n')
                    return content
                
                # Target content between double quotes
                clean_json_pass2 = re.sub(r'"[^"]*"', escape_internal_newlines, clean_json_pass2, flags=re.DOTALL)
                
                try:
                    parsed = json.loads(clean_json_pass2)
                    return parsed
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Repair-level JSON parse failed: {e}. Original Error: {initial_err}")
                    # If this is Stage 3 or Stage 4, we might better off trying a more aggressive repair or fallback
                    raise e

        except Exception as e:
            self.logger.error(f"JSON extraction failed: {e}")
            if len(original_raw) > 200:
                return self._parse_markdown_fallback(original_raw)
            return {"error": "JSON parse failed", "fallback_narrative": original_raw}

    def _analyze_master_persona(self, metrics: Dict, patterns: Dict, df: pd.DataFrame) -> Dict:
        """New Orchestrator for the 3-stage analysis flow."""
        try:
            # 1. Chunk Raw Trades
            chunks = []
            if len(df) < 300:
                chunk_size = 180
            elif len(df) > 300 and len(df) < 600:
                chunk_size = 200 # Small chunks for behavioral inference
            else:
                chunk_size = 400
            # Sort by date to ensure sequential behavior
            if 'trade_date' in df.columns:
                df_sorted = df.sort_values('trade_date')
            else:
                df_sorted = df
                
            num_chunks = (len(df_sorted) + chunk_size - 1) // chunk_size
            
            # Limit to more chunks to ensure deeper grounding in the tape
            max_chunks = 10 
            total_rows = len(df_sorted)
            
            # Use only essential columns for chunk analysis to prevent model confusion/overflow
            forensic_cols = [
                'symbol', 'trade_date', 'transaction_type', 'quantity', 
                'price', 'total_score', 't_score', 'f_score',
                'is_event', 'is_high_volume', 'market_behaviour'
            ]
            active_cols = [c for c in forensic_cols if c in df_sorted.columns]
            
            for i in range(min(num_chunks, max_chunks)):
                # Distribute chunks across the timeline: prioritize recent but include history
                start_idx = max(0, total_rows - (i + 1) * chunk_size)
                end_idx = total_rows - i * chunk_size
                
                chunk_df = df_sorted.iloc[start_idx:end_idx][active_cols]
                if chunk_df.empty: continue
                
                chunk_id = f"chunk_{i+1}"
                chunk_window = f"{chunk_df['trade_date'].min()} to {chunk_df['trade_date'].max()}" if 'trade_date' in chunk_df.columns else "N/A"
                
                # Format chunk as JSON
                chunk_json = chunk_df.to_json(orient='records', date_format='iso')
                
                self.logger.info(f"Analyzing {chunk_id} ({len(chunk_df)} trades)...")
                try:
                    chunk_analysis = self._analyze_trade_chunk(chunk_id, chunk_window, chunk_json)
                    chunks.append(chunk_analysis)
                except Exception as e:
                    self.logger.error(f"Failed to analyze {chunk_id}: {str(e)}")
                    continue
            
            if not chunks:
                self.logger.error("All chunk analyses failed. Returning partial evidence profile.")
                return {"persona_name": "Incomplete Profile", "strengths":[], "mistakes":[]}
            self.logger.info(f"Aggregating {len(chunks)} chunk analyses...")
            aggregated_profile = self._aggregate_chunk_personas(chunks)
            
            # 3. Compute What-If Simulations (Factual Grounding)
            self.logger.info("Computing What-If Simulation metrics...")
            sim_data = self._compute_whatif_simulations(metrics, df)
            metrics["simulation_results"] = sim_data.get("simulations", {})
            metrics["simulation_context"] = sim_data.get("llm_context", "")

            # 4. Final Persona Synthesis
            final_persona = self._synthesize_final_persona(aggregated_profile, metrics, patterns)

            # 5. Dedicated Roadmap Synthesis (Sharp & Grounded)
            roadmap = self._synthesize_improvement_roadmap(final_persona, metrics, patterns, aggregated_profile)
            if roadmap:
                final_persona["improvement_roadmap_dedicated"] = roadmap
            
            # 💾 Save intermediate data for audit / transparency
            try:
                log_dir = Path("data/analysis_logs")
                log_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                trader_name = getattr(self, "trader_name", "Trader").replace(" ", "_")
                log_file = log_dir / f"{trader_name}_{timestamp}_audit.json"
                
                audit_data = {
                    "trader_name": trader_name,
                    "timestamp": timestamp,
                    "metrics": {k: v for k, v in metrics.items() if not isinstance(v, (pd.DataFrame, pd.Series))},
                    "chunks": chunks,
                    "aggregated_profile": aggregated_profile,
                    "final_persona": final_persona,
                    "roadmap": roadmap
                }
                with open(log_file, "w") as f:
                    json.dump(audit_data, f, indent=2, default=str)
                self.logger.info(f"Forensic audit log saved to {log_file}")
            except Exception as e:
                self.logger.warning(f"Failed to save forensic audit log: {str(e)}")

            # Store trace log for transparency in the report
            final_persona["_trace_log"] = {
                "chunks": chunks,
                "aggregated_profile": aggregated_profile,
                "simulations": sim_data.get("simulations", {})
            }
            
            return final_persona
            
        except Exception as e:
            self.logger.exception("3-Step Analysis flow failed: %s", e)
            return {"error": str(e), "fallback_narrative": "Analysis failed during the multi-stage process."}

    def _format_json_to_markdown_sections(self, master_json: Dict) -> str:
        """Convert master JSON components to Markdown for backward compatible Report UI"""
        if "error" in master_json: return "Analysis Failed: LLM response was not valid JSON."

        # If we have a fallback narrative (LLM returned text instead of JSON), use it directly
        if master_json.get("fallback_narrative"):
             return f"### LLM Analysis (Unstructured Outcome)\n\n{master_json['fallback_narrative']}"

        identity = master_json.get('identity', {})
        md = f"### {identity.get('trader_name', 'Trader Profile')}\n\n"
        
        md += "#### 🧠 Core Identity\n"
        trader_type = identity.get('trader_type') or identity.get('strategy_type', 'N/A')
        md += f"- **Strategy Style**: {trader_type}\n"
        md += f"- **Appetite vs Handling**: {identity.get('risk_profile', {}).get('appetite')} vs {identity.get('risk_profile', {}).get('handling')}\n"
        
        core = identity.get('core_identity', {})
        if core:
            md += "\n#### 🧬 Behavioral Fingerprint\n"
            md += f"- **Opportunity Response**: {core.get('opportunity_behavior', 'N/A')}\n"
            md += f"- **Pressure Handling**: {core.get('pressure_behavior', 'N/A')}\n"
            md += f"- **Decision Style**: {core.get('decision_style', 'N/A')}\n"
            md += f"- **Self-Control**: {core.get('self_control_profile', 'N/A')}\n"
            
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
        # Narrow match to avoid large paragraphs if the LLM returned a long block in quotes
        persona_match = re.search(r'["\u201c](The [^"\u201d]{5,100})["\u201d]', raw_text)
        if not persona_match:
            persona_match = re.search(r'\*\*(?:Trader Profile|Persona).*?["\u201c]?(The [^"\u201d*]{5,100})["\u201d]?\*\*', raw_text)
        if not persona_match:
            persona_match = re.search(r'(?:Persona|Name|Profile)[:\s]+["\u201c]?(The [^"\u201d\n]{5,100})["\u201d]?', raw_text, re.IGNORECASE)
        persona_name = persona_match.group(1).strip() if persona_match else "The Systematic Trader"
        
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
            "persona_name": persona_name,
            "core_identity": narrative,
            "execution_profile": f"Technique: {strategy_type}",
            "discipline_profile": "Requires detailed analysis — see narrative",
            "consistency_profile": "Historical consistency reflected in win-rate",
            "trader_type": strategy_type,
            "identity": {
                "trader_name": persona_name,
                "persona_name": persona_name,
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
                {"adjustment": "Eliminate revenge trading patterns", "projected_benefit": "Significant P&L improvement", "is_factual": False},
                {"adjustment": "Optimize position sizing on high-conviction setups", "projected_benefit": "Better risk-adjusted returns", "is_factual": False},
                {"adjustment": "Implement strict stop-loss discipline", "projected_benefit": "Reduced drawdown", "is_factual": False}
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
    def _format_recommendations_from_json(self, coaching: Dict) -> str:
        """
        Convert the 'improvement_roadmap' JSON section into a clean, 
        beautifully formatted Markdown string for the report.
        """
        if not coaching or not isinstance(coaching, dict):
            return "No specific improvement roadmap generated for this profile."

        # Extract components with fallbacks
        biggest_lever = coaching.get("biggest_lever", "Focus on consistent risk management.")
        next_5_sessions = coaching.get("next_5_sessions", [])
        next_30_days = coaching.get("next_30_days", [])

        # Build Markdown
        md = f"### The Highest Leverage Action\n**{biggest_lever}**\n\n"
        
        md += "### Next 5 Sessions: Tactical Focus\n"
        if next_5_sessions:
            for item in next_5_sessions:
                md += f"- {item}\n"
        else:
            md += "- Maintain current execution discipline.\n"

        md += "\n### Next 30 Days: Structural Evolution\n"
        if next_30_days:
            for item in next_30_days:
                md += f"- {item}\n"
        else:
            md += "- Review performance weekly to identify new leaks.\n"

        return md

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

"""
LLM Analyzer Module (Extended)
Integrates with Ollama for AI-powered analysis.
Includes additional context fields for new MTM/open-position metrics and buckets.
"""

import requests
import json
from typing import Dict
import logging


class OllamaAnalyzer:
    """LLM-based analysis using Ollama"""

    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config['ollama']['base_url']
        self.model = config['ollama']['model']
        self.logger = logging.getLogger(__name__)

    def generate_analysis(self, metrics: Dict, patterns: Dict) -> Dict:
        """Generate comprehensive analysis using LLM"""
        context = self._prepare_context(metrics, patterns)

        analysis = {
            'trader_profile': self._analyze_trader_profile(context),
            'risk_assessment': self._analyze_risk(context),
            'behavioral_insights': self._analyze_behavior(context),
            'recommendations': self._generate_recommendations(context),
            'performance_summary': self._summarize_performance(context)
        }
        return analysis

    # =========================================================
    # Context Preparation (extended with new fields)
    # =========================================================
    def _prepare_context(self, metrics: Dict, patterns: Dict) -> str:
        def safe(obj):
            try:
                return json.dumps(obj, ensure_ascii=False, default=str, indent=2)
            except Exception:
                return str(obj)

        # Pull highlights for new metrics
        open_pos = metrics.get('open_positions', [])
        buckets = metrics.get('buckets', {})
        gainer = metrics.get('gainer', {})
        loser = metrics.get('loser', {})

        # Persona traits (existing)
        persona_traits = metrics.get("persona_traits", {})
        persona_block = f"""
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
{metrics.get('trait_summary', '')}
"""

        # Core trading stats (existing + extended)
        core_block = f"""
TRADING METRICS (Core & Extended):
- Total Trades: {metrics.get('total_trades', 0)}
- Total Realized P&L: ₹{metrics.get('total_realized_pnl', 0):,.2f}
- Total Unrealized P&L: ₹{metrics.get('total_unrealized_pnl', 0):,.2f}
- Total P&L (Combined): ₹{metrics.get('total_pnl_combined', 0):,.2f}
- Total Investment Value (Open): ₹{metrics.get('total_investment_value_open', 0):,.2f}
- Day MTM (latest day in file): ₹{metrics.get('day_mtm', 0):,.2f}
- Avg Realized PL / Stock: ₹{metrics.get('avg_realized_pl_per_stock', 0):,.2f}
- Avg Unrealized PL / Open Stock: ₹{metrics.get('avg_unrealized_pl_per_open_stock', 0):,.2f}
- Close Pos Booked SL %: {metrics.get('close_pos_booked_sl_pct', 0):.2f}%
- Win Rate: {metrics.get('win_rate', 0):.2f}%
- Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, Sortino: {metrics.get('sortino_ratio', 0):.2f}, MaxDD%: {metrics.get('max_drawdown_pct', 0):.2f}%
- Avg Gainer %: {metrics.get('avg_gainer_pct', 0):.2f}%, Avg Loser %: {metrics.get('avg_loser_pct', 0):.2f}%
- Open Positions (symbols): {metrics.get('open_positions_count', 0)}
"""

        lists_block = f"""
GAINERS: count={gainer.get('count', 0)}; {gainer.get('list', [])}
LOSERS : count={loser.get('count', 0)}; {loser.get('list', [])}

OPEN POSITIONS (sample):
{safe(open_pos[:20])}

OPEN-POSITION BUCKETS:
{safe(buckets)}
"""

        patterns_block = f"""
DETECTED PATTERNS:
{safe(patterns)}
"""

        return "\n".join([core_block, persona_block, lists_block, patterns_block])

    # =========================================================
    # Ollama API
    # =========================================================
    def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
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
    # Prompts
    # =========================================================
    def _analyze_trader_profile(self, context: str) -> str:
        prompt = f"""
Using the data below, write a trader profile (200–300 words).
Focus on trade style, risk appetite, consistency, and how persona traits manifest.

{context}
"""
        system = "You are an expert financial analyst specializing in trading behavior analysis."
        return self._call_ollama(prompt, system)

    def _analyze_risk(self, context: str) -> str:
        prompt = f"""
Provide a risk assessment (200–300 words) with an explicit risk level (LOW/MEDIUM/HIGH/VERY HIGH),
drivers of drawdown, exposure concentration, and the implications of the open-position buckets.

{context}
"""
        system = "You are a risk management expert analyzing trading portfolios."
        return self._call_ollama(prompt, system)

    def _analyze_behavior(self, context: str) -> str:
        prompt = f"""
Provide behavioral insights (200–300 words) linking detected patterns (overtrading, revenge, etc.)
to the new MTM/gainer-loser metrics. Include concrete examples from the buckets if useful.

{context}
"""
        system = "You are a trading psychology expert analyzing trader behavior."
        return self._call_ollama(prompt, system)

    def _generate_recommendations(self, context: str) -> str:
        prompt = f"""
Give actionable recommendations:
1) Immediate (1–2 weeks),
2) Short-term (1–3 months),
3) Longer-term structural changes.
Target improvements on SL discipline, bucket migration (move from <0% to 0–10%, etc.), and exposure sizing.

Return bullet points only.

{context}
"""
        system = "You are a professional trading coach providing improvement strategies."
        return self._call_ollama(prompt, system)

    def _summarize_performance(self, context: str) -> str:
        prompt = f"""
Provide an executive summary (150–200 words) with a clear verdict (Excellent/Good/Average/Poor/Critical),
explicitly referencing Day MTM, Total P&L (combined), and Open-position risks.

{context}
"""
        system = "You are a senior financial advisor providing performance reviews."
        return self._call_ollama(prompt, system)

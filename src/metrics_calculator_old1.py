"""
Metrics Calculator Module
Calculates comprehensive trading performance metrics
and derives behavioral persona traits
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging
from scipy.stats import skew, kurtosis


class TradingMetricsCalculator:
    """Calculate comprehensive trading metrics and trading persona traits"""

    def __init__(self, config: Dict):
        self.config = config
        self.risk_free_rate = config['metrics']['risk_free_rate']
        self.trading_days = config['metrics']['trading_days_per_year']
        self.logger = logging.getLogger(__name__)

    # =========================================================
    # Core Metric Aggregation
    # =========================================================
    def calculate_all_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate all trading metrics and persona traits"""

        metrics = {
            'total_trades': len(df),
            'total_pnl': self.calculate_total_pnl(df),
            'win_rate': self.calculate_win_rate(df),
            'avg_win': self.calculate_avg_win(df),
            'avg_loss': self.calculate_avg_loss(df),
            'profit_factor': self.calculate_profit_factor(df),
            'sharpe_ratio': self.calculate_sharpe_ratio(df),
            'sortino_ratio': self.calculate_sortino_ratio(df),
            'max_drawdown': self.calculate_max_drawdown(df),
            'max_drawdown_pct': self.calculate_max_drawdown_pct(df),
            'avg_trade_value': self.calculate_avg_trade_value(df),
            'largest_win': self.calculate_largest_win(df),
            'largest_loss': self.calculate_largest_loss(df),
            'consecutive_wins': self.calculate_consecutive_wins(df),
            'consecutive_losses': self.calculate_consecutive_losses(df),
            'avg_holding_period': self.calculate_avg_holding_period(df),
            'avg_trades_per_day': self.calculate_avg_trades_per_day(df),
            'date_range': self.get_date_range(df),
            'trading_days': self.get_trading_days(df)
        }

        # Add persona-level behavioral metrics
        persona_traits = self.calculate_persona_traits(df)
        metrics.update(persona_traits)

        # Add metrics that may have been added (like EMA)
        if 'ema_allocation' in df.columns or any(col.startswith('ema_score') for col in df.columns):
            metrics['ema_enabled'] = True

        return metrics

    # =========================================================
    # Basic Metrics
    # =========================================================
    def calculate_total_pnl(self, df: pd.DataFrame) -> float:
        """Calculate total P&L"""
        return float(df['pnl'].sum())

    def calculate_win_rate(self, df: pd.DataFrame) -> float:
        """Calculate win rate percentage"""
        if len(df) == 0:
            return 0.0
        winning_trades = len(df[df['pnl'] > 0])
        return float(winning_trades / len(df) * 100)

    def calculate_avg_win(self, df: pd.DataFrame) -> float:
        """Calculate average winning trade"""
        winning_trades = df[df['pnl'] > 0]['pnl']
        return float(winning_trades.mean()) if len(winning_trades) > 0 else 0.0

    def calculate_avg_loss(self, df: pd.DataFrame) -> float:
        """Calculate average losing trade"""
        losing_trades = df[df['pnl'] < 0]['pnl']
        return float(losing_trades.mean()) if len(losing_trades) > 0 else 0.0

    def calculate_profit_factor(self, df: pd.DataFrame) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    def calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        if len(df) < 2:
            return 0.0
        returns = df['pnl'] / df['trade_value']
        if returns.std() == 0:
            return 0.0
        excess_return = returns.mean() - (self.risk_free_rate / self.trading_days)
        sharpe = excess_return / returns.std() * np.sqrt(self.trading_days)
        return float(sharpe)

    def calculate_sortino_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        if len(df) < 2:
            return 0.0
        returns = df['pnl'] / df['trade_value']
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        downside_std = downside_returns.std()
        if downside_std == 0:
            return 0.0
        excess_return = returns.mean() - (self.risk_free_rate / self.trading_days)
        sortino = excess_return / downside_std * np.sqrt(self.trading_days)
        return float(sortino)

    def calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown in rupees"""
        df_sorted = df.sort_values('trade_date')
        cumulative_pnl = df_sorted['pnl'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        return float(drawdown.min())

    def calculate_max_drawdown_pct(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown percentage based on cumulative realized P&L"""
        if df.empty or df['pnl'].sum() == 0:
            return 0.0
        df_sorted = df.sort_values('trade_date')
        cumulative_pnl = df_sorted['pnl'].cumsum()
        running_max = cumulative_pnl.cummax().replace(0, np.nan)
        drawdown_pct = (cumulative_pnl - running_max) / running_max * 100
        max_drawdown_pct = abs(float(drawdown_pct.min()))
        return round(max_drawdown_pct, 2)

    def calculate_avg_trade_value(self, df: pd.DataFrame) -> float:
        """Calculate average trade value"""
        return float(df['trade_value'].mean())

    def calculate_largest_win(self, df: pd.DataFrame) -> float:
        """Calculate largest single win"""
        return float(df['pnl'].max()) if len(df) > 0 else 0.0

    def calculate_largest_loss(self, df: pd.DataFrame) -> float:
        """Calculate largest single loss"""
        return float(df['pnl'].min()) if len(df) > 0 else 0.0

    def calculate_consecutive_wins(self, df: pd.DataFrame) -> int:
        """Calculate maximum consecutive wins"""
        df_sorted = df.sort_values('trade_date')
        max_consecutive, current = 0, 0
        for pnl in df_sorted['pnl']:
            if pnl > 0:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        return int(max_consecutive)

    def calculate_consecutive_losses(self, df: pd.DataFrame) -> int:
        """Calculate maximum consecutive losses"""
        df_sorted = df.sort_values('trade_date')
        max_consecutive, current = 0, 0
        for pnl in df_sorted['pnl']:
            if pnl < 0:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        return int(max_consecutive)

    def calculate_avg_holding_period(self, df: pd.DataFrame) -> float:
        """Calculate average holding period in minutes"""
        return float(df['holding_period_minutes'].mean()) if 'holding_period_minutes' in df.columns else 0.0

    def calculate_avg_trades_per_day(self, df: pd.DataFrame) -> float:
        """Calculate average trades per day"""
        trading_days = df['trade_date'].dt.date.nunique()
        return float(len(df) / trading_days) if trading_days > 0 else 0.0

    def get_date_range(self, df: pd.DataFrame) -> str:
        """Get date range of trading data"""
        start_date = df['trade_date'].min().strftime('%Y-%m-%d')
        end_date = df['trade_date'].max().strftime('%Y-%m-%d')
        return f"{start_date} to {end_date}"

    def get_trading_days(self, df: pd.DataFrame) -> int:
        """Get number of unique trading days"""
        return int(df['trade_date'].dt.date.nunique())

    # =========================================================
    # Persona Trait Analysis
    # =========================================================
    def calculate_persona_traits(self, df: pd.DataFrame) -> Dict:
        """Compute behavioral and cognitive persona metrics"""
        if df.empty:
            return {}

        traits = {
            "discipline_score": self._calc_discipline_score(df),
            "emotional_control": self._calc_emotional_control(df),
            "risk_appetite": self._calc_risk_appetite(df),
            "patience": self._calc_patience(df),
            "adaptability": self._calc_adaptability(df),
            "consistency": self._calc_consistency(df),
            "confidence": self._calc_confidence(df),
        }

        persona_type = self._map_persona(traits)
        trait_summary = self._summarize_persona(traits, persona_type)

        return {
            "persona_type": persona_type,
            "trait_summary": trait_summary,
            "persona_traits": traits
        }

    # =========================================================
    # Trait Calculation Methods
    # =========================================================
    def _calc_discipline_score(self, df: pd.DataFrame) -> float:
        trade_size_var = np.std(df["quantity"]) / (np.mean(df["quantity"]) + 1e-6)
        pnl_vol = np.std(df["pnl"])
        trade_freq_var = df["trade_date"].dt.date.value_counts().std()
        discipline = 1 / (1 + (0.5 * trade_size_var + 0.5 * (pnl_vol / 1000) + 0.1 * (trade_freq_var or 0)))
        return round(discipline, 2)

    def _calc_emotional_control(self, df: pd.DataFrame) -> float:
        df_sorted = df.sort_values("trade_date")
        pnl_shift = df_sorted["pnl"].shift(1)
        time_diff = (df_sorted["trade_date"] - df_sorted["trade_date"].shift(1)).dt.total_seconds() / 60
        revenge_trades = ((pnl_shift < 0) & (time_diff < 30)).sum()
        impulsive_trades = (time_diff < 5).sum()
        avg_recovery_time = np.mean(time_diff[pnl_shift < 0]) if (pnl_shift < 0).any() else 30
        control = max(0, 1 - (0.3 * revenge_trades / len(df) + 0.3 * impulsive_trades / len(df)))
        control *= min(1.0, avg_recovery_time / 30)
        return round(control, 2)

    def _calc_risk_appetite(self, df: pd.DataFrame) -> float:
        max_dd = abs(df["pnl"].cumsum().min())
        avg_trade_value = df["trade_value"].mean()
        pnl_std = np.std(df["pnl"])
        raw_score = (pnl_std + avg_trade_value * 0.001) / (max_dd + 1e-6)
        risk_appetite = min(1.0, max(0.0, raw_score * 5))
        return round(risk_appetite, 2)

    def _calc_patience(self, df: pd.DataFrame) -> float:
        if "holding_period_minutes" not in df.columns:
            return 0.5
        pos_trades = df[df["pnl"] > 0]
        neg_trades = df[df["pnl"] < 0]
        pos_hold = pos_trades["holding_period_minutes"].mean() if len(pos_trades) > 0 else 0
        neg_hold = neg_trades["holding_period_minutes"].mean() if len(neg_trades) > 0 else 0
        patience_ratio = pos_hold / (neg_hold + 1)
        return round(min(1.0, patience_ratio / 2), 2)

    def _calc_adaptability(self, df: pd.DataFrame) -> float:
        symbol_perf = df.groupby("symbol")["pnl"].mean()
        time_perf = df.groupby(df["trade_hour"])["pnl"].mean()
        perf_std = np.std(symbol_perf) + np.std(time_perf)
        adaptability = max(0.1, 1 / (1 + perf_std / 100))
        return round(adaptability, 2)

    def _calc_consistency(self, df: pd.DataFrame) -> float:
        pnl_cum = df["pnl"].cumsum()
        pnl_rolling = pnl_cum.rolling(window=5).std().fillna(0)
        volatility = pnl_rolling.mean()
        daily_mean = df.groupby(df["trade_date"].dt.date)["pnl"].mean().mean()
        consistency = max(0.0, 1 - abs(volatility / (abs(daily_mean) + 1)))
        return round(consistency, 2)

    def _calc_confidence(self, df: pd.DataFrame) -> float:
        df_sorted = df.sort_values("trade_date")
        pnl_shift = df_sorted["pnl"].shift(1)
        size_shift = df_sorted["quantity"]
        post_win_growth = size_shift[pnl_shift > 0].mean() / (size_shift.mean() + 1e-6)
        post_loss_growth = size_shift[pnl_shift < 0].mean() / (size_shift.mean() + 1e-6)
        confidence = post_win_growth - post_loss_growth
        confidence_score = min(1.0, max(0.0, 0.5 + confidence * 0.5))
        return round(confidence_score, 2)

    # =========================================================
    # Persona Mapping
    # =========================================================
    def _map_persona(self, traits: Dict[str, float]) -> str:
        r, p, d, c, e = (
            traits["risk_appetite"],
            traits["patience"],
            traits["discipline_score"],
            traits["consistency"],
            traits["emotional_control"],
        )
        if r > 0.8 and e < 0.5:
            return "Aggressive Impulsive Trader"
        elif d > 0.7 and c > 0.7 and e > 0.6:
            return "Disciplined Systematic Trader"
        elif p > 0.7 and r < 0.5:
            return "Patient Swing Trader"
        elif r > 0.6 and c < 0.5:
            return "Momentum Chaser"
        elif e < 0.4:
            return "Emotionally Reactive Trader"
        else:
            return "Balanced Discretionary Trader"

    def _summarize_persona(self, traits: Dict[str, float], persona_type: str) -> str:
        summary = (
            f"Trader shows characteristics of a **{persona_type}**.\n\n"
            f"- Discipline: {traits['discipline_score']*100:.0f}%\n"
            f"- Emotional Control: {traits['emotional_control']*100:.0f}%\n"
            f"- Risk Appetite: {traits['risk_appetite']*100:.0f}%\n"
            f"- Patience: {traits['patience']*100:.0f}%\n"
            f"- Adaptability: {traits['adaptability']*100:.0f}%\n"
            f"- Consistency: {traits['consistency']*100:.0f}%\n"
            f"- Confidence: {traits['confidence']*100:.0f}%\n"
        )
        return summary

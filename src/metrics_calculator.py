"""
Metrics Calculator Module (Extended)
Calculates comprehensive trading performance metrics, open-position analytics,
gainer/loser breakdowns, MTM (day & open), and derives behavioral persona traits.

This module is backward-compatible with your previous version and preserves:
- Core trading metrics (Sharpe, Sortino, DD, consecutive wins/losses, etc.)
- Persona trait calculations and persona mapping
- P&L timeline export for charts (report generator uses this)

New/extended:
- TOTAL Trade
- Gainer / Loser counts and lists (symbol-level net P&L)
- Avg Gainer % and Avg Loser % (symbol-level, explicit % returns)
- DAY MTM (realised on latest trading date in the file)
- Avg Realized profit/loss (total realized / number of symbols traded)
- Avg Unrealized profit/loss (total unrealized / number of open symbols)
- Close Pos Booked SL % (losing realised trades / total realised trades)
- Total Realized, Total Unrealized, Total P/L
- Total Investment Value (open positions cost)
- OPEN POSITION (count and details snapshot)
- Open-position buckets by % change:
  <0%, 0–5%, 5–10%, 10–20%, 20–40%, 40–60%, 60–80%, >80%
  (each with count, list, total value, share of total MTM and share of open positions)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from scipy.stats import skew, kurtosis


class TradingMetricsCalculator:
    """Calculate comprehensive trading metrics and trading persona traits + extended MTM/open-position analytics"""

    def __init__(self, config: Dict):
        self.config = config
        self.risk_free_rate = config['metrics']['risk_free_rate']
        self.trading_days = config['metrics']['trading_days_per_year']
        self.logger = logging.getLogger(__name__)

    # =========================================================
    # Public Entry
    # =========================================================
    def calculate_all_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate all trading metrics and persona traits with extended analytics."""
        df = df.copy()

        # Normalize columns (defensive)
        if 'transaction_type' in df.columns:
            df['transaction_type'] = df['transaction_type'].astype(str).str.upper().replace({'SELL': 'SALE'})
        if 'trade_date' in df.columns and not np.issubdtype(df['trade_date'].dtype, np.datetime64):
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')

        # Basic safety
        if df.empty:
            return {
                'total_trades': 0,
                'pnl_timeline': {'dates': [], 'values': []},
                'persona_type': 'N/A',
                'trait_summary': 'No data',
                'persona_traits': {}
            }

        # -------- Core metrics (existing) --------
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
            'trading_days': self.get_trading_days(df),
            'pnl_timeline': self._build_pnl_timeline(df),
        }

        # -------- Extended position & MTM analytics --------
        pos = self._compute_positions_snapshot(df)
        metrics.update(pos['aggregates'])       # realized/unrealized, investment, day MTM, averages, SL%
        metrics['open_positions'] = pos['positions']  # per-symbol open info
        metrics['buckets'] = pos['buckets']     # bucket analytics
        metrics['gainer'] = pos['gainer']       # counts + list of gainer symbols
        metrics['loser'] = pos['loser']         # counts + list of loser symbols

        # -------- Persona traits (existing) --------
        persona_traits = self.calculate_persona_traits(df)
        metrics.update(persona_traits)

        # Flag if EMA columns exist (backward-compat)
        if 'ema_allocation' in df.columns or any(col.startswith('ema_score') for col in df.columns):
            metrics['ema_enabled'] = True

        # --- 🧩 NEW METRICS (added safely) ---
        # Market / Price-action
        metrics["avg_daily_range"] = (df["high"] - df["low"]).mean()
        metrics["avg_close_to_open_return"] = ((df["close"] - df["open"]) / df["open"]).mean() * 100
        metrics["volatility_index"] = (((df["high"] - df["low"]) / df["open"]).std() * 100)
        metrics["volume_volatility"] = df["volume"].std() / df["volume"].mean() if df["volume"].mean() != 0 else 0
        metrics["avg_volume_per_trade"] = df["volume"].mean()

        # Score-based
        for col in ["t_score", "f_score", "total_score"]:
            if col in df.columns:
                metrics[f"avg_{col}"] = df[col].mean()
                metrics[f"{col}_volatility"] = df[col].std()

        if "total_score" in df.columns:
            score_mean = df["total_score"].mean()
            metrics["high_score_win_rate"] = (df.loc[df["total_score"] > score_mean, "pnl"] > 0).mean() * 100
            metrics["low_score_loss_rate"] = (df.loc[df["total_score"] < score_mean, "pnl"] <= 0).mean() * 100

        # Technical hit rates
        if "is_52week_high" in df.columns:
            metrics["hit_rate_52w_high"] = df["is_52week_high"].mean() * 100
        if "is_52week_low" in df.columns:
            metrics["hit_rate_52w_low"] = df["is_52week_low"].mean() * 100
        if "is_alltime_high" in df.columns:
            metrics["hit_rate_alltime_high"] = df["is_alltime_high"].mean() * 100

        # PnL Shape / Distribution
        metrics["pnl_volatility"] = df["pnl"].std()
        metrics["pnl_skewness"] = df["pnl"].skew()
        metrics["pnl_kurtosis"] = df["pnl"].kurt()
        metrics["value_at_risk_95"] = df["pnl"].quantile(0.05)

        # Position / Holding behavior
        if "holding_period" in df.columns:
            metrics["avg_holding_period"] = df["holding_period"].mean()
            metrics["holding_period_volatility"] = df["holding_period"].std()
            metrics["avg_holding_period_winners"] = df.loc[df["pnl"] > 0, "holding_period"].mean()
            metrics["avg_holding_period_losers"] = df.loc[df["pnl"] <= 0, "holding_period"].mean()

        # Risk / Efficiency
        metrics["return_on_capital"] = df["pnl"].sum() / df["trade_value"].sum() if "trade_value" in df.columns and df["trade_value"].sum() != 0 else 0
        metrics["efficiency_ratio"] = df["pnl"].sum() / df["pnl"].abs().sum() if df["pnl"].abs().sum() != 0 else 0
        metrics["r_multiple_avg"] = (df["pnl"] / df["trade_value"]).mean() if "trade_value" in df.columns else 0
        metrics["downside_deviation"] = df.loc[df["pnl"] < 0, "pnl"].std()

        # Behavioral Insights
        if "trade_hour" in df.columns:
            metrics["trade_timing_bias"] = df["trade_hour"].corr(df["pnl"])
        if "volume" in df.columns and "trade_value" in df.columns:
            metrics["volume_following_behavior"] = df["volume"].corr(df["trade_value"])
        if "total_score" in df.columns:
            metrics["score_alignment_effectiveness"] = df["total_score"].corr(df["pnl"])
        if "pnl" in df.columns:
            avg_win = df.loc[df["pnl"] > 0, "pnl"].mean()
            avg_loss = df.loc[df["pnl"] < 0, "pnl"].mean()
            metrics["reward_to_risk_balance"] = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0


        return metrics

    # =========================================================
    # Core Metrics (unchanged)
    # =========================================================
    def calculate_total_pnl(self, df: pd.DataFrame) -> float:
        return float(df['pnl'].sum())

    def calculate_win_rate(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        winning_trades = len(df[df['pnl'] > 0])
        return float(winning_trades / len(df) * 100)

    def calculate_avg_win(self, df: pd.DataFrame) -> float:
        winning_trades = df[df['pnl'] > 0]['pnl']
        return float(winning_trades.mean()) if len(winning_trades) > 0 else 0.0

    def calculate_avg_loss(self, df: pd.DataFrame) -> float:
        losing_trades = df[df['pnl'] < 0]['pnl']
        return float(losing_trades.mean()) if len(losing_trades) > 0 else 0.0

    def calculate_profit_factor(self, df: pd.DataFrame) -> float:
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    def calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return 0.0
        returns = df['pnl'] / df['trade_value']
        if returns.std() == 0:
            return 0.0
        excess = returns.mean() - (self.risk_free_rate / self.trading_days)
        return float(excess / returns.std() * np.sqrt(self.trading_days))

    def calculate_sortino_ratio(self, df: pd.DataFrame) -> float:
        if len(df) < 2:
            return 0.0
        returns = df['pnl'] / df['trade_value']
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        if downside.std() == 0:
            return 0.0
        excess = returns.mean() - (self.risk_free_rate / self.trading_days)
        return float(excess / downside.std() * np.sqrt(self.trading_days))

    def calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        df_sorted = df.sort_values('trade_date')
        cum = df_sorted['pnl'].cumsum()
        run_max = cum.cummax()
        drawdown = cum - run_max
        return float(drawdown.min())

    def calculate_max_drawdown_pct(self, df: pd.DataFrame) -> float:
        if df.empty or df['pnl'].sum() == 0:
            return 0.0
        df_sorted = df.sort_values('trade_date')
        cum = df_sorted['pnl'].cumsum()
        run_max = cum.cummax().replace(0, np.nan)
        dd_pct = (cum - run_max) / run_max * 100
        return round(abs(float(dd_pct.min())), 2)

    def calculate_avg_trade_value(self, df: pd.DataFrame) -> float:
        return float(df['trade_value'].mean())

    def calculate_largest_win(self, df: pd.DataFrame) -> float:
        return float(df['pnl'].max()) if len(df) > 0 else 0.0

    def calculate_largest_loss(self, df: pd.DataFrame) -> float:
        return float(df['pnl'].min()) if len(df) > 0 else 0.0

    def calculate_consecutive_wins(self, df: pd.DataFrame) -> int:
        df_sorted = df.sort_values('trade_date')
        max_consec, cur = 0, 0
        for pnl in df_sorted['pnl']:
            if pnl > 0:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 0
        return int(max_consec)

    def calculate_consecutive_losses(self, df: pd.DataFrame) -> int:
        df_sorted = df.sort_values('trade_date')
        max_consec, cur = 0, 0
        for pnl in df_sorted['pnl']:
            if pnl < 0:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 0
        return int(max_consec)

    def calculate_avg_holding_period(self, df: pd.DataFrame) -> float:
        return float(df['holding_period_minutes'].mean()) if 'holding_period_minutes' in df.columns else 0.0

    def calculate_avg_trades_per_day(self, df: pd.DataFrame) -> float:
        days = df['trade_date'].dt.date.nunique()
        return float(len(df) / days) if days > 0 else 0.0

    def get_date_range(self, df: pd.DataFrame) -> str:
        start_date = df['trade_date'].min().strftime('%Y-%m-%d')
        end_date = df['trade_date'].max().strftime('%Y-%m-%d')
        return f"{start_date} to {end_date}"

    def get_trading_days(self, df: pd.DataFrame) -> int:
        return int(df['trade_date'].dt.date.nunique())

    # =========================================================
    # Extended: Positions/MTM snapshot
    # =========================================================
    def _compute_positions_snapshot(self, df: pd.DataFrame) -> Dict:
        """
        Build symbol-level net position, avg cost, current price (best-available),
        realised P&L (from 'pnl'), unrealised P&L (from open qty * (cur - cost) * side),
        and rich bucket analytics.
        """
        # Identify latest day in the file (used as "current day")
        latest_date = df['trade_date'].dt.date.max()

        # Symbol aggregates to figure out net quantities and avg costs
        # Treat BUY as +qty, SALE/SELL as -qty
        q_sign = df['transaction_type'].map({'BUY': 1, 'SALE': -1}).fillna(0)
        df['_signed_qty'] = df['quantity'] * q_sign

        # Weighted average cost only for entries that add inventory (BUY for net long, SALE for net short)
        def _avg_cost(series_price, series_qty):
            q = series_qty.sum()
            if q == 0:
                return 0.0
            # Weighted by ABS quantity so both directions can be handled
            w = np.abs(series_qty)
            return float((series_price * w).sum() / w.sum())

        # Last traded price fallback chain: 'ltp'/'current_price' columns > last trade price for symbol
        price_cols = [c for c in df.columns if c.lower() in ('ltp', 'current_price')]
        last_trade_price = df.sort_values('trade_date').groupby('symbol')['price'].last()

        # Compute symbol-level ledger
        sym = df.groupby('symbol').agg(
            net_qty=('_signed_qty', 'sum'),
            avg_cost=('price', lambda s: _avg_cost(s, df.loc[s.index, '_signed_qty'].clip(lower=0) +  # buys positive
                                     (-df.loc[s.index, '_signed_qty']).clip(lower=0))                  # sales positive for weighting
                      ),
            last_price=('price', 'last')
        )

        # Try to override last_price with explicit LTP/current_price if present (take last non-null)
        if price_cols:
            for col in price_cols:
                latest_px = df.sort_values('trade_date').groupby('symbol')[col].last()
                sym['last_price'] = latest_px.fillna(sym['last_price'])

        # If anything still missing, fallback to last_trade_price (already set)
        sym['last_price'] = sym['last_price'].fillna(last_trade_price)

        # Determine open positions (non-zero net_qty)
        sym['side'] = np.where(sym['net_qty'] > 0, 1, np.where(sym['net_qty'] < 0, -1, 0))
        sym_open = sym[sym['net_qty'] != 0].copy()

        # Compute unrealized P&L for open positions
        # For long: (last - cost) * qty; for short: (cost - last) * abs(qty)
        sym_open['invested_value'] = np.abs(sym_open['net_qty']) * sym_open['avg_cost']
        sym_open['unrealized'] = np.where(
            sym_open['side'] >= 0,
            (sym_open['last_price'] - sym_open['avg_cost']) * np.abs(sym_open['net_qty']),
            (sym_open['avg_cost'] - sym_open['last_price']) * np.abs(sym_open['net_qty'])
        )
        # % change on open positions (relative to cost)
        sym_open['pct_change'] = np.where(
            sym_open['invested_value'] > 0,
            sym_open['unrealized'] / sym_open['invested_value'] * 100,
            0.0
        )

        # Realized totals (from matched trades already in df['pnl'])
        total_realized = float(df['pnl'].sum())
        # Day MTM (realised on the latest date present in file)
        day_mtm_realized = float(df.loc[df['trade_date'].dt.date == latest_date, 'pnl'].sum())

        # Investment value (open)
        total_investment_value = float(sym_open['invested_value'].sum())
        total_unrealized = float(sym_open['unrealized'].sum())
        total_pnl_combined = total_realized + total_unrealized

        # Gainer / Loser (symbol-level net including unrealized if open)
        sym_all = []
        # Merge realized per symbol
        realized_by_symbol = df.groupby('symbol')['pnl'].sum()
        for symbol, row in sym.iterrows():
            net_open_unrl = 0.0
            if symbol in sym_open.index:
                net_open_unrl = float(sym_open.loc[symbol, 'unrealized'])
            sym_all.append((symbol, float(realized_by_symbol.get(symbol, 0.0) + net_open_unrl)))
        # Also include symbols that only had realized and are closed
        for symbol in realized_by_symbol.index:
            if symbol not in dict(sym_all):
                sym_all.append((symbol, float(realized_by_symbol[symbol])))

        gainers = [s for s, v in sym_all if v > 0]
        losers = [s for s, v in sym_all if v < 0]

        # Average gainer/loser %: explicit % using symbol-level % return
        # Define symbol % return as total_pnl_symbol / (sum(abs(buy_qty))*avg_cost) if open, else realised denominator:
        sym_pct = []
        for symbol in set(list(realized_by_symbol.index) + list(sym_open.index)):
            realized = float(realized_by_symbol.get(symbol, 0.0))
            if symbol in sym_open.index:
                inv = float(sym_open.loc[symbol, 'invested_value'])
                unrl = float(sym_open.loc[symbol, 'unrealized'])
                denom = inv if inv > 0 else np.nan
                total_sym = realized + unrl
            else:
                # fallback: denominator as sum of absolute trade_value for that symbol
                denom = float(df.loc[df['symbol'] == symbol, 'trade_value'].abs().sum()) or np.nan
                total_sym = realized
            pct = (total_sym / denom * 100) if (denom and denom > 0) else np.nan
            sym_pct.append((symbol, pct))

        gainer_pct = [p for s, p in sym_pct if s in gainers and p == p]
        loser_pct = [p for s, p in sym_pct if s in losers and p == p]
        avg_gainer_pct = float(np.mean(gainer_pct)) if gainer_pct else 0.0
        avg_loser_pct = float(np.mean(loser_pct)) if loser_pct else 0.0

        # Close Pos Booked SL %: only loss-making realised trades proportion
        realized_trades = df[df['pnl'] != 0]
        sl_trades = realized_trades[realized_trades['pnl'] < 0]
        close_pos_booked_sl_pct = float(len(sl_trades) / len(realized_trades) * 100) if len(realized_trades) > 0 else 0.0

        # Averages requested:
        # Avg Realized PL per stock = Total Realized / unique symbols traded
        uniq_sym_traded = realized_by_symbol.index.nunique()
        avg_realized_per_stock = total_realized / uniq_sym_traded if uniq_sym_traded > 0 else 0.0
        # Avg Unrealized per open stock = Total Unrealized / number of open symbols
        open_sym_count = sym_open.index.nunique()
        avg_unrealized_per_open_stock = total_unrealized / open_sym_count if open_sym_count > 0 else 0.0

        # Buckets on open positions by pct change
        buckets = self._build_open_buckets(sym_open)

        # Build positions list for report
        positions = []
        for symbol, r in sym_open.sort_values('pct_change').iterrows():
            positions.append({
                'symbol': symbol,
                'net_qty': int(r['net_qty']),
                'avg_cost': float(r['avg_cost']),
                'last_price': float(r['last_price']),
                'invested_value': float(r['invested_value']),
                'unrealized': float(r['unrealized']),
                'pct_change': float(r['pct_change'])
            })

        aggregates = {
            # headline totals
            'total_trades': int(len(df)),
            'day_mtm': day_mtm_realized,  # realised MTM for latest day in file
            'total_realized_pnl': total_realized,
            'total_unrealized_pnl': total_unrealized,
            'total_pnl_combined': total_pnl_combined,
            'total_investment_value_open': total_investment_value,

            # averages
            'avg_realized_pl_per_stock': avg_realized_per_stock,
            'avg_unrealized_pl_per_open_stock': avg_unrealized_per_open_stock,

            # SL %
            'close_pos_booked_sl_pct': close_pos_booked_sl_pct,

            # OPEN POSITION count (symbols)
            'open_positions_count': int(open_sym_count),

            # gainer/loser overview
            'avg_gainer_pct': avg_gainer_pct,
            'avg_loser_pct': avg_loser_pct,
        }

        gainer = {'count': len(gainers), 'list': sorted(gainers)}
        loser = {'count': len(losers), 'list': sorted(losers)}

        return {
            'aggregates': aggregates,
            'positions': positions,
            'buckets': buckets,
            'gainer': gainer,
            'loser': loser
        }

    def _build_open_buckets(self, sym_open: pd.DataFrame) -> Dict[str, Dict]:
        """
        Build requested open-position % change buckets with:
        - count, list of symbols, total unrealized value in bucket,
        - share of total MTM (sum of unrealized), and share of open positions count.
        """
        if sym_open.empty:
            labels = ['<0%', '0-5%', '5-10%', '10-20%', '20-40%', '40-60%', '60-80%', '>80%']
            return {k: {'count': 0, 'list': [], 'total_value': 0.0, 'mtm_share_pct': 0.0, 'open_pos_share_pct': 0.0} for k in labels}

        bins = [-np.inf, 0, 5, 10, 20, 40, 60, 80, np.inf]
        labels = ['<0%', '0-5%', '5-10%', '10-20%', '20-40%', '40-60%', '60-80%', '>80%']
        cat = pd.cut(sym_open['pct_change'], bins=bins, labels=labels, right=False)

        total_unrl = sym_open['unrealized'].sum()
        total_cnt = len(sym_open)

        out = {}
        for label in labels:
            bucket_rows = sym_open[cat == label]
            syms = bucket_rows.index.tolist()
            val = float(bucket_rows['unrealized'].sum())
            cnt = int(len(bucket_rows))
            out[label] = {
                'count': cnt,
                'list': syms,
                'total_value': val,
                'mtm_share_pct': float((val / total_unrl * 100) if total_unrl != 0 else 0.0),
                'open_pos_share_pct': float((cnt / total_cnt * 100) if total_cnt != 0 else 0.0)
            }
        return out

    def _build_pnl_timeline(self, df: pd.DataFrame) -> Dict[str, List]:
        """Cumulative P&L timeline for chart (unchanged behavior, daily granularity)."""
        df_sorted = df.sort_values('trade_date')
        daily = df_sorted.groupby(df_sorted['trade_date'].dt.date)['pnl'].sum()
        cum = daily.cumsum()
        return {'dates': [str(d) for d in cum.index], 'values': [float(v) for v in cum.values]}

    # =========================================================
    # Persona Trait Analysis (unchanged)
    # =========================================================
    def calculate_persona_traits(self, df: pd.DataFrame) -> Dict:
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

    # ---- trait helpers (unchanged from your version) ----
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
        return round(min(1.0, max(0.0, raw_score * 5)), 2)

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
        return round(max(0.1, 1 / (1 + perf_std / 100)), 2)

    def _calc_consistency(self, df: pd.DataFrame) -> float:
        pnl_cum = df["pnl"].cumsum()
        pnl_rolling = pnl_cum.rolling(window=5).std().fillna(0)
        volatility = pnl_rolling.mean()
        daily_mean = df.groupby(df["trade_date"].dt.date)["pnl"].mean().mean()
        return round(max(0.0, 1 - abs(volatility / (abs(daily_mean) + 1))), 2)

    def _calc_confidence(self, df: pd.DataFrame) -> float:
        df_sorted = df.sort_values("trade_date")
        pnl_shift = df_sorted["pnl"].shift(1)
        size_shift = df_sorted["quantity"]
        post_win_growth = size_shift[pnl_shift > 0].mean() / (size_shift.mean() + 1e-6)
        post_loss_growth = size_shift[pnl_shift < 0].mean() / (size_shift.mean() + 1e-6)
        confidence = post_win_growth - post_loss_growth
        return round(min(1.0, max(0.0, 0.5 + confidence * 0.5)), 2)

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
        return (
            f"Trader shows characteristics of a **{persona_type}**.\n\n"
            f"- Discipline: {traits['discipline_score']*100:.0f}%\n"
            f"- Emotional Control: {traits['emotional_control']*100:.0f}%\n"
            f"- Risk Appetite: {traits['risk_appetite']*100:.0f}%\n"
            f"- Patience: {traits['patience']*100:.0f}%\n"
            f"- Adaptability: {traits['adaptability']*100:.0f}%\n"
            f"- Consistency: {traits['consistency']*100:.0f}%\n"
            f"- Confidence: {traits['confidence']*100:.0f}%\n"
        )

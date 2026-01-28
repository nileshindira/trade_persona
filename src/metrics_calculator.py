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
from typing import Dict, List
import logging
# from scipy.stats import skew, kurtosis
from datetime import date, datetime




class TradingMetricsCalculator:
    """Calculate comprehensive trading metrics and trading persona traits + extended MTM/open-position analytics"""

    def __init__(self, config: Dict):
        self.config = config
        self.risk_free_rate = config['metrics']['risk_free_rate']
        self.trading_days = config['metrics']['trading_days_per_year']
        self.logger = logging.getLogger(__name__)


    def classify_instrument(self,symbol: str):
        s = str(symbol).upper().strip().replace(",", "")
        parts = s.split()

        INDEXES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}

        # OPTION
        if len(parts) >= 3 and parts[1] in ("CE", "PE"):

            # INDEX OPTION
            if parts[2] in INDEXES:
                return parts[2], "OPTION-INDEX"

            # EQUITY OPTION
            return parts[2], "OPTION-EQUITY"

        # EQUITY
        if parts[0] == "EQ":
            return parts[1], "EQUITY"

        if len(parts) == 1 and parts[0].isalpha():
            return parts[0], "EQUITY"

        # fallback
        return parts[0], "UNKNOWN"

    def compute_symbol_holding_days(self, df_sym):
        """
        Accepts df filtered for a single symbol.
        Returns list of holding durations (days) for each opened->closed cycle.
        """
        df_sym = df_sym.sort_values("trade_date")

        qty_stack = []  # (qty, date)
        hold_days = []

        for _, row in df_sym.iterrows():
            if row["transaction_type"] == "BUY":
                qty_stack.append([row["quantity"], row["trade_date"].date()])

            elif row["transaction_type"] in ("SALE", "SELL"):
                sell_qty = row["quantity"]
                sell_date = row["trade_date"].date()

                # FIFO matching
                while sell_qty > 0 and qty_stack:
                    buy_qty, buy_date = qty_stack[0]

                    matched = min(buy_qty, sell_qty)
                    sell_qty -= matched
                    qty_stack[0][0] -= matched

                    # holding days
                    hold_days.append((sell_date - buy_date).days)

                    if qty_stack[0][0] == 0:
                        qty_stack.pop(0)

        return hold_days

    # =========================================================
    # Public Entry
    # =========================================================
    def calculate_all_metrics(self, df: pd.DataFrame, pnl_csv :str) -> Dict:
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
            'pnl_timeline': self._build_pnl_timeline(pnl_csv),
        }

        # -------- Extended position & MTM analytics --------
        pos = self._compute_positions_snapshot(df)
        metrics.update(pos['aggregates'])       # realized/unrealized, investment, day MTM, averages, SL%
        metrics['open_positions'] = pos['positions']  # per-symbol open info
        metrics['closed_positions'] = pos['closed_positions']  # per-symbol open info
        metrics['buckets'] = pos['buckets']     # bucket analytics
        metrics['gainer'] = pos['gainer']       # counts + list of gainer symbols
        metrics['loser'] = pos['loser']         # counts + list of loser symbols
        metrics['symbol_details'] = pos['symbol_details']
        metrics['trader_type'] = pos['trader_type']


        # -------- Persona traits (existing) --------
        persona_traits = self.calculate_persona_traits(df)
        metrics.update(persona_traits)

        hard_flags = self.compute_hard_flags(metrics)
        metrics["_hard_flags"] = hard_flags

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
        metrics["pnl_volatility"] = self.pnl_df["MTM"].std()
        metrics["pnl_skewness"] = self.pnl_df["MTM"].skew()
        metrics["pnl_kurtosis"] = self.pnl_df["MTM"].kurt()
        metrics["pnl_kurtosis"] = self.pnl_df["MTM"].kurt()
        metrics["value_at_risk_95"] = self.pnl_df["MTM"].quantile(0.05)

        # --- 🧩 NEW: Behavioral Context Metrics ---
        # 1. Event Based Trading
        if "is_event" in df.columns:
            event_trades = df[df["is_event"] == True]
            metrics["event_trading_count"] = len(event_trades)
            metrics["event_trading_win_rate"] = self.calculate_win_rate(event_trades)
            metrics["event_trading_pnl"] = self.calculate_total_pnl(event_trades)
            
        # 2. News Based Trading
        if "is_news" in df.columns:
            news_trades = df[df["is_news"] == True]
            metrics["news_trading_count"] = len(news_trades)
            metrics["news_trading_win_rate"] = self.calculate_win_rate(news_trades)
            metrics["news_trading_pnl"] = self.calculate_total_pnl(news_trades)
            
            # Breakdown by News Category
            if "news_category" in df.columns:
                # Top 3 categories by count
                cat_counts = news_trades["news_category"].value_counts().head(3).to_dict()
                metrics["news_category_breakdown"] = cat_counts

        # 3. Volume Based Trading
        if "is_high_volume" in df.columns:
            high_vol_trades = df[df["is_high_volume"] == True]
            metrics["high_volume_trading_count"] = len(high_vol_trades)
            metrics["high_volume_trading_win_rate"] = self.calculate_win_rate(high_vol_trades)
            metrics["high_volume_trading_pnl"] = self.calculate_total_pnl(high_vol_trades)

        # 4. Market Behaviour (Trend Alignment)
        # Check alignment with Nifty (if available from enrichment)
        if "nifty50_pct_chg_1w" in df.columns:
            # Simple Trend Definition: Nifty > 0 AND Long Trade OR Nifty < 0 AND Short Trade
            # We use 'transaction_type' (BUY=Long, SALE=Short) for simplicity on entry
            def is_aligned(row):
                nifty_trend = row["nifty50_pct_chg_1w"]
                # Assuming BUY is bullish, SELL is bearish (simplified for single leg)
                # Ideally check position net delta, but usage of transaction_type is a proxy
                if pd.isna(nifty_trend): return False
                if row["transaction_type"] == "BUY" and nifty_trend > 0: return True
                if row["transaction_type"] in ["SALE", "SELL"] and nifty_trend < 0: return True
                return False

            df["_trend_aligned"] = df.apply(is_aligned, axis=1)
            aligned_trades = df[df["_trend_aligned"] == True]
            metrics["trend_aligned_win_rate"] = self.calculate_win_rate(aligned_trades)
            metrics["trend_alignment_score"] = (len(aligned_trades) / len(df) * 100) if len(df) > 0 else 0

        # Position / Holding behavior
        if "holding_period" in df.columns:
            metrics["avg_holding_period"] = df["holding_period"].mean()
            metrics["holding_period_volatility"] = df["holding_period"].std()
            metrics["avg_holding_period_winners"] = df.loc[df["pnl"] > 0, "holding_period"].mean()
            metrics["avg_holding_period_losers"] = df.loc[df["pnl"] <= 0, "holding_period"].mean()
        
        # --- 🧩 NEW: Holding Analytics Segments ---
        metrics["trade_analytics"] = self._calculate_holding_analytics(df)

        # Risk / Efficiency
        metrics["return_on_capital"] = self.cumulative_pnl / df["trade_value"].sum() if "trade_value" in df.columns and df["trade_value"].sum() != 0 else 0
        metrics["efficiency_ratio"] = self.cumulative_pnl / df["pnl"].abs().sum() if df["pnl"].abs().sum() != 0 else 0
        metrics["r_multiple_avg"] = (df["pnl"] / df["trade_value"]).mean() if "trade_value" in df.columns else 0
        metrics["downside_deviation"] = df.loc[df["pnl"] < 0, "pnl"].std()

        # Behavioral Insights
        if "trade_hour" in df.columns:
            th = pd.to_numeric(df["trade_hour"], errors="coerce")
            pnl = pd.to_numeric(df["pnl"], errors="coerce")

            # Filter valid rows
            mask = th.notna() & pnl.notna()

            if mask.sum() >= 2 and th[mask].nunique() > 1:
                metrics["trade_timing_bias"] = th[mask].corr(pnl[mask])
            else:
                metrics["trade_timing_bias"] = 0.0
        if "volume" in df.columns and "trade_value" in df.columns:
            metrics["volume_following_behavior"] = df["volume"].corr(df["trade_value"])
        if "total_score" in df.columns:
            metrics["score_alignment_effectiveness"] = df["total_score"].corr(df["pnl"])
        if "pnl" in df.columns:
            avg_win = self.pnl_df.loc[self.pnl_df["MTM"] > 0, "MTM"].mean()
            avg_loss = self.pnl_df.loc[self.pnl_df["MTM"] < 0, "MTM"].mean()
            metrics["reward_to_risk_balance"] = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0
            # ================================
            # NEW – Instrument Cluster Metrics (symbol → category)
            # ================================
            if 'symbol' in df.columns and 'trade_value' in df.columns:

                def _get_asset_cluster(s):
                    asset, kind = self.classify_instrument(s)
                    # kind can be:
                    #   EQUITY
                    #   OPTION-INDEX
                    #   OPTION-EQUITY
                    #   UNKNOWN
                    return kind

                df["_asset_kind"] = df["symbol"].apply(_get_asset_cluster)

                cluster_value = df.groupby("_asset_kind")["trade_value"].sum().astype(float)
                total_asset_val = float(cluster_value.sum())

                if total_asset_val > 0:
                    cluster_pct = (cluster_value / total_asset_val * 100).round(2)
                else:
                    cluster_pct = cluster_value * 0.0

                asset_clusters = [
                    {"asset_kind": cat, "value": float(pct)}
                    for cat, pct in cluster_pct.items()
                ]
            else:
                asset_clusters = []
            metrics["chart_data"]={}
            metrics["chart_data"]["asset_clusters"] = asset_clusters

            # ================================
            # NEW – asset_name cluster pie data (symbol -> asset_name)
            # ================================
            if 'symbol' in df.columns and 'trade_value' in df.columns:

                def _get_asset_name(s):
                    asset, kind = self.classify_instrument(s)
                    return asset

                df["_asset_name"] = df["symbol"].apply(_get_asset_name)

                name_value = df.groupby("_asset_name")["trade_value"].sum().astype(float)
                total_name_val = float(name_value.sum())

                if total_name_val > 0:
                    name_pct = (name_value / total_name_val * 100).round(2)
                else:
                    name_pct = name_value * 0.0

                asset_name_clusters = [
                    {"asset_name": name, "value": float(pct)}
                    for name, pct in name_pct.items()
                ]
            else:
                asset_name_clusters = []


            metrics["symbol_cluster"] = asset_name_clusters
            metrics["equity_trade_pct"] = float(cluster_pct.get("EQUITY", 0))
            metrics["option_equity_trade_pct"] = float(cluster_pct.get("OPTION-EQUITY", 0))
            metrics["option_index_trade_pct"] = float(cluster_pct.get("OPTION-INDEX", 0))

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

    def _bucket_from_pct(self, pct):
        if pct < 0: return "<0%"
        if pct < 5: return "0-5%"
        if pct < 10: return "5-10%"
        if pct < 20: return "10-20%"
        if pct < 40: return "20-40%"
        if pct < 60: return "40-60%"
        if pct < 80: return "60-80%"
        return ">80%"

    # =========================================================
    # Extended: Positions/MTM snapshot
    # =========================================================

    def _calculate_holding_analytics(self, df: pd.DataFrame) -> Dict:
        """
        Segment trades by holding period using 'holding_period_minutes' if available.
        Buckets:
          - Scalp: < 30 mins
          - Intraday: 30 mins to < 1 day (1440 mins)
          - Swing: 1 day to 7 days
          - Position: > 7 days
        """
        # Default structure
        segments = {
            "Scalp (<30m)": {"min": 0, "max": 30},
            "Intraday (30m-1d)": {"min": 30, "max": 1440},
            "Swing (1d-7d)": {"min": 1440, "max": 10080},
            "Position (>7d)": {"min": 10080, "max": float('inf')},
        }

        results = {}

        # 1. Check if column exists
        col = "holding_period_minutes"
        if col not in df.columns:
            # Try to calculate if missing but we have dates
            if "entry_time" in df.columns and "exit_time" in df.columns:
                try:
                    df[col] = (pd.to_datetime(df["exit_time"]) - pd.to_datetime(df["entry_time"])).dt.total_seconds() / 60
                except:
                    pass

        if col not in df.columns:
             for name in segments:
                results[name] = {
                    "trade_count": 0,
                    "win_rate": 0.0,
                    "avg_pnl": 0.0,
                    "total_pnl": 0.0
                }
             return {"holding_period_segments": results}

        # 2. Iterate buckets
        for name, limits in segments.items():
            mask = (df[col] >= limits["min"]) & (df[col] < limits["max"])
            subset = df[mask]
            
            count = len(subset)
            total_pnl = subset["pnl"].sum() if count > 0 else 0.0
            avg_pnl = total_pnl / count if count > 0 else 0.0
            
            # Win rate
            wins = len(subset[subset["pnl"] > 0])
            win_rate = (wins / count * 100) if count > 0 else 0.0

            results[name] = {
                "trade_count": count,
                "win_rate": float(win_rate),
                "avg_pnl": float(avg_pnl),
                "total_pnl": float(total_pnl)
            }

        return {"holding_period_segments": results}

    def _compute_positions_snapshot(self, df: pd.DataFrame) -> Dict:
        """
        Build symbol-level open/closed positions from already-classified df:
          - Uses position_status (OPEN/CLOSED)
          - Uses position_type (LONG-OPEN / SHORT-OPEN)
          - Computes open MTM using LTP/close (hooks for expiry/intrinsic logic)
          - Computes closed position aggregates and score lists
        """
        df = df.copy()



        # Safety guard: if data_processor hasn't added these (old files), fall back
        if "position_status" not in df.columns or "position_type" not in df.columns:
            # Legacy fallback: treat any net_qty != 0 as OPEN
            df["transaction_type"] = df["transaction_type"].astype(str).str.upper()

            def signed_qty(row):
                t = row["transaction_type"]
                if t == "BUY":
                    return row["quantity"]
                elif t in ("SALE", "SELL"):
                    return -row["quantity"]
                return 0.0

            df["_signed_qty"] = df.apply(signed_qty, axis=1)
            net = df.groupby("symbol")["_signed_qty"].sum().rename("symbol_net_qty")
            df = df.merge(net, left_on="symbol", right_index=True, how="left")
            df["position_status"] = df["symbol_net_qty"].apply(lambda q: "OPEN" if q != 0 else "CLOSED")
            df["position_type"] = df["symbol_net_qty"].apply(
                lambda q: "LONG-OPEN" if q > 0 else ("SHORT-OPEN" if q < 0 else "")
            )

        # Latest date in file -> for day MTM
        latest_date = df["trade_date"].dt.date.max()

        # Helper: get LTP / current price for each symbol
        def get_symbol_ltp(g: pd.DataFrame) -> float:
            # Prefer explicit LTP/current_price if present
            for col in g.columns:
                if col.lower() in ("ltp", "current_price"):
                    s = g.sort_values("trade_date")[col].dropna()
                    if len(s) > 0:
                        return float(s.iloc[-1])
            # Fallback to 'close' from DB
            if "close" in g.columns:
                s = g.sort_values("trade_date")["close"].dropna()
                if len(s) > 0:
                    return float(s.iloc[-1])
            # Fallback to last trade price
            s = g.sort_values("trade_date")["price"]
            return float(s.iloc[-1])

        # Symbol-level aggregation
        group = df.groupby("symbol")

        sym_rows = []
        for symbol, g in group:
            g = g.sort_values("trade_date")

            pos_status = g["position_status"].iloc[0]
            pos_type = g["position_type"].iloc[0]
            # BUY/SELL masks
            buy_mask = g["transaction_type"] == "BUY"
            sell_mask = g["transaction_type"].isin(["SALE", "SELL"])

            buy_qty = g.loc[buy_mask, "quantity"].sum()
            sell_qty = g.loc[sell_mask, "quantity"].sum()

            buy_value = (g.loc[buy_mask, "price"] * g.loc[buy_mask, "quantity"]).sum()
            sell_value = (g.loc[sell_mask, "price"] * g.loc[sell_mask, "quantity"]).sum()

            buy_price = float(buy_value / buy_qty) if buy_qty > 0 else 0.0
            sell_price = float(sell_value / sell_qty) if sell_qty > 0 else 0.0

            ltp = get_symbol_ltp(g)

            # Net qty = BUY - SELL
            net_qty = float(buy_qty - sell_qty)

            # Collect score lists (date-ordered)
            if "t_score" in g.columns:
                buy_t_scores = g.loc[buy_mask, "t_score"].dropna().tolist()
                sell_t_scores = g.loc[sell_mask, "t_score"].dropna().tolist()
            else:
                buy_t_scores, sell_t_scores = [], []

            if "f_score" in g.columns:
                buy_f_scores = g.loc[buy_mask, "f_score"].dropna().tolist()
                sell_f_scores = g.loc[sell_mask, "f_score"].dropna().tolist()
            else:
                buy_f_scores, sell_f_scores = [], []

            sym_rows.append({
                "symbol": symbol,
                "position_status": pos_status,
                "position_type": pos_type,
                "net_qty": net_qty,
                "buy_qty": float(buy_qty),
                "sell_qty": float(sell_qty),
                "buy_price": buy_price,
                "sell_price": sell_price,
                "ltp": ltp,
                "buy_t_scores": buy_t_scores,
                "buy_f_scores": buy_f_scores,
                "sell_t_scores": sell_t_scores,
                "sell_f_scores": sell_f_scores,
            })


        sym_df = pd.DataFrame(sym_rows).set_index("symbol")
        realized_by_symbol = df.groupby("symbol")["pnl"].sum()



        # =========================
        # OPEN POSITIONS (LONG-OPEN / SHORT-OPEN)
        # =========================
        sym_open = sym_df[sym_df["position_status"] == "OPEN"].copy()

        # Compute invested value, unrealized, mtm_pct
        invested_values = []
        unrealized_values = []
        pct_changes = []

        for symbol, row in sym_open.iterrows():
            net_qty = row["net_qty"]
            pos_type = row["position_type"]
            ltp = row["ltp"]
            # if len(symbol.split(' ') >2)
            expiry_str = symbol.split(' ')[3]
            strike = symbol.split(' ')[-1]
            option_type = symbol.split(' ')[1]

            # Convert string → date object
            expiry_date = datetime.strptime(expiry_str, "%d%b%Y").date()

            today_date = date.today()
            if pos_type == "LONG-OPEN":
                # Long open qty is net_qty > 0
                open_qty = abs(net_qty)
                entry_price = row["buy_price"]
                invested = open_qty * entry_price
                if today_date > expiry_date:
                    if option_type == 'CE':
                        if float(strike) > ltp:
                            unreal = -1 * invested
                        else:
                            unreal = ((strike-ltp) * open_qty) - invested
                    elif option_type == 'PE':
                        if float(strike) < ltp:
                            unreal = -1 * open_qty
                        else:
                            unreal = ((ltp-strike) * open_qty) - invested
                    else:
                        unreal = ((entry_price - ltp) * open_qty) - invested
                else:
                    # MTM pre-expiry
                    unreal = (ltp - entry_price) * open_qty

            elif pos_type == "SHORT-OPEN":
                # Short open qty is |net_qty| (more SELL than BUY)
                open_qty = abs(net_qty)
                entry_price = row["sell_price"]
                invested = open_qty * entry_price
                if today_date > expiry_date:
                    if option_type == 'CE':
                        if float(strike) > float(ltp):
                            unreal = invested
                        else:
                            unreal = invested - ((float(strike)-float(ltp)) * open_qty)
                    elif option_type == 'PE':
                        if float(strike) < float(ltp):
                            unreal = invested
                        else:
                            unreal = invested - ((float(ltp)-float(strike)) * open_qty)
                    else:
                        unreal = (entry_price - float(ltp)) * open_qty
                else:
                    # 👉 MTM from LTP (short premium trade)
                    unreal = (entry_price - ltp) * open_qty

            else:
                open_qty = 0
                invested = 0.0
                unreal = 0.0

            invested_values.append(invested)
            unrealized_values.append(unreal)
            pct_changes.append(float(unreal / invested * 100) if invested > 0 else 0.0)

        sym_open["invested_value"] = invested_values
        sym_open["unrealized"] = unrealized_values
        sym_open["pct_change"] = pct_changes

        # =========================
        # CANONICAL POSITIONS DF (UI SINGLE SOURCE OF TRUTH)
        # =========================
        positions_df = []

        for symbol, r in sym_df.iterrows():
            realized = float(realized_by_symbol.get(symbol, 0.0))

            if symbol in sym_open.index:
                o = sym_open.loc[symbol]
                unrealized = float(o["unrealized"])
                invested = float(o["invested_value"])
                return_pct = float(o["pct_change"])
                status = "OPEN"
            else:
                unrealized = 0.0
                df_sym = df[df["symbol"] == symbol]
                invested = float(df_sym["trade_value"].abs().sum())
                return_pct = (realized / invested * 100) if invested > 0 else 0.0
                status = "CLOSED"

            total_pnl = realized + unrealized

            positions_df.append({
                "symbol": symbol,
                "position_status": status,
                "position_type": r["position_type"],
                "qty": int(abs(r["net_qty"])),
                "invested_value": invested,
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "total_pnl": total_pnl,
                "return_pct": return_pct,
                "bucket": self._bucket_from_pct(return_pct),
                "is_gainer": total_pnl > 0,
                "is_loser": total_pnl < 0,
            })

        # ======== OPEN POSITIONS LIST FOR UI ========
        positions = []
        for symbol, r in sym_open.sort_values("pct_change").iterrows():
            # For LONG-OPEN we take last buy-side scores; for SHORT-OPEN last sell-side scores
            if r["position_type"] == "LONG-OPEN":
                t_score = r["buy_t_scores"][-1] if r["buy_t_scores"] else 0
                f_score = r["buy_f_scores"][-1] if r["buy_f_scores"] else 0
                txn = "BUY"
            else:  # SHORT-OPEN
                t_score = r["sell_t_scores"][-1] if r["sell_t_scores"] else 0
                f_score = r["sell_f_scores"][-1] if r["sell_f_scores"] else 0
                txn = "SELL"

            positions.append({
                "symbol": symbol,
                "transaction_type": txn,  # BUY for LONG-OPEN, SELL for SHORT-OPEN
                "rate": float(r["buy_price"] if txn == "BUY" else r["sell_price"]),
                "qty": int(abs(r["net_qty"])),
                "value": float(r["invested_value"]),
                "unrealized": float(r["unrealized"]),
                "mtm_pct": float(r["pct_change"]),
                "holding_days": 0,  # TODO: wire from pair_trades if you want true holding days
                "t_score": float(t_score),
                "f_score": float(f_score),
            })

        # =========================
        # CLOSED POSITIONS (your new required schema)
        # =========================
        sym_closed = sym_df[sym_df["position_status"] == "CLOSED"].copy()

######################################################################
        all_days = []

        for symbol, g in df.groupby("symbol"):
            all_days.extend(self.compute_symbol_holding_days(g))

        if not all_days:
            trader_type = "NO CLOSED TRADES"
        else:
            import numpy as np

            all_days = np.array(all_days)

            pct_intraday = np.mean(all_days == 0) * 100
            pct_positional = np.mean((all_days > 0) & (all_days <= 10)) * 100
            pct_longterm = np.mean(all_days > 30) * 100

            # Priority based on user's rule
            if pct_intraday >= 70:
                trader_type = "INTRADAY TRADER"
            elif pct_positional >= 70:
                trader_type = "POSITIONAL TRADER"
            elif pct_longterm >= 70:
                trader_type = "LONG TERM TRADER"
            else:
                # Mixed → choose max % bucket
                max_bucket = max(
                    [("INTRADAY", pct_intraday),
                     ("POSITIONAL", pct_positional),
                     ("LONGTERM", pct_longterm)],
                    key=lambda x: x[1]
                )[0]

                trader_type = f"MIXED ({max_bucket})"
######################################################################
        closed_positions = []
        for symbol, r in sym_closed.iterrows():
            closed_positions.append({
                "symbol": symbol,
                "buy_price": float(r["buy_price"]),
                "buy_qty": float(r["buy_qty"]),
                "sell_price": float(r["sell_price"]),
                "sell_qty": float(r["sell_qty"]),
                "realized_pnl": float(realized_by_symbol.get(symbol, 0.0)),
                "buy_t_scores": r["buy_t_scores"],
                "buy_f_scores": r["buy_f_scores"],
                "sell_t_scores": r["sell_t_scores"],
                "sell_f_scores": r["sell_f_scores"],
            })

        # =========================
        # Totals, Gainers/Losers & Buckets
        # =========================
        total_realized = float(realized_by_symbol.sum())
        total_unrealized = float(sym_open["unrealized"].sum())
        total_pnl_combined = self.cumulative_pnl
        total_investment_value = float(sym_open["invested_value"].sum())

        # Day MTM = realized pnl on latest trading date in file
        day_mtm_realized = self.last_pnl

        open_sym_count = int(sym_open.shape[0])
        avg_realized_per_stock = total_realized / realized_by_symbol.index.nunique() if realized_by_symbol.index.nunique() > 0 else 0.0
        avg_unrealized_per_open_stock = total_unrealized / open_sym_count if open_sym_count > 0 else 0.0

        # Gainer / loser classification including unrealized for open

        gainers = [p["symbol"] for p in positions_df if p["is_gainer"]]
        losers = [p["symbol"] for p in positions_df if p["is_loser"]]

        gainer = {"count": len(gainers), "list": sorted(gainers)}
        loser = {"count": len(losers), "list": sorted(losers)}

        # Avg gainer/loser % using pct_change/open returns if available
        sym_pct = []
        for symbol, r in sym_df.iterrows():
            if symbol in sym_open.index:
                denom = sym_open.loc[symbol, "invested_value"]
                total_sym = realized_by_symbol.get(symbol, 0.0) + sym_open.loc[symbol, "unrealized"]
            else:
                denom = df.loc[df["symbol"] == symbol, "trade_value"].abs().sum()
                total_sym = realized_by_symbol.get(symbol, 0.0)

            pct = (total_sym / denom * 100) if denom and denom > 0 else float("nan")
            sym_pct.append((symbol, pct))

        gainer_pct = [p for (s, p) in sym_pct if s in gainers and p == p]
        loser_pct = [p for (s, p) in sym_pct if s in losers and p == p]
        avg_gainer_pct = float(np.mean(gainer_pct)) if gainer_pct else 0.0
        avg_loser_pct = float(np.mean(loser_pct)) if loser_pct else 0.0

        # Close Pos Booked SL %: proportion of losing realized trades
        realized_trades = df[df["pnl"] != 0]
        sl_trades = realized_trades[realized_trades["pnl"] < 0]
        close_pos_booked_sl_pct = float(len(sl_trades) / len(realized_trades) * 100) if len(
            realized_trades) > 0 else 0.0

        # Buckets on open positions by pct change
        buckets = self._build_open_buckets(sym_open) if not sym_open.empty else self._build_open_buckets(sym_open)

        aggregates = {
            "total_trades": int(len(df)),
            "day_mtm": day_mtm_realized,
            "total_realized_pnl": total_realized,
            "total_unrealized_pnl": total_unrealized,
            "total_pnl_combined": total_pnl_combined,
            "total_investment_value_open": total_investment_value,
            "avg_realized_pl_per_stock": avg_realized_per_stock,
            "avg_unrealized_pl_per_open_stock": avg_unrealized_per_open_stock,
            "close_pos_booked_sl_pct": close_pos_booked_sl_pct,
            "open_positions_count": open_sym_count,
            "avg_gainer_pct": avg_gainer_pct,
            "avg_loser_pct": avg_loser_pct,
        }

        # Build symbol_details (for side panels)
        symbol_details = {}
        for symbol, r in sym_df.iterrows():
            if symbol in sym_open.index:
                # Use open metrics
                o = sym_open.loc[symbol]
                net_qty = o["net_qty"]
                entry_price = o["buy_price"] if o["position_type"] == "LONG-OPEN" else o["sell_price"]
                value = o["invested_value"]
                pnl = o["unrealized"]
                return_pct = o["pct_change"]
            else:
                # Use closed metrics
                df_sym = df[df["symbol"] == symbol]
                net_qty = 0
                entry_price = r["buy_price"]
                value = df_sym["trade_value"].abs().sum()
                pnl = realized_by_symbol.get(symbol, 0.0)
                return_pct = (pnl / value * 100) if value > 0 else 0.0

            symbol_details[symbol] = {
                "symbol": symbol,
                "buy_rate": float(r["buy_price"]),
                "sell_rate": float(r["sell_price"]),
                "qty": int(max(r["buy_qty"], r["sell_qty"])),
                "value": float(value),
                "pnl": float(pnl),
                "days": 0,  # TODO: derive from pair_trades if needed
                "return_pct": float(return_pct),
                "buy_t_score": r["buy_t_scores"][0] if r["buy_t_scores"] else 0,
                "buy_f_score": r["buy_f_scores"][0] if r["buy_f_scores"] else 0,
                "sell_t_score": r["sell_t_scores"][0] if r["sell_t_scores"] else 0,
                "sell_f_score": r["sell_f_scores"][0] if r["sell_f_scores"] else 0,
            }

        gainer = {"count": len(gainers), "list": sorted(gainers)}
        loser = {"count": len(losers), "list": sorted(losers)}

        return {
            "aggregates": aggregates,
            "positions_df": positions_df,  # ✅ CANONICAL
            "positions": positions,  # legacy open-only (safe)
            "closed_positions": closed_positions,
            "buckets": buckets,
            "gainer": gainer,
            "loser": loser,
            "symbol_details": symbol_details,
            "trader_type": trader_type,
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

    def _build_pnl_timeline(self, csv_path: str) -> Dict[str, List]:
        df = pd.read_csv(csv_path)
        self.pnl_df = df
        df = df.rename(columns={'Date': 'trade_date', 'MTM': 'pnl'})
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        daily = df.groupby(df['trade_date'].dt.date)['pnl'].sum()
        cumulative = daily.cumsum()
        self.last_pnl = df["pnl"].iloc[-1]
        self.cumulative_pnl = float(cumulative.iloc[-1]) if len(cumulative) else 0.0
        return {
            'dates': [str(d) for d in cumulative.index],
            'values': [float(v) for v in cumulative.values]
        }
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

    def compute_hard_flags(self, metrics: Dict) -> Dict:
        trader_type = metrics.get("trader_type", "MIXED")
        dd = metrics.get("max_drawdown_pct", 0)
        avg_trades = metrics.get("avg_trades_per_day", 0)
        pf = metrics.get("profit_factor", 0)
        emo = metrics.get("persona_traits", {}).get("emotional_control", 0)

        # --- Drawdown thresholds ---
        dd_limit = {
            "INTRADAY TRADER": 20,
            "POSITIONAL TRADER": 30,
            "LONG TERM TRADER": 35,
            "MIXED": 25
        }.get(trader_type, 25)

        # --- Overtrading thresholds ---
        trade_limit = {
            "INTRADAY TRADER": 12,
            "POSITIONAL TRADER": 3,
            "LONG TERM TRADER": 0.5,
            "MIXED": 6
        }.get(trader_type, 6)

        hard_flags = {
            "capital_at_risk_high": dd > dd_limit,
            "overtrading": avg_trades > trade_limit,
            "negative_expectancy": pf < 1,
            "fragile_edge": 1 <= pf < 1.3,
            "emotional_instability": emo < 0.4,
            "severe_emotional_trading": emo < 0.3,
        }

        return hard_flags

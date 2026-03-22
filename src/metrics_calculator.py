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
import os
# from scipy.stats import skew, kurtosis
from datetime import date, datetime, timedelta
from sqlalchemy import create_engine




class TradingMetricsCalculator:
    """Calculate comprehensive trading metrics and trading persona traits + extended MTM/open-position analytics"""

    def __init__(self, config: Dict):
        self.config = config
        self.risk_free_rate = config['metrics']['risk_free_rate']
        self.trading_days = config['metrics']['trading_days_per_year']
        self.logger = logging.getLogger(__name__)

        # Load sector mapping from local CSV
        self.sector_map = {}
        try:
            mapping_path = "/home/system-4/PycharmProjects/trade_persona/industry_sector_detail - All.csv"
            if os.path.exists(mapping_path):
                map_df = pd.read_csv(mapping_path)
                if "Symbol" in map_df.columns and "Industry_O" in map_df.columns:
                    # Map Symbol -> Industry_O
                    self.sector_map = dict(zip(
                        map_df["Symbol"].astype(str).str.upper().str.strip(), 
                        map_df["Industry_O"].astype(str).str.strip()
                    ))
                    self.logger.info(f"Successfully loaded {len(self.sector_map)} sector mappings.")
                else:
                    self.logger.warning(f"Sector mapping file at {mapping_path} missing 'Symbol' or 'Industry_O'.")
            else:
                self.logger.warning(f"Sector mapping file not found at {mapping_path}")
        except Exception as e:
            self.logger.error(f"Error loading sector mapping: {e}")

    def _map_sectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map symbols to sectors using the loaded mapping file."""
        if not self.sector_map:
            if "industry" not in df.columns:
                df["industry"] = "Other Sector"
            return df
        
        def get_sector(symbol):
            base_sym, _ = self.classify_instrument(symbol)
            return self.sector_map.get(base_sym.upper().strip(), "Other Sector")

        df["industry"] = df["symbol"].apply(get_sector)
        return df


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
    def calculate_all_metrics(self, df: pd.DataFrame, pnl_csv :str, nifty_data: Dict = None) -> Dict:
        """Calculate all trading metrics and persona traits with extended analytics."""
        df = df.copy()
        
        # Populate industry column using mapping file
        df = self._map_sectors(df)

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
        # Filter to only "traded" rows (matched trades with non-zero pnl)
        # for metrics that should reflect actual trade decisions
        traded_df = df[df['pnl'] != 0]  # Exclude unmatched open legs
        metrics = {
            'total_trades': len(traded_df),  # Actual trade count, not CSV rows
            'total_rows': len(df),  # Raw row count for reference
            'total_pnl': self.calculate_total_pnl(df),
            'win_rate': self.calculate_win_rate(df),
            'overall_win_rate': self.calculate_overall_win_rate(df),
            'avg_win': self.calculate_avg_win(df),
            'avg_loss': self.calculate_avg_loss(df),
            'avg_win_pct_of_all_wins': self.calculate_avg_win_pct(df),
            'avg_loss_pct_of_all_losses': self.calculate_avg_loss_pct(df),
            'multivariate_pattern_analysis': self._compute_multivariate_pattern_analysis(df),
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
            'pnl_timeline': self._build_pnl_timeline(pnl_csv, nifty_data),
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
        if "pnl" in self.pnl_df.columns:
            metrics["pnl_volatility"] = self.pnl_df["pnl"].std()
            metrics["pnl_skewness"] = self.pnl_df["pnl"].skew()
            metrics["pnl_kurtosis"] = self.pnl_df["pnl"].kurt()
            metrics["value_at_risk_95"] = self.pnl_df["pnl"].quantile(0.05)
        elif "MTM" in self.pnl_df.columns:
            metrics["pnl_volatility"] = self.pnl_df["MTM"].std()
            metrics["pnl_skewness"] = self.pnl_df["MTM"].skew()
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

        # 3.5. Market Behaviour / Chart Quality (Good vs Bad Charts)
        # Ensure metrics are initialized
        metrics.update({
            "good_chart_trading_count": 0, "good_chart_trading_win_rate": 0, "good_chart_trading_pnl": 0,
            "bad_chart_trading_count": 0, "bad_chart_trading_win_rate": 0, "bad_chart_trading_pnl": 0,
            "chart_behavior_breakdown": []
        })

        # Handle potential column name variations
        col = "chart_charts" if "chart_charts" in df.columns else ("market_behaviour" if "market_behaviour" in df.columns else None)
        
        if col:
            # Good Charts
            good_chart_trades = df[df[col].astype(str).str.contains("Good|High", case=False, na=False)]
            metrics["good_chart_trading_count"] = len(good_chart_trades)
            metrics["good_chart_trading_win_rate"] = self.calculate_win_rate(good_chart_trades)
            metrics["good_chart_trading_pnl"] = self.calculate_total_pnl(good_chart_trades)
            
            # Bad Charts
            bad_chart_trades = df[df[col].astype(str).str.contains("Bad", case=False, na=False)]
            metrics["bad_chart_trading_count"] = len(bad_chart_trades)
            metrics["bad_chart_trading_win_rate"] = self.calculate_win_rate(bad_chart_trades)
            metrics["bad_chart_trading_pnl"] = self.calculate_total_pnl(bad_chart_trades)

            # Chart Behavior Pie Data
            metrics["chart_behavior_breakdown"] = [
                {"label": "Good Charts", "value": len(good_chart_trades)},
                {"label": "Bad Charts", "value": len(bad_chart_trades)},
                {"label": "Neutral/Other", "value": len(df) - len(good_chart_trades) - len(bad_chart_trades)}
            ]

        # 3.6. ATH / ATL Context
        # Ensure metrics are initialized
        metrics.update({
            "ath_trading_count": 0, "ath_trading_win_rate": 0, "ath_trading_pnl": 0,
            "atl_trading_count": 0, "atl_trading_win_rate": 0, "atl_trading_pnl": 0
        })

        # Fallback for ATH flags
        if "is_alltime_high" not in df.columns and "dist_from_52w_high_pct" in df.columns:
            # If within 0.5% of 52w high, consider it a proxy for ATH/High price
            df["is_alltime_high"] = df["dist_from_52w_high_pct"] >= -0.5
        elif "is_alltime_high" not in df.columns and "is_52week_high" in df.columns:
            df["is_alltime_high"] = df["is_52week_high"]

        if "is_alltime_high" in df.columns:
            ath_trades = df[df["is_alltime_high"] == True]
            metrics["ath_trading_count"] = len(ath_trades)
            metrics["ath_trading_win_rate"] = self.calculate_win_rate(ath_trades)
            metrics["ath_trading_pnl"] = self.calculate_total_pnl(ath_trades)

        if "is_alltime_low" not in df.columns and "is_52week_low" in df.columns:
             df["is_alltime_low"] = df["is_52week_low"]

        if "is_alltime_low" in df.columns:
            atl_trades = df[df["is_alltime_low"] == True]
            metrics["atl_trading_count"] = len(atl_trades)
            metrics["atl_trading_win_rate"] = self.calculate_win_rate(atl_trades)
            metrics["atl_trading_pnl"] = self.calculate_total_pnl(atl_trades)

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
            pcol = "pnl" if "pnl" in self.pnl_df.columns else "MTM"
            avg_win = self.pnl_df.loc[self.pnl_df[pcol] > 0, pcol].mean()
            avg_loss = self.pnl_df.loc[self.pnl_df[pcol] < 0, pcol].mean()
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

            # --- NEW: Sector-based Segregation (Value Distribution) ---
            if 'industry' in df.columns and 'trade_value' in df.columns:
                sector_value = df.groupby("industry")["trade_value"].sum().astype(float)
                total_val = float(sector_value.sum())
                if total_val > 0:
                    sector_val_pct = (sector_value / total_val * 100).round(2)
                    metrics["sector_segregation"] = [
                        {"label": s, "value": float(p)}
                        for s, p in sector_val_pct.items()
                    ]
                else:
                    metrics["sector_segregation"] = []

        # ================================
        # Execution Efficiency (Entry/Exit) & Timing
        # ================================
        if all(col in df.columns for col in ["high", "low", "price", "pnl", "quantity"]):
            def calc_diagnostics(row):
                r = row["high"] - row["low"]
                if r <= 0: return 0.5, 0.0
                
                if row["transaction_type"] == "BUY":
                    # Proximity to Low (0.0 = bought at exact low, 1.0 = bought at exact high)
                    eff = (row["price"] - row["low"]) / r
                    return eff, 0.0
                else: # SALE / SELL
                    # Proximity to High (0.0 = sold at exact high, 1.0 = sold at exact low)
                    eff = (row["high"] - row["price"]) / r
                    
                    # Giveback: how much of the day's potential move did we leave on the table?
                    try:
                        pnl_per_qty = row["pnl"] / (row["quantity"] + 1e-6)
                        entry_cost = row["price"] - pnl_per_qty
                        
                        potential_profit = row["high"] - entry_cost
                        actual_profit = row["pnl"]  # Use the actual PnL
                        
                        if potential_profit > 0 and actual_profit > 0:
                            giveback = (row["high"] - row["price"]) / potential_profit
                        else:
                            giveback = 0.0
                    except:
                        giveback = 0.0
                    return eff, giveback
            
            diag_results = df.apply(calc_diagnostics, axis=1)
            df["entry_efficiency"] = diag_results.apply(lambda x: x[0])
            df["profit_giveback_raw"] = diag_results.apply(lambda x: x[1])
            
            buy_df = df[df["transaction_type"] == "BUY"]
            sell_df = df[df["transaction_type"].isin(["SALE", "SELL"])]

            metrics["avg_entry_efficiency"] = float(buy_df["entry_efficiency"].mean()) if not buy_df.empty else 0.5
            metrics["avg_exit_efficiency"] = float(sell_df["entry_efficiency"].mean()) if not sell_df.empty else 0.5
            metrics["avg_profit_giveback"] = float(sell_df["profit_giveback_raw"].mean()) if not sell_df.empty else 0.0
            
            # Since efficiency is now 'proximity' (0 = best), invert it for the 0-100 score
            metrics["timing_skill_score"] = float((1.0 - metrics["avg_entry_efficiency"]) * 100)
            metrics["exit_quality_score"] = float((1.0 - metrics["avg_exit_efficiency"]) * 100)

        # ================================
        # NEW – Market Behaviour & Industry Distributions for Pie Charts
        # ================================
        if "market_behaviour" in df.columns:
            # Check if column has data
            if not df["market_behaviour"].dropna().empty:
                mb_counts = df["market_behaviour"].value_counts()
                metrics["market_behaviour_distribution"] = [
                    {"label": str(k), "value": int(v)}
                    for k, v in mb_counts.items()
                ]
            else:
                metrics["market_behaviour_distribution"] = []

        if "industry" in df.columns:
            # Check if column has data
            if not df["industry"].dropna().empty:
                ind_counts = df["industry"].value_counts()
                metrics["industry_distribution"] = [
                    {"label": str(k), "value": int(v)}
                    for k, v in ind_counts.items()
                ]
            else:
                metrics["industry_distribution"] = []

        # ==================================
        # 🧩 NEW – Evidence Packs for LLM Grounding
        # ==================================
        metrics["evidence_packs"] = self._extract_evidence_packs(df)

        # ==================================
        # 🆕 Sectoral Profit Analysis
        # ==================================
        metrics["sectoral_analysis"] = self._compute_sectoral_analysis(df)

        # ==================================
        # 🆕 Option Writer vs Buyer Detection
        # ==================================
        metrics["option_strategy"] = self._detect_option_strategy(df)

        # ==================================
        # 🆕 PnL Reconciliation Check (FIFO vs MTM)
        # ==================================
        fifo_pnl = metrics.get('total_pnl', 0)
        mtm_pnl = self.cumulative_pnl
        if mtm_pnl != 0:
            pnl_gap_pct = abs(fifo_pnl - mtm_pnl) / max(abs(fifo_pnl), abs(mtm_pnl), 1) * 100
        else:
            pnl_gap_pct = 0
        metrics["pnl_reconciliation"] = {
            "fifo_pnl": round(fifo_pnl, 2),
            "mtm_pnl": round(mtm_pnl, 2),
            "gap_pct": round(pnl_gap_pct, 1),
            "is_aligned": pnl_gap_pct < 10,
        }
        if pnl_gap_pct > 10:
            self.logger.warning(
                f"⚠️ PnL MISMATCH: FIFO={fifo_pnl:,.0f} vs MTM={mtm_pnl:,.0f} ({pnl_gap_pct:.1f}% diff)"
            )

        # ==================================
        # 🆕 Strategy Inference & Behavioral Simulations
        # ==================================
        metrics["strategy_inference"] = self._infer_trading_strategy(df, metrics)
        metrics["what_if_analysis"] = self._simulate_what_if(df)
        metrics["loss_patterns"] = self._mine_loss_patterns(df)
        metrics["recent_market_context"] = self._get_recent_market_context(df)

        # ==================================
        # 🎯 Phase 2 – Persona Depth Layer
        # ==================================
        metrics["archetype"] = self._classify_archetype(metrics)
        metrics["trade_dna"] = self._compute_trade_dna(df, metrics)
        metrics["behavioral_pressure_map"] = self._compute_pressure_map(df)
        metrics["behavioral_consistency_score"] = self._compute_consistency_score(df, metrics)
        metrics["emotional_leakage_index"] = self._compute_emotional_leakage(metrics)

        return metrics


    # =================================================================
    # PHASE 2 — PERSONA DEPTH METHODS
    # =================================================================

    def _classify_archetype(self, metrics: Dict) -> Dict:
        """Rule-based archetype classification from calculated metrics."""
        win_rate    = metrics.get("win_rate", 0)
        hold_min    = metrics.get("avg_holding_period", 0)
        total_t     = metrics.get("total_trades", 1)
        fomo        = metrics.get("loss_patterns", {}).get("fomo_trades", 0)
        opt_strat   = metrics.get("option_strategy", {}).get("strategy_type", "")
        pnl_vol     = metrics.get("pnl_volatility", 0)
        consec_loss = metrics.get("consecutive_losses", 0)
        news_count  = metrics.get("evidence_packs", {}).get("fomo_entries", [])

        fomo_count  = metrics.get("evidence_packs", {}).get("fomo_count", 0)
        fomo_ratio = fomo_count / max(total_t, 1)
        hold_days  = hold_min / 60 / 24  # convert minutes to days

        # Priority order — first matching rule wins
        if opt_strat == "OPTION_WRITER" and win_rate > 55:
            return {"name": "The Premium Farmer", "code": "PREMIUM_FARMER",
                    "tagline": "Systematic income collector via option selling", "icon": "🌾",
                    "family": "Income", "confidence": 0.88}
        if win_rate > 72 and hold_min < 90 and fomo_ratio < 0.1:
            return {"name": "The Precision Sniper", "code": "PRECISION_SNIPER",
                    "tagline": "High-accuracy, low-volume, surgical execution", "icon": "🎯",
                    "family": "Precision", "confidence": 0.85}
        if fomo_ratio > 0.15 or (win_rate < 50 and pnl_vol > 10000):
            return {"name": "The FOMO Chaser", "code": "FOMO_CHASER",
                    "tagline": "Reactive, impulse-driven entry patterns", "icon": "🏃",
                    "family": "Reactive", "confidence": 0.80}
        if hold_days > 2 and win_rate > 55:
            return {"name": "The Patient Swing Trader", "code": "SWING_TRADER",
                    "tagline": "Multi-day positional plays with patience", "icon": "📅",
                    "family": "Swing", "confidence": 0.82}
        if hold_min < 45 and total_t > 500:
            return {"name": "The Momentum Scalper", "code": "MOMENTUM_SCALPER",
                    "tagline": "High-frequency short holds targeting quick moves", "icon": "⚡",
                    "family": "Momentum", "confidence": 0.78}
        if len(news_count) > 3 and win_rate > 58:
            return {"name": "The Contrarian Oracle", "code": "CONTRARIAN_ORACLE",
                    "tagline": "News-triggered contra trades with edge", "icon": "🔮",
                    "family": "Contrarian", "confidence": 0.75}
        if consec_loss > 6 and pnl_vol > 15000:
            return {"name": "The Chaos Trader", "code": "CHAOS_TRADER",
                    "tagline": "Unpredictable patterns, high variance profile", "icon": "🌀",
                    "family": "Chaotic", "confidence": 0.70}
        # Default fallback — overnight mean-reversion
        return {"name": "The Overnight Oracle", "code": "OVERNIGHT_ORACLE",
                "tagline": "Overnight mean-reversion with a high win rate", "icon": "🌙",
                "family": "Swing", "confidence": 0.72}

    def _compute_trade_dna(self, df: pd.DataFrame, metrics: Dict) -> Dict:
        """Derive the trader's behavioral fingerprint from trade data."""
        hold_min  = metrics.get("avg_holding_period", 0)
        intraday  = metrics.get("intraday_vs_overnight", {})
        opt_strat = metrics.get("option_strategy", {})
        t_skill   = metrics.get("timing_skill_score", 0)

        # Preferred session
        intra_cnt = intraday.get("intraday", {}).get("count", 0) if isinstance(intraday, dict) else 0
        over_cnt  = intraday.get("overnight", {}).get("count", 0) if isinstance(intraday, dict) else 0
        session   = "Intraday" if intra_cnt > over_cnt else "Overnight"

        # Instrument preference
        opt_count = opt_strat.get("option_trade_count", 0)
        eq_count  = metrics.get("total_trades", 1) - opt_count
        instrument = "Options" if opt_count > eq_count else "Equity"

        # Hold duration class
        if hold_min < 30:
            hold_class = "Scalp (< 30 min)"
        elif hold_min < 240:
            hold_class = "Short-Term (< 4 hrs)"
        elif hold_min < 1440:
            hold_class = "Intraday (< 1 day)"
        else:
            hold_class = "Swing (Multi-day)"

        # News signal check
        news_trades = len(metrics.get("evidence_packs", {}).get("fomo_entries", []))
        signal_style = "News-Driven" if news_trades > 5 else "Score-Driven" if "total_score" in df.columns else "Chart-Pattern"

        # Entry quality
        if t_skill > 60:
            entry_q = "Strong"
        elif t_skill > 30:
            entry_q = "Average"
        else:
            entry_q = "Weak"

        return {
            "preferred_session":    session,
            "preferred_instrument": instrument,
            "hold_duration_class":  hold_class,
            "signal_style":         signal_style,
            "entry_quality":        entry_q,
            "avg_hold_minutes":     round(hold_min, 1),
        }

    def _compute_pressure_map(self, df: pd.DataFrame) -> Dict:
        """Analyze behavioral shifts under pressure (after losses/wins)."""
        if df.empty or "pnl" not in df.columns:
            return {}

        df_s = df.sort_values("trade_datetime").reset_index(drop=True) if "trade_datetime" in df.columns else df.reset_index(drop=True)
        df_s["_prev_pnl"] = df_s["pnl"].shift(1)
        df_s["_prev_neg"]  = df_s["_prev_pnl"] < 0
        df_s["_prev_pos"]  = df_s["_prev_pnl"] > 0

        # Revenge Trade Index: avg PnL after a loss trade
        after_loss = df_s[df_s["_prev_neg"] == True]["pnl"]
        revenge_idx = float(after_loss.mean()) if not after_loss.empty else 0.0

        # Hot-hand bias: avg PnL after a win
        after_win  = df_s[df_s["_prev_pos"] == True]["pnl"]
        hot_hand   = float(after_win.mean()) if not after_win.empty else 0.0

        # Loss aversion: hold longer when losing?
        if "holding_period" in df_s.columns:
            hold_win  = df_s[df_s["pnl"] > 0]["holding_period"].mean()
            hold_loss = df_s[df_s["pnl"] < 0]["holding_period"].mean()
            loss_aversion = round(hold_loss / max(hold_win, 1), 2) if hold_win else 0
        else:
            loss_aversion = 0

        # Interpretation
        tendencies = []
        if revenge_idx < -500:
            tendencies.append({"pattern": "Revenge Trading", "impact": "High", "signature": f"Avg PnL after a loss: ₹{revenge_idx:,.0f}"})
        if hot_hand > 0 and after_win.std() > after_loss.std():
            tendencies.append({"pattern": "Hot-Hand Overconfidence", "impact": "Medium", "signature": "Increases position after winning streaks"})
        if loss_aversion > 1.5:
            tendencies.append({"pattern": "Loss-Hope Holding", "impact": "High", "signature": f"Holds losing trades {loss_aversion}× longer than winners"})

        return {
            "revenge_trade_index": round(revenge_idx, 2),
            "hot_hand_avg_pnl_after_win": round(hot_hand, 2),
            "loss_aversion_ratio": loss_aversion,
            "tendencies": tendencies,
        }

    def _compute_consistency_score(self, df: pd.DataFrame, metrics: Dict) -> float:
        """Single 0–100 score measuring behavioral consistency."""
        win_rate    = metrics.get("win_rate", 50)
        max_dd_pct  = abs(metrics.get("max_drawdown_pct", 0))
        consec_win  = metrics.get("consecutive_wins", 0)
        consec_loss = metrics.get("consecutive_losses", 1)
        streak_ratio = consec_win / max(consec_loss, 1)

        # Weekly win-rate variance (stability)
        weekly_wr_var = 0.0
        if "trade_date" in df.columns and "pnl" in df.columns:
            df_c = df.copy()
            df_c["_week"] = pd.to_datetime(df_c["trade_date"], errors="coerce").dt.isocalendar().week
            weekly_stats = df_c.groupby("_week")["pnl"].apply(lambda x: (x > 0).mean() * 100)
            if len(weekly_stats) > 1:
                weekly_wr_var = float(weekly_stats.std())

        stability_score = max(0, 100 - weekly_wr_var) * 0.4
        drawdown_score  = max(0, 100 - max_dd_pct) * 0.3
        streak_score    = min(100, streak_ratio * 20) * 0.3

        return round(stability_score + drawdown_score + streak_score, 1)

    def _compute_emotional_leakage(self, metrics: Dict) -> float:
        """
        Emotional Leakage Index (0–100): higher = more irrational behavior detected.
        Calculated using normalized severity scores for FOMO, Loss Averaging, 
        Revenge Trading, and Loss Aversion.
        """
        total_trades = max(metrics.get("total_trades", 0), 1)

        evidence = metrics.get("evidence_packs", {})
        pressure = metrics.get("behavioral_pressure_map", {})

        # 1. Retrieve raw counts from evidence packs or list lengths (if not yet updated)
        fomo_count    = evidence.get("fomo_count", len(evidence.get("fomo_entries", [])))
        avg_count     = evidence.get("loss_averaging_count", len(evidence.get("loss_averaging", [])))
        revenge_count = evidence.get("revenge_count", len(evidence.get("revenge_sequences", [])))

        # 2. Retrieve pressure metrics
        revenge_idx   = pressure.get("revenge_trade_index", 0.0)
        loss_aversion = pressure.get("loss_aversion_ratio", 1.0)
        avg_loss      = abs(metrics.get("avg_loss", 0.0)) or 1.0

        # --- Sub-Component 1: FOMO Severity (s_fomo) ---
        # 20% FOMO trades = 1.0 severity
        s_fomo = min(1.0, fomo_count / max(1, total_trades * 0.20))

        # --- Sub-Component 2: Loss Averaging Severity (s_avg) ---
        # 15% trades with loss averaging = 1.0 severity
        s_avg = min(1.0, avg_count / max(1, total_trades * 0.15))

        # --- Sub-Component 3: Revenge Trading Severity (s_revenge) ---
        # Frequency (15% = 1.0) + Performance Degradation (avg error > avg_loss)
        s_revenge_freq = min(1.0, revenge_count / max(1, total_trades * 0.15))
        s_revenge_pnl  = min(1.0, max(0.0, -revenge_idx) / avg_loss)
        s_revenge      = 0.5 * s_revenge_freq + 0.5 * s_revenge_pnl

        # --- Sub-Component 4: Holding Losers (s_hold) ---
        # 1.0 = baseline, 3.0+ = severe bias
        s_hold = 0.0 if loss_aversion <= 1 else min(1.0, (loss_aversion - 1.0) / 2.0)

        # Combine with weights
        eli = 100 * (
            0.30 * s_fomo +
            0.25 * s_avg + 
            0.25 * s_revenge + 
            0.20 * s_hold
        )
        
        # Store individual severities for granular insight
        metrics["behavioral_severities"] = {
            "fomo": round(s_fomo, 2),
            "averaging": round(s_avg, 2),
            "revenge": round(s_revenge, 2),
            "holding": round(s_hold, 2)
        }
        
        return round(min(100.0, max(0.0, eli)), 1)

    def _get_recent_market_context(self, df: pd.DataFrame) -> Dict:
        """Fetch minute-level data & news for the most recently traded symbols over the last 3 active days."""

        context = {"per_minute_data_t_plus_1": [], "news": []}
        try:
            if "trade_date" not in df.columns or df.empty:
                return context

            # Identify last 3 trading days
            recent_dates = sorted(df["trade_date"].dropna().unique())[-3:]
            recent_trades = df[df["trade_date"].isin(recent_dates)]
            symbols = recent_trades["symbol"].unique().tolist()
            if not symbols:
                return context

            # Fetch Minute Data + T+1
            start_dt = pd.to_datetime(recent_dates[0]).strftime("%Y-%m-%d")
            end_dt = (pd.to_datetime(recent_dates[-1]) + timedelta(days=2)).strftime("%Y-%m-%d")

            db_conf = self.config["database"]
            engine = create_engine(f"postgresql://{db_conf['user']}:{db_conf['password']}@{db_conf['host']}:{db_conf.get('port', 5432)}/{db_conf['dbname']}")

            # 1. Fetch 1-min Candle Data
            query_candles = """
                SELECT symbol, MIN(candle_ts) as start_time, MAX(candle_ts) as end_time,
                       MAX(high_price) as range_high, MIN(low_price) as range_low,
                       SUM(volume) as total_volume, AVG(open_interest) as avg_oi
                FROM candle_data
                WHERE symbol = ANY(%s) AND candle_ts >= %s AND candle_ts <= %s
                GROUP BY symbol;
            """
            candles_df = pd.read_sql(query_candles, engine, params=(symbols, start_dt, end_dt))
            
            # Format nicely for LLM context (avoid dumping huge raw arrays)
            for _, row in candles_df.iterrows():
                context["per_minute_data_t_plus_1"].append({
                    "symbol": row["symbol"],
                    "period": f"{row['start_time']} to {row['end_time']}",
                    "high": float(row["range_high"]) if pd.notnull(row["range_high"]) else None,
                    "low": float(row["range_low"]) if pd.notnull(row["range_low"]) else None,
                    "volume": float(row["total_volume"]) if pd.notnull(row["total_volume"]) else None,
                    "avg_open_interest": float(row["avg_oi"]) if pd.notnull(row["avg_oi"]) else None
                })
            
            if "news_category" in recent_trades.columns:
                news_df = recent_trades[recent_trades["is_news"] == True]
                for _, row in news_df.iterrows():
                    context["news"].append({
                        "symbol": row["symbol"],
                        "date": str(row["trade_date"]),
                        "category": str(row["news_category"])
                    })
        except Exception as e:
            self.logger.error(f"Error fetching recent market context: {e}")

        return context

    def _extract_evidence_packs(self, df: pd.DataFrame) -> Dict:
        """
        Extract specific trade clusters as 'evidence' for the LLM to ground its claims.
        Focus: Inefficient exits, FOMO entries, Revenge/Spirals, and Loss Averaging.
        Calculates total counts for behavioral frequency analysis.
        """
        evidence = {
            "fomo_count": 0,
            "loss_averaging_count": 0,
            "revenge_count": 0,
            "inefficient_exits": [],
            "fomo_entries": [],
            "revenge_sequences": [],
            "loss_averaging": []
        }
        
        # Ensure trade_datetime exists
        if "trade_datetime" not in df.columns:
            df["trade_datetime"] = pd.to_datetime(df["trade_date"])

        # 1. Inefficient Winning Exits (Profit Leakage)
        winners = df[df["pnl"] > 0].copy()
        if not winners.empty and "entry_efficiency" in winners.columns:
             # For SELL trades, 'entry_efficiency' is actually 'exit_efficiency'
             inefficient_exits = winners[winners["transaction_type"] == "SELL"].sort_values("entry_efficiency")
             evidence["inefficient_exits"] = inefficient_exits[["symbol", "trade_date", "pnl", "entry_efficiency"]].head(5).to_dict("records")

        # 2. FOMO-like Entries (BUY at high extremes)
        fomo_all = df[
            (df["transaction_type"] == "BUY") & 
            ((df.get("is_52week_high", False) == True) | (df.get("is_alltime_high", False) == True))
        ]
        evidence["fomo_count"] = len(fomo_all)
        evidence["fomo_entries"] = fomo_all.sort_values("entry_efficiency", ascending=False).head(5)[["symbol", "trade_date", "price"]].to_dict("records")

        # 3. Revenge Clusters (Time-based proximity to losses)
        revenge_trades = []
        df_sorted = df.sort_values("trade_datetime")
        for i in range(1, len(df_sorted)):
            prev = df_sorted.iloc[i-1]
            curr = df_sorted.iloc[i]
            if prev["pnl"] < 0:
                time_diff = (curr["trade_datetime"] - prev["trade_datetime"]).total_seconds() / 60
                if 0 < time_diff < 30:
                    evidence["revenge_count"] += 1
                    if len(revenge_trades) < 5:
                        revenge_trades.append({
                            "symbol": curr["symbol"], 
                            "time_since_last_loss_min": round(time_diff, 1),
                            "result": "WIN" if curr["pnl"] > 0 else "LOSS", 
                            "pnl": float(curr["pnl"])
                        })
        evidence["revenge_sequences"] = revenge_trades

        # 4. Loss Averaging (Multiple buys at lower prices)
        loss_avg_examples = []
        buys = df[df["transaction_type"] == "BUY"]
        if not buys.empty:
            for (symbol, day), group in buys.groupby(["symbol", "trade_date"]):
                if len(group) >= 2:
                    group = group.sort_values("trade_datetime")
                    # Final price lower than first implies adding to a losing position on the way down
                    if group["price"].iloc[-1] < group["price"].iloc[0]:
                        evidence["loss_averaging_count"] += 1
                        if len(loss_avg_examples) < 5:
                            loss_avg_examples.append({"symbol": symbol, "date": str(day), "adds": len(group)})
        evidence["loss_averaging"] = loss_avg_examples

        return evidence

    def _compute_sectoral_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze sector-level profitability and trading patterns with detailed metrics"""
        if "industry" not in df.columns or df["industry"].dropna().empty:
            return {}

        sector_stats = []
        # Group by mapped industry
        for sector, group in df.groupby("industry"):
            # A 'trade' is a row with non-zero pnl (FIFO matched)
            traded_group = group[group["pnl"] != 0]
            if traded_group.empty: continue
            
            pnl_sum = float(traded_group["pnl"].sum())
            total_trades = len(traded_group)
            
            wins = traded_group[traded_group["pnl"] > 0]
            losses = traded_group[traded_group["pnl"] < 0]
            
            win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
            loss_rate = (len(losses) / total_trades * 100) if total_trades > 0 else 0
            
            max_pnl = float(traded_group["pnl"].max())
            max_loss = float(traded_group["pnl"].min()) # Most negative value
            
            sector_stats.append({
                "sector": sector,
                "pnl": pnl_sum,
                "win_rate": round(win_rate, 2),
                "loss_rate": round(loss_rate, 2),
                "max_pnl": max_pnl,
                "max_loss": max_loss,
                "trade_count": total_trades,
                "avg_pnl": pnl_sum / total_trades if total_trades > 0 else 0
            })

        if not sector_stats: return {}

        # Identify Top/Bottom sectors
        sorted_stats = sorted(sector_stats, key=lambda x: x["pnl"])
        
        return {
            "all_sectors": sorted_stats[::-1], # Best to worst
            "money_maker": sorted_stats[-1]["sector"] if sorted_stats[-1]["pnl"] > 0 else None,
            "money_leaker": sorted_stats[0]["sector"] if sorted_stats[0]["pnl"] < 0 else None,
            "sector_count": len(sector_stats)
        }

    def _detect_option_strategy(self, df: pd.DataFrame) -> Dict:
        """Detect whether trader is primarily an option writer or buyer and categorize activities"""
        # Filter for options only
        option_df = df[df["symbol"].str.contains("CE|PE", na=False)].copy()
        if option_df.empty:
            return {"type": "EQUITY_ONLY"}

        # Logic for buyer vs writer: First transaction in a symbol determines intent
        writer_signals = 0
        buyer_signals = 0
        
        for symbol, group in option_df.groupby("symbol"):
            first_trade = group.sort_values("trade_date").iloc[0]
            if first_trade["transaction_type"] == "SELL":
                writer_signals += 1
            else:
                buyer_signals += 1
        
        strategy_type = "OPTION_WRITER" if writer_signals > buyer_signals else "OPTION_BUYER"
        
        # Categorize by moneyness if possible (needs spot price)
        # Using DB 'close' as proxy for spot if available
        moneyness_stats = {"ATM": {"count": 0, "pnl": 0}, "OTM": {"count": 0, "pnl": 0}, "ITM": {"count": 0, "pnl": 0}}
        
        # Simple placeholder for moneyness until spot price logic is refined
        return {
            "strategy_type": strategy_type,
            "writer_ratio": writer_signals / (writer_signals + buyer_signals) if (writer_signals+buyer_signals) > 0 else 0,
            "option_trade_count": len(option_df),
            "option_pnl": float(option_df["pnl"].sum()),
            "moneyness_analysis": moneyness_stats
        }

    def _infer_trading_strategy(self, df: pd.DataFrame, metrics: Dict) -> Dict:
        """Clusters trades and uses performance attributes to infer the underlying strategy"""
        strategies = []
        
        # 1. Premium Selling (Option Writing)
        opt_strat = metrics.get("option_strategy", {})
        if opt_strat.get("strategy_type") == "OPTION_WRITER":
            strategies.append({"name": "Premium Selling", "confidence": 0.9, "desc": "Income generation via theta decay"})

        # 2. Scalping (Very short holds)
        avg_hold = df["holding_period"].mean() if "holding_period" in df.columns else 0
        if 0 < avg_hold < 30:
            strategies.append({"name": "Scalping", "confidence": 0.8, "desc": "High-frequency, short-duration trades"})
            
        # 3. Momentum / Breakout
        if metrics.get("avg_t_score", 0) > 70:
            strategies.append({"name": "Momentum Chasing", "confidence": 0.7, "desc": "Entering stocks with strong technical trends"})

        # 4. Mean Reversion 
        # (Buying low proximity, Selling high proximity - works in ranges)
        if metrics.get("timing_skill_score", 0) > 60 and metrics.get("avg_t_score", 0) < 50:
            strategies.append({"name": "Mean Reversion", "confidence": 0.6, "desc": "Buying dips and selling peaks in ranges"})

        return {
            "detected_strategies": sorted(strategies, key=lambda x: x["confidence"], reverse=True),
            "primary_strategy": strategies[0]["name"] if strategies else "Unclassified / Mixed"
        }

    def _simulate_what_if(self, df: pd.DataFrame) -> Dict:
        """Simulate how performance would change with specific behavioral adjustments"""
        baseline_pnl = df["pnl"].sum()
        if len(df) == 0:
            return {"baseline_pnl": 0, "rules": []}

        # Filter to actual trades
        traded_df = df[df["pnl"] != 0].copy()
        
        # 1. Simulation: "What if Max Loss was Capped at 3x Avg Loss?"
        avg_loss = abs(traded_df[traded_df["pnl"] < 0]["pnl"].mean()) if not traded_df[traded_df["pnl"] < 0].empty else 1
        cap = avg_loss * 3
        sim_a_pnl = traded_df["pnl"].apply(lambda x: max(x, -cap) if x < 0 else x).sum()
        delta_cap = sim_a_pnl - baseline_pnl
        
        # 2. Simulation: "Avoid Worst Trading Day"
        df_tmp = df.copy()
        df_tmp["day_name"] = df_tmp["trade_date"].dt.day_name()
        day_pnl = df_tmp.groupby("day_name")["pnl"].sum()
        worst_day = day_pnl.idxmin() if not day_pnl.empty else None
        
        delta_day = 0
        if worst_day and day_pnl[worst_day] < 0:
            sim_day_pnl = df_tmp[df_tmp["day_name"] != worst_day]["pnl"].sum()
            delta_day = sim_day_pnl - baseline_pnl
        else:
            worst_day = "N/A"

        # 3. Simulation: "Eliminate Revenge Trades"
        # Revenge: Trade after a loss within a short window, often with larger size
        df_s = df.copy().sort_values("trade_date")
        df_s["prev_pnl"] = df_s["pnl"].shift(1)
        # Identify losses followed by another loss (revenge sequence)
        revenge_mask = (df_s["prev_pnl"] < 0) & (df_s["pnl"] < 0)
        sim_revenge_pnl = df_s.loc[~revenge_mask, "pnl"].sum()
        delta_revenge = sim_revenge_pnl - baseline_pnl

        # 4. Simulation: "Avoid Low-Score Trades (Technical Discipline)"
        if "total_score" in df.columns:
            low_score_mask = df["total_score"] < 4  # Assuming 0-10 scale
            sim_score_pnl = df.loc[~low_score_mask, "pnl"].sum()
            delta_score = sim_score_pnl - baseline_pnl
        else:
            delta_score = 0

        rules = []
        if delta_cap > 0:
            pct = (delta_cap / abs(baseline_pnl) * 100) if baseline_pnl != 0 else 0
            rules.append({ "action": f"Cap losses at ₹{cap:,.0f} (3x Avg Loss)", "delta": float(delta_cap), "impact_pct": float(pct) })
        
        if delta_day > 0:
            pct = (delta_day / abs(baseline_pnl) * 100) if baseline_pnl != 0 else 0
            rules.append({ "action": f"Stop trading on {worst_day}s", "delta": float(delta_day), "impact_pct": float(pct) })
            
        if delta_revenge > 0:
            pct = (delta_revenge / abs(baseline_pnl) * 100) if baseline_pnl != 0 else 0
            rules.append({ "action": "Eliminate impulsive revenge trades", "delta": float(delta_revenge), "impact_pct": float(pct) })

        if delta_score > 0:
            pct = (delta_score / abs(baseline_pnl) * 100) if baseline_pnl != 0 else 0
            rules.append({ "action": "Avoid low-probability (low score) setups", "delta": float(delta_score), "impact_pct": float(pct) })

        # Sort by impact
        rules = sorted(rules, key=lambda x: x["delta"], reverse=True)

        return {
            "baseline_pnl": float(baseline_pnl),
            "cap_max_loss_pnl": float(sim_a_pnl),
            "worst_day": worst_day,
            "rules": rules[:4] # Top 4 impacts
        }

    def _mine_loss_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify toxic clusters (leaks) by correlating losses with time/day/context"""
        if df.empty: return {}
        
        df = df.copy()
        df["day_name"] = df["trade_date"].dt.day_name()
        
        # 1. Day-of-week leak
        day_pnl = df.groupby("day_name")["pnl"].sum().sort_values()
        toxic_day = day_pnl.index[0] if day_pnl.iloc[0] < 0 else None
        
        # 2. Time-of-day leak (First Hour vs Last Hour)
        def time_bucket(hour):
            if hour is None: return "Unknown"
            if hour <= 10: return "Open/First Hour"
            if hour >= 14: return "Close/Last Hour"
            return "Mid-Day"
            
        if "trade_hour" in df.columns:
            df["time_bucket"] = df["trade_hour"].apply(time_bucket)
            time_pnl = df.groupby("time_bucket")["pnl"].sum().sort_values()
            toxic_time = time_pnl.index[0] if time_pnl.iloc[0] < 0 else None
        else:
            toxic_time = None
            
        return {
            "toxic_day": toxic_day,
            "toxic_time": toxic_time,
            "summary": f"Most losses occur on {toxic_day} during {toxic_time}" if toxic_day and toxic_time else "No specific temporal leaks found"
        }

    # =========================================================
    # Core Metrics (unchanged)
    # =========================================================
    def calculate_total_pnl(self, df: pd.DataFrame) -> float:
        return float(df['pnl'].sum())

    def calculate_win_rate(self, df: pd.DataFrame) -> float:
        # Exclude pnl=0 rows (unmatched open legs) from win rate calculation
        traded = df[df['pnl'] != 0]
        if len(traded) == 0:
            return 0.0
        winning_trades = len(traded[traded['pnl'] > 0])
        return float(winning_trades / len(traded) * 100)

    def calculate_avg_win(self, df: pd.DataFrame) -> float:
        winning_trades = df[df['pnl'] > 0]['pnl']
        return float(winning_trades.mean()) if len(winning_trades) > 0 else 0.0

    def calculate_avg_loss(self, df: pd.DataFrame) -> float:
        losing_trades = df[df['pnl'] < 0]['pnl']
        return float(losing_trades.mean()) if len(losing_trades) > 0 else 0.0

    def calculate_overall_win_rate(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        winning_trades = len(df[df['pnl'] > 0])
        return float(winning_trades / len(df) * 100)

    def calculate_avg_win_pct(self, df: pd.DataFrame) -> float:
        winners = df[df['pnl'] > 0]
        if winners.empty or "trade_value" not in winners.columns:
            return 0.0
        pcts = (winners['pnl'] / winners['trade_value'].abs() * 100).clip(-500, 500)
        return float(pcts.mean())

    def calculate_avg_loss_pct(self, df: pd.DataFrame) -> float:
        losers = df[df['pnl'] < 0]
        if losers.empty or "trade_value" not in losers.columns:
            return 0.0
        pcts = (losers['pnl'] / losers['trade_value'].abs() * 100).clip(-500, 500)
        return float(pcts.mean())

    def _compute_multivariate_pattern_analysis(self, df: pd.DataFrame) -> list:
        # Identify the pattern of failures and wins based on combinations
        temp_df = df.copy()
        
        for col in ["f_score", "t_score", "total_score"]:
            if col in temp_df.columns:
                mean_val = temp_df[col].mean()
                temp_df[f"{col}_bucket"] = temp_df[col].apply(lambda x: "High" if pd.notna(x) and x > mean_val else "Low")
            else:
                temp_df[f"{col}_bucket"] = "N/A"
                
        for col in ["is_news", "is_event", "is_alltime_high", "is_high_volume"]:
            if col not in temp_df.columns:
                temp_df[col] = False
                
        group_cols = [
            "f_score_bucket", "t_score_bucket", "total_score_bucket", 
            "is_news", "is_alltime_high", "is_high_volume"
        ]
        
        group_cols = [c for c in group_cols if c in temp_df.columns]
        results = []
        if not group_cols:
            return results
            
        grouped = temp_df.groupby(group_cols)
        
        for name, group in grouped:
            if len(group) < 3: 
                continue
            
            pattern_parts = []
            if not isinstance(name, tuple):
                name = (name,)
                
            for i, col in enumerate(group_cols):
                val = name[i]
                if pd.isna(val) or val == "N/A": 
                    continue
                if isinstance(val, bool) or isinstance(val, np.bool_):
                    if val:
                        pattern_parts.append(col.replace("is_", "").replace("_", " ").upper())
                else:
                    pattern_parts.append(f"{val} {col.replace('_bucket', '').replace('_', ' ').title()}")
            
            pattern_name = " + ".join(pattern_parts) if pattern_parts else "Baseline (No Features)"
            
            win_count = len(group[group["pnl"] > 0])
            total_count = len(group)
            win_rate = (win_count / total_count * 100) if total_count > 0 else 0
            avg_pnl = group["pnl"].mean()
            
            results.append({
                "pattern": pattern_name,
                "count": total_count,
                "win_count": win_count,
                "win_rate": round(win_rate, 2),
                "avg_pnl_per_trade": round(avg_pnl, 2),
                "total_pnl": round(group["pnl"].sum(), 2)
            })
            
        results = sorted(results, key=lambda x: x["count"], reverse=True)
        return results[:15]

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
        return float(df['holding_period'].mean()) if 'holding_period' in df.columns else 0.0

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
        Segment trades by holding period using 'holding_period' if available.
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
        col = "holding_period"
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
            # Prefer explicit 'price' from file (this is the premium for options, or last stock price)
            if "price" in g.columns:
                s = g.sort_values("trade_date")["price"].dropna()
                if len(s) > 0:
                    return float(s.iloc[-1])
            # Fallback to 'close' from DB
            if "close" in g.columns:
                s = g.sort_values("trade_date")["close"].dropna()
                if len(s) > 0:
                    return float(s.iloc[-1])
            return 0.0

        def get_underlying_close(g: pd.DataFrame) -> float:
            # This is the joined price from nse_stock_quotes based on __base_sym
            if "close" in g.columns:
                s = g.sort_values("trade_date")["close"].dropna()
                if len(s) > 0:
                    return float(s.iloc[-1])
            return 0.0

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
            u_close = get_underlying_close(g)

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
                "underlying_close": u_close,
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
            
            # Asset classification
            asset_name, asset_kind = self.classify_instrument(symbol)
            is_option = "OPTION" in asset_kind
            
            open_qty = abs(net_qty)
            entry_price = row["buy_price"] if pos_type == "LONG-OPEN" else row["sell_price"]
            invested = open_qty * entry_price
            unreal = 0.0

            if pos_type in ["LONG-OPEN", "SHORT-OPEN"]:
                if is_option:
                    try:
                        pts = str(symbol).upper().replace(",", "").split()
                        # Robust option parsing: EO CE SYMBOL DATE STRIKE
                        opt_type = "CE" if "CE" in pts else ("PE" if "PE" in pts else "CE")
                        strike = 0.0
                        expiry_date = None
                        
                        for p in pts:
                            # Try date: 28AUG2025
                            if any(m in p for m in ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]):
                                try:
                                    expiry_date = datetime.strptime(p, "%d%b%Y").date()
                                except: pass
                            # Try strike
                            p_cl = p.replace(',','')
                            if p_cl.replace('.','',1).isdigit() and float(p_cl) > 100:
                                strike = float(p_cl)
                        
                        today_date = date.today()
                        u_ltp = row["underlying_close"]
                        
                        # If past expiry, compute intrinsic value
                        if expiry_date and today_date > expiry_date and u_ltp > 0:
                            if opt_type == 'CE':
                                payoff = max(0.0, u_ltp - strike)
                            else: # PE
                                payoff = max(0.0, strike - u_ltp)
                            
                            if pos_type == "LONG-OPEN":
                                unreal = (payoff * open_qty) - invested
                            else: # SHORT-OPEN
                                unreal = invested - (payoff * open_qty)
                        else:
                            # Standard MTM using symbol Ltp (premium)
                            if pos_type == "LONG-OPEN":
                                unreal = (ltp - entry_price) * open_qty
                            else: # SHORT-OPEN (Premium Recv - Current Premium cost to buy back)
                                unreal = (entry_price - ltp) * open_qty
                    except Exception as e:
                        # Fallback
                        unreal = (ltp - entry_price) * open_qty if pos_type == "LONG-OPEN" else (entry_price - ltp) * open_qty
                else:
                    # EQUITY MTM: (Current Price - Entry) * Qty
                    unreal = (ltp - entry_price) * open_qty if pos_type == "LONG-OPEN" else (entry_price - ltp) * open_qty
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

        gainer_pct = [min(500, p)  for (s, p) in sym_pct if s in gainers and p == p]
        loser_pct  = [max(-500, p) for (s, p) in sym_pct if s in losers  and p == p]
        avg_gainer_pct = float(np.mean(gainer_pct)) if gainer_pct else 0.0
        avg_loser_pct  = float(np.mean(loser_pct))  if loser_pct  else 0.0

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

    def _build_pnl_timeline(self, csv_path: str, nifty_data: Dict = None) -> Dict[str, List]:
        df = pd.read_csv(csv_path)
        self.pnl_df = df
        
        # Robust column mapping
        col_map = {c.lower(): c for c in df.columns}
        date_col = next((col_map[k] for k in ['trade_date', 'date', 'trade date', 'timestamp'] if k in col_map), None)
        pnl_col = next((col_map[k] for k in ['pnl', 'mtm', 'profit', 'realized_pnl'] if k in col_map), None)
        
        if not date_col or not pnl_col:
            self.logger.warning(f"Could not identify Date or PnL column in {csv_path}. Columns: {list(df.columns)}")
            return {'dates': [], 'values': [], 'nifty_values': []}

        df = df.rename(columns={date_col: 'trade_date', pnl_col: 'pnl'})
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        self.pnl_df = df
        
        # Calculate daily pnl
        daily = df.groupby(df['trade_date'].dt.date)['pnl'].sum()
        cumulative = daily.cumsum()
        self.last_pnl = float(df["pnl"].iloc[-1]) if not df.empty else 0.0
        self.cumulative_pnl = float(cumulative.iloc[-1]) if len(cumulative) else 0.0
        
        # Build benchmark data if available
        bench_values = []
        bench_original_values = []
        if nifty_data:
            first_nifty = None
            for d in cumulative.index:
                d_str = str(d) # YYYY-MM-DD
                val = nifty_data.get(d_str)
                if val is not None:
                    if first_nifty is None:
                        first_nifty = val
                    
                    # 1. Scaled/Normalized (Existing behavior)
                    norm_val = (val / first_nifty - 1) * max(abs(cumulative.max()), 100000)
                    bench_values.append(float(norm_val))
                    
                    # 2. Original Scale (Requested)
                    bench_original_values.append(float(val))
                else:
                    bench_values.append(bench_values[-1] if bench_values else 0.0)
                    bench_original_values.append(bench_original_values[-1] if bench_original_values else 0.0)

        return {
            'dates': [str(d) for d in cumulative.index],
            'values': [float(v) for v in cumulative.values],
            'benchmark_values': bench_values if bench_values else [],
            'benchmark_original_values': bench_original_values if bench_original_values else []
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
            "risk_handling": self._calc_risk_handling(df), # NEW: Handling skill
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

    def _calc_risk_handling(self, df: pd.DataFrame) -> float:
        """Risk Handling = SKILL at managing risk (cutting losers, recovering from drawdowns).
        Distinct from Risk Appetite (willingness to take risk).
        """
        losses = df[df['pnl'] < 0]
        if losses.empty:
            return 0.8

        # 1. Loss cutting efficiency: avg loss vs max loss (lower ratio = better cutting)
        max_loss = abs(losses['pnl'].min())
        avg_loss = abs(losses['pnl'].mean())
        loss_control = 1 - (avg_loss / max_loss) if max_loss > 0 else 0.5

        # 2. Winner/loser holding ratio (good handlers cut losers faster than winners)
        winners = df[df['pnl'] > 0]
        if 'holding_period' in df.columns:
            avg_hold_winners = winners['holding_period'].mean() if not winners.empty else 30
            avg_hold_losers = losses['holding_period'].mean() if not losses.empty else 30
            # Good handling: losers held shorter than winners → ratio > 1
            hold_ratio = avg_hold_winners / (avg_hold_losers + 1e-6)
            hold_skill = min(1.0, hold_ratio / 2.0)  # 2x ratio = perfect
        else:
            hold_skill = 0.5

        # 3. Drawdown recovery: how quickly does the equity curve recover after dips?
        pnl_cum = df['pnl'].cumsum()
        peak = pnl_cum.expanding().max()
        drawdowns = peak - pnl_cum
        # Recovery speed: proportion of time in drawdown
        in_dd = (drawdowns > 0).sum()
        recovery_speed = 1.0 - (in_dd / (len(df) + 1e-6))

        # 4. Profit factor as a skill indicator (separate from appetite)
        pf = self.calculate_profit_factor(df)
        reliability = min(1.0, pf / 2.0)

        # Composite: weighted towards actual skill metrics
        handling = (0.25 * loss_control +
                    0.25 * hold_skill +
                    0.20 * recovery_speed +
                    0.30 * reliability)

        return round(min(1.0, float(handling)), 2)

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
        """Risk Appetite = willingness to take risk.
        Based on avg loss size, max single loss, and loss frequency.
        NOT based on diversification (that's an investing metric, not trading).
        """
        losses = df[df['pnl'] < 0]
        if losses.empty:
            return 0.3  # No losses = conservatively low appetite

        # 1. Avg loss relative to avg trade value (how much pain do they accept per trade?)
        avg_loss = abs(losses['pnl'].mean())
        avg_trade_val = df['trade_value'].mean()
        loss_tolerance = avg_loss / (avg_trade_val + 1e-6)  # Higher = more risk appetite

        # 2. Max single loss relative to avg loss (tail risk acceptance)
        max_loss = abs(losses['pnl'].min())
        tail_risk = max_loss / (avg_loss + 1e-6)  # Higher = accepts extreme losses
        tail_score = min(1.0, tail_risk / 5.0)  # Normalize: 5x avg loss = max

        # 3. Loss frequency (how often they're willing to lose)
        traded = df[df['pnl'] != 0]
        loss_freq = len(losses) / (len(traded) + 1e-6)

        # 4. Position sizing on losers vs winners (do they size up on risky trades?)
        winners = df[df['pnl'] > 0]
        avg_qty_losers = losses['quantity'].mean() if not losses.empty else 0
        avg_qty_winners = winners['quantity'].mean() if not winners.empty else 1
        size_aggression = avg_qty_losers / (avg_qty_winners + 1e-6)

        # Composite score: higher = more risk appetite
        raw = (0.3 * loss_tolerance * 10 +  # Scale up for meaningful contribution
               0.25 * tail_score +
               0.25 * loss_freq +
               0.2 * min(1.0, size_aggression))

        return round(min(1.0, max(0.0, raw)), 2)

    def _calc_patience(self, df: pd.DataFrame) -> float:
        if "holding_period" not in df.columns:
            return 0.5
        pos_trades = df[df["pnl"] > 0]
        neg_trades = df[df["pnl"] < 0]
        pos_hold = pos_trades["holding_period"].mean() if len(pos_trades) > 0 else 0
        neg_hold = neg_trades["holding_period"].mean() if len(neg_trades) > 0 else 0
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

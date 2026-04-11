"""
Data Processor Module
Handles data loading, cleaning, and preparation for analysis using SQLAlchemy
"""
import json
import os
import pandas as pd
import numpy as np
import logging
import psycopg2
from sqlalchemy import create_engine
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, date

class TradingDataProcessor:
    """Process and clean trading data from various sources"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Initialize engine for reuse
        db_conf = self.config["database"]
        self.engine = create_engine(f"postgresql://{db_conf['user']}:{db_conf['password']}@{db_conf['host']}:{db_conf.get('port', 5432)}/{db_conf['dbname']}")

    def _get_base_symbol(self, symbol: str) -> str:
        """Robust symbol extraction to match with database records (underlying/symbol)"""
        s = str(symbol).upper().strip()
        parts = s.split()
        
        # Handle formats like: "IO CE BANKNIFTY 31Jul2025 59500"
        if len(parts) >= 3 and parts[1] in ("CE", "PE"):
            return parts[2]
            
        # Handle formats like: "RELIANCE EQ"
        return parts[0]

    def getadditionaldata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get additional data (scores, highs/lows, etc.) from database and merge with input DataFrame"""
        try:
            db_conf = self.config["database"]
            table_name = db_conf.get("table_name", "stock_data")

            # --- Normalize date range from input df ---
            df["trade_datetime"] = pd.to_datetime(df["trade_date"], errors="coerce")
            df["trade_date_only"] = df["trade_datetime"].dt.date

            start_date = df["trade_date_only"].min()
            end_date = df["trade_date_only"].max()

            if pd.isna(start_date) or pd.isna(end_date):
                return df

            # Fetch relevant columns from DB - now expanded for consolidated trace sheet
            query = f"""
                SELECT symbol, date, open, high, low, close, volume,
                       dma_10, dma_21, dma_50, dma_100, 
                       ema_10, ema_21, ema_50, ema_100, 
                       t_score, f_score, total_score,
                       is_52week_high, is_52week_low, is_alltime_high,
                       is_alltime_low, macd, macd_signal, macd_histogram,
                       rsi, k_percent, d_percent, j_percent, cci, roc,
                       atr, atr_5, z_score_75, pvo, mfi, adx, adxr, 
                       vwma, bb_middle, bb_upper, bb_lower,
                       sma_50, sma_200, news_impactscore, is_news,
                       is_event, is_high_volume, ema_score, news_sentiment,
                       market_behaviour, news_category, chart_charts, sector_ema
                FROM {table_name}
                WHERE date >= %s AND date <= %s;
            """

            db_df = pd.read_sql(query, self.engine, params=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
            
            # Normalize column names
            db_df.columns = db_df.columns.str.lower()
            db_df["date"] = pd.to_datetime(db_df["date"]).dt.date
            db_df["symbol"] = db_df["symbol"].str.upper().str.strip()

            # Merge logic for symbols (base symbol matching)
            df['__match_sym'] = df['symbol'].apply(self._get_base_symbol)
            
            # Use vectorized merge
            merged = pd.merge(
                df,
                db_df,
                left_on=['__match_sym', 'trade_date_only'],
                right_on=['symbol', 'date'],
                how='left',
                suffixes=('', '_db')
            )
            
            # Cleanup
            merged.drop(columns=['symbol_db', 'date', '__match_sym', 'trade_date_only'], inplace=True, errors='ignore')
            return merged

        except Exception as e:
            self.logger.error(f"❌ Error getting additional data: {str(e)}")
            return df

    def _enrich_from_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich with data from 'market_data' database"""
        if df.empty: return df
        try:
            db_conf = self.config["database"].copy()
            db_conf["dbname"] = "market_data"
            mkt_engine = create_engine(f"postgresql://{db_conf['user']}:{db_conf['password']}@{db_conf['host']}:{db_conf.get('port', 5432)}/{db_conf['dbname']}")
            
            # Use trade_date_only for matching
            df['trade_date_only'] = pd.to_datetime(df['trade_date']).dt.date
            unique_dates = df['trade_date_only'].dropna().unique()
            date_strs = [d.strftime("%Y-%m-%d") for d in unique_dates]
            if not date_strs: return df

            # 1. nse_stock
            query_stock = "SELECT symbol, date, dist_from_52w_high_pct, pct_chg_1w, pct_chg_1m, pct_chg_6m FROM nse_stock WHERE date IN %s"
            mkt_stock_df = pd.read_sql(query_stock, mkt_engine, params=(tuple(date_strs),))
            
            if not mkt_stock_df.empty:
                mkt_stock_df['date'] = pd.to_datetime(mkt_stock_df['date']).dt.date
                mkt_stock_df['symbol'] = mkt_stock_df['symbol'].str.upper().str.strip()
                df['__match_sym'] = df['symbol'].apply(self._get_base_symbol)
                df = df.merge(mkt_stock_df, left_on=['__match_sym', 'trade_date_only'], right_on=['symbol', 'date'], how='left', suffixes=('', '_mkt'))
                df.drop(columns=['__match_sym', 'symbol_mkt', 'date_mkt', 'date'], inplace=True, errors='ignore')

            # 2. nse_index_ind
            indices = ('NIFTY', 'NIFTYMIDCAP150', 'NIFTYSMLCAP250')
            query_index = "SELECT symbol, date, close, pct_chg_1w, ema_50, ema_200 FROM nse_index_ind WHERE symbol IN %s AND date IN %s"
            mkt_index_df = pd.read_sql(query_index, mkt_engine, params=(indices, tuple(date_strs)))
            
            if not mkt_index_df.empty:
                mkt_index_df['date'] = pd.to_datetime(mkt_index_df['date']).dt.date
                sym_map = {'NIFTY': 'nifty50', 'NIFTYMIDCAP150': 'midcap', 'NIFTYSMLCAP250': 'smlcap'}
                for sym, prefix in sym_map.items():
                    subset = mkt_index_df[mkt_index_df['symbol'] == sym].copy()
                    if subset.empty: continue
                    subset = subset[['date', 'close']].rename(columns={'close': f'{prefix}_close'})
                    df = df.merge(subset, left_on='trade_date_only', right_on='date', how='left')
                    df.drop(columns=['date'], inplace=True, errors='ignore')

            df.drop(columns=['trade_date_only'], inplace=True, errors='ignore')
            return df
        except Exception as e:
            self.logger.error(f"❌ Error enriching market data: {e}")
            return df

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Main load entry point used by main.py"""
        df = pd.read_csv(filepath)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        
        # Rename mapping to ensure we have 'trade_date', 'symbol', 'transaction_type', 'quantity', 'price'
        # The ANISH_NEW.csv uses these names already (lowercase)
        
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        df = df.dropna(subset=['trade_date'])
        
        df = self.getadditionaldata(df)
        df = self._enrich_from_market_data(df)
        # Note: clean_data, classify_positions, and pair_trades are called sequentially in main.py
        return df

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        required_cols = self.config["data"]["required_columns"]
        missing = [col for col in required_cols if col not in df.columns]
        return (len(missing) == 0, missing)

    def handle_expired_options(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If an option position is short and the expiry has passed, add a 0-price BUY 
        trade at 3:30 PM on the expiry date to close the position.
        """
        if df.empty: return df
        
        # Calculate net quantity for each symbol to find open positions
        # Use a copy for calculation to determine which symbols need closing
        df_calc = df.copy()
        df_calc['_signed_qty'] = df_calc.apply(
            lambda r: r['quantity'] if str(r['transaction_type']).upper() == 'BUY' else -r['quantity'], 
            axis=1
        )
        net_qtys = df_calc.groupby('symbol')['_signed_qty'].sum()
        
        new_trades = []
        today = date.today()
        
        for symbol, net_qty in net_qtys.items():
            # Only consider short positions (net_qty < 0)
            if net_qty < -1e-5: 
                opt_info = self._parse_option_symbol(symbol)
                # Check if it's an option and the expiry date has passed
                if opt_info and today > opt_info['expiry']:
                    # close it at 0 on expiry day at 3:30 PM
                    expiry_date = opt_info['expiry']
                    expiry_ts = datetime.combine(expiry_date, datetime.strptime("15:30:00", "%H:%M:%S").time())
                    
                    # Create the closing "BUY" trade row
                    new_trade = {
                        'trade_date': expiry_ts,
                        'symbol': symbol,
                        'transaction_type': 'BUY',
                        'quantity': abs(float(net_qty)),
                        'price': 0.0,
                        'trade_hour': 15,
                        'trade_day_of_week': expiry_ts.weekday(),
                        'trade_month': expiry_ts.month,
                        'trade_value': 0.0,
                        'Date': expiry_ts,
                        'Symbol': symbol.upper()
                    }
                    
                    # Ensure all original columns are present to avoid concat issues
                    for col in df.columns:
                        if col not in new_trade:
                            new_trade[col] = np.nan
                            
                    new_trades.append(new_trade)
                    # self.logger.info(f"Adding synthetic expiry close for {symbol}: BUY {abs(net_qty)} @ 0.0 on {expiry_ts}")
        self.logger.info((f"Added all Sytheic tradess to close open positions of options expired"))
        if new_trades:
            # Append new trades and re-sort
            new_df = pd.DataFrame(new_trades)
            df = pd.concat([df, new_df], ignore_index=True)
            df = df.sort_values('trade_date').reset_index(drop=True)
            
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        for col in ['quantity', 'price']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
        
        # Derived fields
        df['trade_hour'] = df['trade_date'].dt.hour
        df['trade_day_of_week'] = df['trade_date'].dt.dayofweek
        df['trade_month'] = df['trade_date'].dt.month
        df['trade_value'] = df['price'] * df['quantity']
        df['Date'] = df['trade_date'] # For backend components
        df['Symbol'] = df['symbol'].str.upper() # For backend components
        
        # Add synthetic trades for expired options
        df = self.handle_expired_options(df)
        
        return df.sort_values('trade_date').reset_index(drop=True)

    def pair_trades(self, df: pd.DataFrame, output_dir: str = "data/reports", trader_name: str = "Trader") -> pd.DataFrame:
        """Full FIFO trade pairing logic with holding period calculation"""
        df = df.copy().sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        df["pnl"] = 0.0
        df["holding_period"] = 0.0 # Minutes
        
        trade_pairs = []
        for symbol, g in df.groupby("symbol"):
            buy_q, sell_q = [], []
            for idx, row in g.iterrows():
                qty, price, t = float(row['quantity']), float(row['price']), row['trade_date']
                if row['transaction_type'].upper() == 'BUY':
                    matched_weighted_hold = 0.0
                    total_matched_qty = 0.0
                    
                    while qty > 0 and sell_q:
                        sq_list = sell_q[0] # [qty, price, time]
                        m = min(qty, sq_list[0])
                        df.loc[idx, 'pnl'] += (sq_list[1] - price) * m
                        
                        # Holding period (buy closes a sell/short)
                        duration = (t - sq_list[2]).total_seconds() / 60.0
                        matched_weighted_hold += duration * m
                        total_matched_qty += m
                        
                        qty -= m
                        sq_list[0] -= m
                        if sq_list[0] <= 0: sell_q.pop(0)
                        
                    if total_matched_qty > 0:
                        df.loc[idx, 'holding_period'] = matched_weighted_hold / total_matched_qty
                        
                    if qty > 0: buy_q.append([qty, price, t])
                else: # Transaction type is SALE/SELL
                    matched_weighted_hold = 0.0
                    total_matched_qty = 0.0
                    
                    while qty > 0 and buy_q:
                        bq_list = buy_q[0] # [qty, price, time]
                        m = min(qty, bq_list[0])
                        df.loc[idx, 'pnl'] += (price - bq_list[1]) * m
                        
                        # Holding period (sell closes a buy/long)
                        duration = (t - bq_list[2]).total_seconds() / 60.0
                        matched_weighted_hold += duration * m
                        total_matched_qty += m
                        
                        qty -= m
                        bq_list[0] -= m
                        if bq_list[0] <= 0: buy_q.pop(0)
                        
                    if total_matched_qty > 0:
                        df.loc[idx, 'holding_period'] = matched_weighted_hold / total_matched_qty
                        
                    if qty > 0: sell_q.append([qty, price, t])
        
        # Save output for diagnostics
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, f"{trader_name}_paired_trades_detailed.csv"), index=False)
        return df

    def get_nifty_data(self, df: pd.DataFrame) -> Dict:
        """Standardized NIFTY retrieval for charts"""
        try:
            dates = pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d").unique().tolist()
            if not dates: return {}
            query = "SELECT date, close FROM stock_data WHERE symbol = 'NIFTY' AND date IN %s"
            db_df = pd.read_sql(query, self.engine, params=(tuple(dates),))
            db_df['date'] = pd.to_datetime(db_df['date']).dt.strftime("%Y-%m-%d")
            return dict(zip(db_df['date'], db_df['close']))
        except Exception as e:
            self.logger.error(f"❌ NIFTY data error: {e}")
            return {}

    def _parse_option_symbol(self, symbol: str) -> Optional[Dict]:
        try:
            s = str(symbol).upper().strip()
            parts = s.split()
            if len(parts) >= 3 and parts[1] in ("CE", "PE"):
                # "IO CE BANKNIFTY 31Jul2025 59500" -> parts[2] is underlying
                expiry = datetime.strptime(parts[3], "%d%b%Y").date() if len(parts) > 3 and any(m in parts[3] for m in ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]) else None
                # Alternative format check if expiry is parts[0]
                if not expiry:
                    try: expiry = datetime.strptime(parts[0], "%d%b%Y").date()
                    except: pass
                
                strike = float(parts[2]) if parts[2].replace('.','',1).isdigit() else 0.0
                if strike == 0.0 and len(parts) > 4: # Format: IO CE BANKNIFTY 31Jul2025 59500
                    strike = float(parts[4]) if parts[4].replace('.','',1).isdigit() else 0.0

                opt_type = parts[1]
                underlying = parts[2]
                return {"expiry": expiry, "strike": strike, "type": opt_type, "underlying": underlying}
        except: pass
        return None

    def classify_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each symbol's current position status"""
        df = df.copy()
        df["_signed_qty"] = df.apply(lambda r: r["quantity"] if r["transaction_type"].upper() == "BUY" else -r["quantity"], axis=1)
        net_map = df.groupby("symbol")["_signed_qty"].sum().to_dict()
        
        today = date.today()
        for sym, q in net_map.items():
            if q == 0: continue
            opt = self._parse_option_symbol(sym)
            if opt and today > opt["expiry"]: net_map[sym] = 0.0
                
        df["symbol_net_qty"] = df["symbol"].map(net_map)
        df["position_status"] = df["symbol_net_qty"].apply(lambda q: "OPEN" if q != 0 else "CLOSED")
        df["position_type"] = df.apply(lambda r: ("LONG-OPEN" if r["symbol_net_qty"] > 0 else "SHORT-OPEN") if r["position_status"] == "OPEN" else "", axis=1)
        df.drop(columns=["_signed_qty"], inplace=True)
        return df

    def aggregate_daily_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper for daily pnl aggregation"""
        return df.groupby(df['trade_date'].dt.date).agg({'pnl': 'sum', 'trade_value': 'sum', 'symbol': 'count'}).rename(columns={'symbol': 'num_trades'})

    def fetch_minute_candles(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            query = "SELECT symbol, candle_ts, open_price, high_price, low_price, close_price, volume FROM candle_data WHERE symbol = ANY(%s) AND candle_ts >= %s AND candle_ts <= %s"
            return pd.read_sql(query, self.engine, params=(symbols, start_date, end_date))
        except Exception as e:
            self.logger.error(f"❌ Candle fetch error: {e}")
            return pd.DataFrame()

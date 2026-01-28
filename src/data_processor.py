"""
Data Processor Module
Handles data loading, cleaning, and preparation for analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import psycopg2

class TradingDataProcessor:
    """Process and clean trading data from various sources"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def getadditionaldata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get additional data (scores, highs/lows, etc.) from database and merge with input DataFrame"""
        try:
            db_conf = self.config["database"]
            table_name = db_conf.get("table_name", "stock_data")

            conn = psycopg2.connect(
                dbname=db_conf["dbname"],
                user=db_conf["user"],
                password=db_conf["password"],
                host=db_conf["host"],
                port=db_conf.get("port", 5432)
            )

            # --- Fetch relevant columns from DB ---
            query = f"""
                SELECT symbol, date, open, high, low, close, volume,
                       t_score, f_score, total_score,
                       is_52week_high, is_52week_low, is_alltime_high,
                        is_alltime_low, is_event, atr, is_high_volume, is_news, news_category,
                        market_behaviour, chart_charts
                FROM {table_name};
            """
            db_df = pd.read_sql(query, conn)
            conn.close()

            # Normalize column names
            db_df.columns = db_df.columns.str.lower()

            # Convert date formats to comparable values
            db_df["date"] = pd.to_datetime(db_df["date"], errors="coerce").dt.date
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date

            # --- Preprocess symbols for faster match ---
            # Extract possible base symbol tokens (first 4 space-separated parts)
            df_tokens = df["symbol"].str.upper().str.split(" ").apply(lambda x: x[:4] if isinstance(x, list) else [])
            df = df.assign(
                token_0=df_tokens.str[0],
                token_1=df_tokens.str[1],
                token_2=df_tokens.str[2],
                token_3=df_tokens.str[3]
            )

            db_df["symbol"] = db_df["symbol"].str.upper()

            # --- Create index for fast lookup ---
            db_df.set_index(["symbol", "date"], inplace=True)
            db_df.sort_index(level=["symbol", "date"], inplace=True)
            merged_rows = []
            for i, row in df.iterrows():
                trade_date = row["trade_date"]
                candidates = []

                for t in [row.get("token_0"), row.get("token_1"), row.get("token_2"), row.get("token_3")]:
                    if pd.notna(t):
                        try:
                            candidates.append(db_df.loc[(t, trade_date)])
                            break
                        except KeyError:
                            pass

                if candidates:
                    # Convert row to DataFrame
                    left_df = pd.DataFrame([row])

                    # Convert candidate row(s) to DataFrame
                    right = candidates[0]
                    if isinstance(right, pd.Series):
                        right_df = pd.DataFrame([right])
                    else:
                        right_df = right.reset_index(drop=True)  # already DataFrame

                    merged_data = pd.concat([left_df.reset_index(drop=True), right_df], axis=1)


                else:
                    merged_data = row.to_frame().T.reset_index(drop=True)

                merged_rows.append(merged_data)

            final_df = pd.concat(merged_rows, ignore_index=True)
            final_df.drop(columns=["token_0", "token_1", "token_2", "token_3"], inplace=True, errors="ignore")
            self.logger.info(f"✅ Added additional DB data to {len(df)} records.")
            return final_df

        except Exception as e:
            self.logger.error(f"❌ Error getting additional data: {str(e)}")
            raise

    def get_nifty_data(self, df):
        try:
            db_conf = self.config["database"]
            table_name = db_conf.get("table_name", "stock_data")

            conn = psycopg2.connect(
                dbname=db_conf["dbname"],
                user=db_conf["user"],
                password=db_conf["password"],
                host=db_conf["host"],
                port=db_conf.get("port", 5432)
            )

            # Extract unique dates AND convert to string
            dates = df['Date'].dt.strftime("%m/%d/%Y").unique().tolist()

            # SQL with placeholders
            query = f"""
                SELECT date, close
                FROM {table_name}
                WHERE symbol = 'NIFTY' AND date IN %s;
            """

            # Execute query
            db_df = pd.read_sql(query, conn, params=(tuple(dates),))


            conn.close()

            # Convert DB dates to string as well
            db_df['date'] = pd.to_datetime(db_df['date']).dt.strftime("%Y-%m-%d")
            db_df = db_df.sort_values("date")
            # Build dict: {"2024-01-01": close}
            nifty_data = dict(zip(db_df['date'], db_df['close']))
            # print(nifty_data)
            return nifty_data

        except Exception as e:
            print("Error:", e)
            return {}, {}

    def _enrich_from_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich DF with data from 'market_data' database:
        1. nse_stock: dist_from_52w_high_pct, pct_chg_1w, pct_chg_1m, pct_chg_6m
        2. nse_index_ind: NIFTY, NIFTYMIDCAP150, NIFTYSMLCAP250 context
        """
        if df.empty:
            return df

        try:
            # Re-use config but switch DB name
            db_conf = self.config["database"].copy()
            db_conf["dbname"] = "market_data"

            unique_dates = df['trade_date'].dropna().unique()
            if len(unique_dates) == 0:
                return df
                
            # Convert numpy dates to python dates/strings for SQL
            date_strs = [str(d) for d in unique_dates]
            date_tuple = tuple(date_strs)
            
            # Identify symbols to fetch from nse_stock
            # Clean symbols to match DB (e.g. "RELIANCE" from "RELIANCE EQ")
            # This is a heuristic; might need adjustment based on valid symbols
            symbols = df['symbol'].unique().tolist()
            # Try to match raw symbols or standard tokens
            # For efficiency in SQL, we might just fetch all for the DATES.
            # Assuming nse_stock isn't massive for just a few days of data.
            
            conn = psycopg2.connect(
                dbname=db_conf["dbname"],
                user=db_conf["user"],
                password=db_conf["password"],
                host=db_conf["host"],
                port=db_conf.get("port", 5432)
            )

            # --- 1. Fetch nse_stock data (Momentum Metrics) ---
            # We fetch for ALL available stocks on these DATES to ensure better matching
            # filtering by symbol list in SQL can be tricky with partial matches.
            query_stock = f"""
                SELECT symbol, date, 
                       dist_from_52w_high_pct, pct_chg_1w, pct_chg_1m, pct_chg_6m
                FROM nse_stock
                WHERE date IN %s
            """
            
            # Handle single date tuple quirk in Python (x,)
            params = (date_tuple,)
            
            mkt_stock_df = pd.read_sql(query_stock, conn, params=params)
             
            if not mkt_stock_df.empty:
                mkt_stock_df['date'] = pd.to_datetime(mkt_stock_df['date']).dt.date
                # Pre-process match keys
                # We'll simple-match on 'symbol' (uppercase) and 'date'
                # But df['symbol'] might be 'RELIANCE EQ'. 
                # We'll try to extract the first token from df['symbol'] for matching.
                
                # Create a lookup key in main df
                df['__base_sym'] = df['symbol'].astype(str).str.split().str[0].str.upper().str.strip()
                mkt_stock_df['symbol'] = mkt_stock_df['symbol'].str.upper().str.strip()
                
                # Merge
                # We left join to keep all trades
                df = df.merge(
                    mkt_stock_df, 
                    left_on=['__base_sym', 'trade_date'], 
                    right_on=['symbol', 'date'], 
                    how='left', 
                    suffixes=('', '_mkt')
                )
                
                # Cleanup
                df.drop(columns=['__base_sym', 'symbol_mkt', 'date_mkt', 'date'], inplace=True, errors='ignore')

            # --- 2. Fetch nse_index_ind data (Market Context) ---
            # We need specific indices
            indices = ['NIFTY', 'NIFTYMIDCAP150', 'NIFTYSMLCAP250']
            
            query_index = f"""
                SELECT symbol, date, close, pct_chg_1w, ema_50, ema_200
                FROM nse_index_ind
                WHERE symbol IN %s AND date IN %s
            """
            
            mkt_index_df = pd.read_sql(query_index, conn, params=(tuple(indices), date_tuple))
            conn.close()

            if not mkt_index_df.empty:
                mkt_index_df['date'] = pd.to_datetime(mkt_index_df['date']).dt.date
                
                # Pivot this data so each Date has columns like:
                # nifty_close, midcap_close, etc.
                
                # Map DB symbols to readable prefixes
                sym_map = {
                    'NIFTY': 'nifty50',
                    'NIFTYMIDCAP150': 'midcap',
                    'NIFTYSMLCAP250': 'smlcap'
                }
                mkt_index_df['prefix'] = mkt_index_df['symbol'].map(sym_map).fillna('other')
                
                # We want to pivot on 'date'
                # Columns to pivot: close, pct_chg_1w, ema_50, ema_200
                
                pivoted_dfs = []
                for idx_sym, prefix in sym_map.items():
                    subset = mkt_index_df[mkt_index_df['symbol'] == idx_sym].copy()
                    if subset.empty:
                        continue
                        
                    subset = subset[['date', 'close', 'pct_chg_1w', 'ema_50', 'ema_200']]
                    subset.columns = ['date', f'{prefix}_close', f'{prefix}_pct_chg_1w', f'{prefix}_ema_50', f'{prefix}_ema_200']
                    pivoted_dfs.append(subset)
                
                if pivoted_dfs:
                    # Merge all index data info a single market_context_df per date
                    from functools import reduce
                    market_ctx = reduce(
                        lambda left, right: pd.merge(left, right, on='date', how='outer'), 
                        pivoted_dfs
                    )
                    
                    # Merge into main df
                    df = df.merge(market_ctx, left_on='trade_date', right_on='date', how='left')
                    df.drop(columns=['date'], inplace=True, errors='ignore')

            self.logger.info("✅ Enriched with Market Data (NSE Stock + Indices)")
            return df

        except Exception as e:
            self.logger.error(f"❌ Error enriching market data: {str(e)}")
            # Return original df on failure to avoid pipeline break
            return df


    # =========================================================
    # Data Loading
    # =========================================================
    def load_data(self, filepath: str, source_type: str = "csv") -> pd.DataFrame:
        """Load trading data from file + enrich + clean + classify positions."""
        try:
            if source_type == "csv":
                df = pd.read_csv(filepath)
            elif source_type == "excel":
                df = pd.read_excel(filepath)
            elif source_type == "json":
                df = pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            self.logger.info(f"📂 Loaded {len(df)} records from {filepath}")

            # 1) Add DB scores/technicals (stock_db)
            df = self.getadditionaldata(df)

            # 1.5) Add Market Data (market_data DB: nse_stock, nse_index_ind)
            df = self._enrich_from_market_data(df)

            # 2) Clean & normalize
            df = self.clean_data(df)

            # 3) Classify positions (OPEN/CLOSED and LONG-OPEN/SHORT-OPEN)
            df = self.classify_positions(df)

            return df

        except Exception as e:
            self.logger.error(f"❌ Error loading data: {str(e)}")
            raise

    # =========================================================
    # Data Validation
    # =========================================================
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data has required columns and structure"""
        required_cols = self.config["data"]["required_columns"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return False, missing_cols
        return True, []

    # =========================================================
    # Data Cleaning
    # =========================================================
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess trading data"""
        df = df.copy()

        # Convert to datetime safely
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

        # Drop rows with missing or invalid trade dates
        missing_dates = df["trade_date"].isna().sum()
        if missing_dates > 0:
            self.logger.warning(f"⚠️ Dropping {missing_dates} rows with invalid or missing trade_date values.")
            df = df.dropna(subset=["trade_date"])

        # Drop missing essential fields
        df = df.dropna(subset=["symbol", "price", "quantity"])

        # Derived columns
        df["trade_hour"] = df["trade_date"].dt.hour
        df["trade_day_of_week"] = df["trade_date"].dt.dayofweek
        df["trade_month"] = df["trade_date"].dt.month

        # Compute trade value if absent
        if "trade_value" not in df.columns:
            df["trade_value"] = df["price"] * df["quantity"]

        # Sort chronologically
        df = df.sort_values("trade_date").reset_index(drop=True)

        self.logger.info(f"🧹 Cleaned data: {len(df)} records")
        return df

    # =========================================================
    # Trade Pairing (FIFO-based P&L)
    # =========================================================
    from pathlib import Path
    import pandas as pd

    def pair_trades(
            self,
            df: pd.DataFrame,
            output_dir: str = "data/trade_exports",
            trader_name: str = "Trader",
    ) -> pd.DataFrame:
        """
        FIFO trade pairing.
        PnL is computed STRICTLY as (sell_price - buy_price) * matched_qty
        Open trades carry pnl = 0 (no mark-to-market).
        """

        df = df.copy()
        df["transaction_type"] = df["transaction_type"].str.upper()
        df["pnl"] = 0.0
        df["holding_period_minutes"] = 0.0

        df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        trade_pairs = []

        for symbol, symbol_df in df.groupby("symbol", sort=False):

            buy_queue = []  # [qty, price, date, t_score, f_score]
            sell_queue = []  # [qty, price, date, t_score, f_score]

            for idx, row in symbol_df.iterrows():
                typ = row["transaction_type"]
                qty = float(row["quantity"])
                price = float(row["price"])
                date = row["trade_date"]
                t_score = row.get("t_score")
                f_score = row.get("f_score")

                # ===================== BUY =====================
                if typ == "BUY":
                    remaining = qty

                    while remaining > 0 and sell_queue:
                        s_qty, s_price, s_date, s_t, s_f = sell_queue[0]
                        matched = min(remaining, s_qty)

                        pnl = (s_price - price) * matched
                        hold = (date - s_date).total_seconds() / 60

                        trade_pairs.append({
                            "symbol": symbol,
                            "buy_date": date,
                            "buy_price": price,
                            "buy_qty": matched,
                            "sell_date": s_date,
                            "sell_price": s_price,
                            "sell_qty": matched,
                            "pnl": pnl,
                            "holding_period_minutes": hold,
                            "status": "CLOSED",
                        })

                        df.loc[idx, "pnl"] += pnl
                        df.loc[idx, "holding_period_minutes"] += hold

                        remaining -= matched
                        s_qty -= matched

                        if s_qty == 0:
                            sell_queue.pop(0)
                        else:
                            sell_queue[0][0] = s_qty

                    if remaining > 0:
                        buy_queue.append([remaining, price, date, t_score, f_score])

                # ===================== SALE =====================
                elif typ == "SALE":
                    remaining = qty

                    while remaining > 0 and buy_queue:
                        b_qty, b_price, b_date, b_t, b_f = buy_queue[0]
                        matched = min(remaining, b_qty)

                        pnl = (price - b_price) * matched
                        hold = (date - b_date).total_seconds() / 60

                        trade_pairs.append({
                            "symbol": symbol,
                            "buy_date": b_date,
                            "buy_price": b_price,
                            "buy_qty": matched,
                            "sell_date": date,
                            "sell_price": price,
                            "sell_qty": matched,
                            "pnl": pnl,
                            "holding_period_minutes": hold,
                            "status": "CLOSED",
                        })

                        df.loc[idx, "pnl"] += pnl
                        df.loc[idx, "holding_period_minutes"] += hold

                        remaining -= matched
                        b_qty -= matched

                        if b_qty == 0:
                            buy_queue.pop(0)
                        else:
                            buy_queue[0][0] = b_qty

                    if remaining > 0:
                        sell_queue.append([remaining, price, date, t_score, f_score])

            # ========== OPEN POSITIONS (NO MTM HERE) ==========
            for qty, price, date, _, _ in buy_queue:
                trade_pairs.append({
                    "symbol": symbol,
                    "buy_date": date,
                    "buy_price": price,
                    "buy_qty": qty,
                    "sell_date": None,
                    "sell_price": 0,
                    "sell_qty": qty,
                    "pnl": (0-price)*qty,
                    "holding_period_minutes": 0.0,
                    "status": "OPEN",
                })

            for qty, price, date, _, _ in sell_queue:
                trade_pairs.append({
                    "symbol": symbol,
                    "buy_date": None,
                    "buy_price": 0,
                    "buy_qty": qty,
                    "sell_date": date,
                    "sell_price": price,
                    "sell_qty": qty,
                    "pnl": (price-0)*qty,
                    "holding_period_minutes": 0.0,
                    "status": "OPEN",
                })

        # ================= EXPORTS =================
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        detailed_file = output_path / f"{trader_name}_paired_trades_detailed.csv"
        summary_file = output_path / f"{trader_name}_trade_pairs_summary.csv"

        df.to_csv(detailed_file, index=False)
        self.logger.info(f"📊 Detailed trades saved: {detailed_file}")

        summary_df = pd.DataFrame(trade_pairs)
        summary_df.to_csv(summary_file, index=False)
        self.logger.info(f"📘 Trade pairs saved: {summary_file}")

        realized_pnl = summary_df["pnl"].sum()
        self.logger.info(f"✅ Realized FIFO PnL: ₹{realized_pnl:,.2f}")

        return df

    # =========================================================
    # Daily Aggregation
    # =========================================================
    def aggregate_daily_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate trades by day"""
        daily_stats = (
            df.groupby(df["trade_date"].dt.date)
            .agg(
                {
                    "trade_value": "sum",
                    "pnl": "sum",
                    "symbol": "count",
                    "quantity": "sum",
                }
            )
            .rename(columns={"symbol": "num_trades"})
        )
        return daily_stats

    # def classify_positions(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Classify each symbol into OPEN / CLOSED positions
    #     and each trade row into OPEN-LEG or CLOSED-LEG.
    #
    #     Ensures transaction_type consistency and removes ambiguity for MetricsCalculator.
    #     """
    #     df = df.copy()
    #     df["transaction_type"] = df["transaction_type"].str.upper()
    #
    #     # ----- Compute net qty at SYMBOL level -----
    #     df["_signed_qty"] = df.apply(
    #         lambda r: r["quantity"] if r["transaction_type"] == "BUY"
    #         else -r["quantity"],
    #         axis=1
    #     )
    #
    #     symbol_net = df.groupby("symbol")["_signed_qty"].sum().rename("symbol_net_qty")
    #
    #     # merge net qty into df
    #     df = df.merge(symbol_net, left_on="symbol", right_index=True, how="left")
    #
    #     # ----- Position status -----
    #     df["position_status"] = df["symbol_net_qty"].apply(
    #         lambda q: "OPEN" if q != 0 else "CLOSED"
    #     )
    #
    #     # ----- Long / Short / Mixed classification -----
    #     # long if buys > sells, short if sells > buys
    #
    #     def classify_direction(g):
    #         buy_qty = g.loc[g["transaction_type"] == "BUY", "quantity"].sum()
    #         sell_qty = g.loc[g["transaction_type"] == "SALE", "quantity"].sum()
    #
    #         if buy_qty > sell_qty:
    #             return "LONG"
    #         elif sell_qty > buy_qty:
    #             return "SHORT"
    #         else:
    #             return "MIXED"  # equal qty or hedged structure
    #
    #     direction_map = df.groupby("symbol").apply(classify_direction).rename("long_short_type")
    #     df = df.merge(direction_map, left_on="symbol", right_index=True, how="left")
    #
    #     # ----- Mark each row as OPEN-LEG / CLOSED-LEG -----
    #     df["row_trade_category"] = df.apply(
    #         lambda r: "OPEN-LEG" if r["position_status"] == "OPEN" else "CLOSED-LEG",
    #         axis=1
    #     )
    #
    #     # cleanup
    #     df.drop(columns=["_signed_qty"], inplace=True)
    #
    #     return df

    def classify_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each symbol into OPEN/CLOSED and LONG-OPEN/SHORT-OPEN
        based on net quantity (BUY - SELL).
        This removes ambiguity for MetricsCalculator.
        """
        df = df.copy()
        df["transaction_type"] = df["transaction_type"].astype(str).str.upper()

        # Treat both 'SALE' and 'SELL' as sell side
        def signed_qty(row):
            typ = row["transaction_type"]
            if typ == "BUY":
                return row["quantity"]
            elif typ in ("SALE", "SELL"):
                return -row["quantity"]
            return 0.0

        df["_signed_qty"] = df.apply(signed_qty, axis=1)

        # --- SYMBOL-LEVEL NET QTY ---
        symbol_net = df.groupby("symbol")["_signed_qty"].sum().rename("symbol_net_qty")
        df = df.merge(symbol_net, left_on="symbol", right_index=True, how="left")


        #--- add the logic to close the net option sell derivatives to 0 if they are out of money , given we have to find the stock spot and compute expiry spot and then on day of expiry we have to make it 0 if the open positions is not closed after expiry date. ---

        # --- POSITION STATUS: OPEN/CLOSED ---
        df["position_status"] = df["symbol_net_qty"].apply(
            lambda q: "OPEN" if q != 0 else "CLOSED"
        )

        # --- POSITION TYPE: LONG-OPEN / SHORT-OPEN (only meaningful when OPEN) ---
        def classify_pos_type(q, status):
            if status == "OPEN":
                if q > 0:
                    return "LONG-OPEN"
                elif q < 0:
                    return "SHORT-OPEN"
            return ""  # for CLOSED or zero

        df["position_type"] = df.apply(
            lambda r: classify_pos_type(r["symbol_net_qty"], r["position_status"]),
            axis=1,
        )

        # Optional: mark each row as part of an open-leg or closed-leg ecosystem
        df["row_trade_category"] = df["position_status"].map(
            lambda s: "OPEN-LEG" if s == "OPEN" else "CLOSED-LEG"
        )

        df.drop(columns=["_signed_qty"], inplace=True)
        return df

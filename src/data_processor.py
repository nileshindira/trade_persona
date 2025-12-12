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
                       is_52week_high, is_52week_low, is_alltime_high, is_alltime_low
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

            # Build dict: {"2024-01-01": close}
            nifty_data = dict(zip(db_df['date'], db_df['close']))
            # print(nifty_data)
            return nifty_data

        except Exception as e:
            print("Error:", e)
            return {}, {}


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

            # 1) Add DB scores/technicals
            df = self.getadditionaldata(df)

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
    def pair_trades(self, df: pd.DataFrame, output_dir: str = "data/trade_exports", trader_name: str = "Trader") -> pd.DataFrame:

        """
        Pair BUY and SALE trades correctly for realized P&L using FIFO matching.
        Exports three files:
          1. paired_trades_detailed.csv  -> raw trades with P&L per row
          2. realized_trades_only.csv    -> only closed trades (nonzero P&L)
          3. trade_pairs_summary.csv     -> paired BUY–SALE rows with full details
        """
        df = df.copy()
        df["pnl"] = 0.0
        df["holding_period_minutes"] = 0.0
        df["transaction_type"] = df["transaction_type"].astype(str).str.upper()
        df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        all_unclosed_buys, all_unclosed_sales = [], []
        trade_pairs = []  # to store paired trade summaries

        for symbol in df["symbol"].unique():
            symbol_mask = df["symbol"] == symbol
            symbol_trades = df.loc[symbol_mask].copy()
            buy_stack, sale_stack = [], []

            for i, row in symbol_trades.iterrows():
                typ, qty, price, tdate, t_t_score, t_f_score = (
                    row["transaction_type"],
                    float(row["quantity"]),
                    float(row["price"]),
                    row["trade_date"],
                    row["t_score"],
                    row["f_score"]
                )

                # ===== LONG TRADES =====
                if typ == "BUY":
                    remaining_qty, pnl_total, hold_total, hold_count = qty, 0.0, 0.0, 0
                    while remaining_qty > 0 and sale_stack:
                        s_qty, s_price, s_date, s_t_score, s_f_score = sale_stack[0]
                        matched = min(s_qty, remaining_qty)
                        pnl = (s_price - price) * matched
                        hold = (tdate - s_date).total_seconds() / 60
                        pnl_total += pnl
                        hold_total += hold
                        hold_count += 1

                        # record trade pair
                        trade_pairs.append(
                            {
                                "symbol": symbol,
                                "buy_date": s_date,
                                "buy_price": s_price,
                                "buy_qty": matched,
                                "buy_score": t_t_score,
                                "buy_f_score": t_f_score,
                                "sell_date": tdate,
                                "sell_price": price,
                                "sell_qty": matched,
                                "sell_score": s_t_score,
                                "sell_f_score": s_f_score,
                                "pnl": pnl,
                                "status": "CLOSED",
                            }
                        )

                        s_qty -= matched
                        remaining_qty -= matched
                        if s_qty == 0:
                            sale_stack.pop(0)
                        else:
                            sale_stack[0][0] = s_qty
                            sale_stack[0][3] = s_t_score
                            sale_stack[0][4] = s_f_score

                    if pnl_total != 0:
                        df.loc[i, "pnl"] = pnl_total
                        df.loc[i, "holding_period_minutes"] = hold_total / max(hold_count, 1)
                    if remaining_qty > 0:
                        buy_stack.append([remaining_qty, price, tdate, t_t_score, t_f_score])

                # ===== SHORT TRADES =====
                elif typ == "SALE":
                    remaining_qty, pnl_total, hold_total, hold_count = qty, 0.0, 0.0, 0
                    while remaining_qty > 0 and buy_stack:
                        b_qty, b_price, b_date, b_t_score, b_f_score = buy_stack[0]
                        matched = min(b_qty, remaining_qty)
                        pnl = (price - b_price) * matched
                        hold = (tdate - b_date).total_seconds() / 60
                        pnl_total += pnl
                        hold_total += hold
                        hold_count += 1

                        trade_pairs.append(
                            {
                                "symbol": symbol,
                                "buy_date": b_date,
                                "buy_price": b_price,
                                "buy_qty": matched,
                                "buy_score": b_t_score,
                                "buy_f_score": b_f_score,
                                "sell_date": tdate,
                                "sell_price": price,
                                "sell_qty": matched,
                                "sell_score": t_t_score,
                                "sell_f_score": t_f_score,
                                "pnl": pnl,
                                "status": "CLOSED",
                            }
                        )

                        b_qty -= matched
                        remaining_qty -= matched
                        if b_qty == 0:
                            buy_stack.pop(0)

                        else:
                            buy_stack[0][0] = b_qty
                            buy_stack[0][3] = b_t_score
                            buy_stack[0][4] = b_f_score

                    if pnl_total != 0:
                        df.loc[i, "pnl"] = pnl_total
                        df.loc[i, "holding_period_minutes"] = hold_total / max(hold_count, 1)
                    if remaining_qty > 0:
                        sale_stack.append([remaining_qty, price, tdate, t_t_score, t_f_score])

            # any leftover buys/sales → open positions
            for qty, price, date, t_score, f_score in buy_stack:
                trade_pairs.append(
                    {
                        "symbol": symbol,
                        "buy_date": date,
                        "buy_price": price,
                        "buy_qty": qty,
                        "buy_score": t_score,
                        "buy_f_score": f_score,
                        "sell_date": None,
                        "sell_price": None,
                        "sell_qty": 0,
                        "pnl": 0,
                        "status": "OPEN",
                    }
                )
            for qty, price, date, t_score, f_score in sale_stack:
                trade_pairs.append(
                    {
                        "symbol": symbol,
                        "buy_date": None,
                        "buy_price": None,
                        "buy_qty": 0,
                        "buy_score": t_score,
                        "buy_f_score": f_score,
                        "sell_date": date,
                        "sell_price": price,
                        "sell_qty": qty,
                        "pnl": 0,
                        "status": "OPEN",
                    }
                )

        # === Fill missing values ===
        df["pnl"] = df["pnl"].fillna(0)
        df["holding_period_minutes"] = df["holding_period_minutes"].fillna(0)

        manual_pnl = (
                (df.loc[df["transaction_type"] == "SALE", "price"] * df["quantity"]).sum()
                - (df.loc[df["transaction_type"] == "BUY", "price"] * df["quantity"]).sum()
        )
        realized_pnl = df["pnl"].sum()

        self.logger.info(f"✅ Manual-style P&L (Σ(SALE−BUY)): ₹{manual_pnl:,.2f}")
        self.logger.info(f"✅ Realized matched P&L (FIFO): ₹{realized_pnl:,.2f}")
        self.logger.info(f"🧮 Difference: ₹{realized_pnl - manual_pnl:,.2f}")

        # === Export files ===
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Trader-specific filenames
        detailed_file = output_path / f"{trader_name}_paired_trades_detailed.csv"
        realized_file = output_path / f"{trader_name}_realized_trades_only.csv"
        summary_file = output_path / f"{trader_name}_trade_pairs_summary.csv"

        # raw trade rows
        df.to_csv(
            detailed_file,
            index=False,
            columns=[
                "symbol",
                "trade_date",
                "transaction_type",
                "quantity",
                "price",
                "pnl",
                "holding_period_minutes",
                "trade_value",
            ],
        )
        self.logger.info(f"📊 Detailed trade file saved: {detailed_file}")

        # realized only
        df[df["pnl"] != 0].to_csv(realized_file, index=False)
        self.logger.info(f"💰 Realized trades file saved: {realized_file}")

        # trade pair summary
        summary_df = pd.DataFrame(trade_pairs)
        summary_df = (
            summary_df.groupby(
                ["symbol", "buy_date", "buy_price", "sell_date", "sell_price", "status"],
                dropna=False,
                as_index=False,
            )
            .agg({"buy_qty": "sum", "sell_qty": "sum", "pnl": "sum"})
        )
        summary_df.sort_values(["symbol", "buy_date", "sell_date"], inplace=True)
        summary_df.to_csv(summary_file, index=False)
        self.logger.info(f"📘 Trade pair summary saved: {summary_file}")

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

    def classify_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each symbol into OPEN / CLOSED positions
        and each trade row into OPEN-LEG or CLOSED-LEG.

        Ensures transaction_type consistency and removes ambiguity for MetricsCalculator.
        """
        df = df.copy()
        df["transaction_type"] = df["transaction_type"].str.upper()

        # ----- Compute net qty at SYMBOL level -----
        df["_signed_qty"] = df.apply(
            lambda r: r["quantity"] if r["transaction_type"] == "BUY"
            else -r["quantity"],
            axis=1
        )

        symbol_net = df.groupby("symbol")["_signed_qty"].sum().rename("symbol_net_qty")

        # merge net qty into df
        df = df.merge(symbol_net, left_on="symbol", right_index=True, how="left")

        # ----- Position status -----
        df["position_status"] = df["symbol_net_qty"].apply(
            lambda q: "OPEN" if q != 0 else "CLOSED"
        )

        # ----- Long / Short / Mixed classification -----
        # long if buys > sells, short if sells > buys

        def classify_direction(g):
            buy_qty = g.loc[g["transaction_type"] == "BUY", "quantity"].sum()
            sell_qty = g.loc[g["transaction_type"] == "SALE", "quantity"].sum()

            if buy_qty > sell_qty:
                return "LONG"
            elif sell_qty > buy_qty:
                return "SHORT"
            else:
                return "MIXED"  # equal qty or hedged structure

        direction_map = df.groupby("symbol").apply(classify_direction).rename("long_short_type")
        df = df.merge(direction_map, left_on="symbol", right_index=True, how="left")

        # ----- Mark each row as OPEN-LEG / CLOSED-LEG -----
        df["row_trade_category"] = df.apply(
            lambda r: "OPEN-LEG" if r["position_status"] == "OPEN" else "CLOSED-LEG",
            axis=1
        )

        # cleanup
        df.drop(columns=["_signed_qty"], inplace=True)

        return df

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

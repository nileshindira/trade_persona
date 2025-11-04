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

            merged_rows = []
            for i, row in df.iterrows():
                trade_date = row["trade_date"]
                candidates = []

                for t in [row.get("token_0"), row.get("token_1"), row.get("token_2"), row.get("token_3")]:
                    if pd.notna(t) and (t, trade_date) in db_df.index:
                        candidates.append(db_df.loc[(t, trade_date)])
                        break

                if candidates:
                    merged_data = pd.concat([row.to_frame().T.reset_index(drop=True), candidates[0].to_frame().T.reset_index(drop=True)], axis=1)
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
    # =========================================================
    # Data Loading
    # =========================================================
    def load_data(self, filepath: str, source_type: str = "csv") -> pd.DataFrame:
        """Load trading data from file"""
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
            df = self.getadditionaldata(df)
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
                typ, qty, price, tdate = (
                    row["transaction_type"],
                    float(row["quantity"]),
                    float(row["price"]),
                    row["trade_date"],
                )

                # ===== LONG TRADES =====
                if typ == "BUY":
                    remaining_qty, pnl_total, hold_total, hold_count = qty, 0.0, 0.0, 0
                    while remaining_qty > 0 and sale_stack:
                        s_qty, s_price, s_date = sale_stack[0]
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
                                "sell_date": tdate,
                                "sell_price": price,
                                "sell_qty": matched,
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

                    if pnl_total != 0:
                        df.loc[i, "pnl"] = pnl_total
                        df.loc[i, "holding_period_minutes"] = hold_total / max(hold_count, 1)
                    if remaining_qty > 0:
                        buy_stack.append([remaining_qty, price, tdate])

                # ===== SHORT TRADES =====
                elif typ == "SALE":
                    remaining_qty, pnl_total, hold_total, hold_count = qty, 0.0, 0.0, 0
                    while remaining_qty > 0 and buy_stack:
                        b_qty, b_price, b_date = buy_stack[0]
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
                                "sell_date": tdate,
                                "sell_price": price,
                                "sell_qty": matched,
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

                    if pnl_total != 0:
                        df.loc[i, "pnl"] = pnl_total
                        df.loc[i, "holding_period_minutes"] = hold_total / max(hold_count, 1)
                    if remaining_qty > 0:
                        sale_stack.append([remaining_qty, price, tdate])

            # any leftover buys/sales → open positions
            for qty, price, date in buy_stack:
                trade_pairs.append(
                    {
                        "symbol": symbol,
                        "buy_date": date,
                        "buy_price": price,
                        "buy_qty": qty,
                        "sell_date": None,
                        "sell_price": None,
                        "sell_qty": 0,
                        "pnl": 0,
                        "status": "OPEN",
                    }
                )
            for qty, price, date in sale_stack:
                trade_pairs.append(
                    {
                        "symbol": symbol,
                        "buy_date": None,
                        "buy_price": None,
                        "buy_qty": 0,
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

import psycopg2
import pandas as pd
import numpy as np
from clickhouse_connect import get_client


PG_CONN = {
    "host": "localhost",
    "dbname": "stockdb",
    "user": "postgres",
    "password": "postgres",
    "port": 5432
}

CLICKHOUSE_CONN = {
    "host": "localhost",
    "username": "default",
    "password": "indira",
    "database": "market_data"
}

SOURCE_TABLE = "stock_data"
TARGET_TABLE = "eod_full"
BATCH_SIZE = 100_000


def setup_clickhouse():
    print("Connecting to ClickHouse...")

    ch = get_client(**CLICKHOUSE_CONN)

    db = ch.command("SELECT currentDatabase()")
    print("Connected ClickHouse DB:", db)

    if db != "market_data":
        raise Exception("Not in correct database!")

    ch.command("""
    CREATE TABLE IF NOT EXISTS market_data.eod_full
    (
        symbol String,
        trade_date Date,
        open Float32, high Float32, low Float32, close Float32,
        volume UInt64,
        dma_10 Float32, dma_21 Float32, dma_50 Float32, dma_100 Float32,
        ema_10 Float32, ema_21 Float32, ema_50 Float32, ema_100 Float32,
        t_score Float32, f_score Float32,
        is_52week_high UInt8, is_52week_low UInt8,
        is_alltime_high UInt8, is_alltime_low UInt8,
        total_score Float32,
        macd Float32,
        macd_signal Float32
    )
    ENGINE = MergeTree
    PARTITION BY toYYYYMM(trade_date)
    ORDER BY (symbol, trade_date)
    """)

    print("Table check done.")
    return ch


def migrate_data(ch):
    print("Connecting to Postgres...")
    pg = psycopg2.connect(**PG_CONN)

    count_cur = pg.cursor()
    count_cur.execute(f"SELECT COUNT(*) FROM {SOURCE_TABLE}")
    total_rows = count_cur.fetchone()[0]
    count_cur.close()

    print(f"Total rows to migrate: {total_rows:,}\n")

    cur = pg.cursor(name="stream_cursor")
    cur.itersize = BATCH_SIZE
    cur.execute(f"SELECT * FROM {SOURCE_TABLE}")

    EXPECTED_COLS = [
        "symbol", "trade_date",
        "open", "high", "low", "close", "volume",
        "dma_10", "dma_21", "dma_50", "dma_100",
        "ema_10", "ema_21", "ema_50", "ema_100",
        "t_score", "f_score",
        "is_52week_high", "is_52week_low",
        "is_alltime_high", "is_alltime_low",
        "total_score",
        "macd", "macd_signal"
    ]

    rows_done = 0

    while True:
        batch = cur.fetchmany(BATCH_SIZE)
        if not batch:
            break

        df = pd.DataFrame(batch, columns=[c.name for c in cur.description])

        df.rename(columns={"date": "trade_date"}, inplace=True)

        # FIX NULL macd_signal
        if "macd_signal" not in df.columns:
            df["macd_signal"] = 0.0
        else:
            df["macd_signal"] = df["macd_signal"].fillna(0.0)

        # Date convert
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

        # Boolean → UInt8
        for col in ["is_52week_high", "is_52week_low", "is_alltime_high", "is_alltime_low"]:
            df[col] = df[col].astype("uint8")

        # Float32 conversion
        float_cols = [
            "open", "high", "low", "close",
            "dma_10", "dma_21", "dma_50", "dma_100",
            "ema_10", "ema_21", "ema_50", "ema_100",
            "t_score", "f_score", "total_score", "macd", "macd_signal"
        ]
        for col in float_cols:
            df[col] = df[col].astype("float32")

        # Volume
        df["volume"] = df["volume"].astype("uint64")

        # Order columns exactly
        df = df[EXPECTED_COLS]

        # Insert
        ch.insert_df(TARGET_TABLE, df)

        rows_done += len(df)
        print(f"Migrated {rows_done:,}/{total_rows:,} rows...")

    pg.close()
    print("\nMigration Complete!")


if __name__ == "__main__":
    ch = setup_clickhouse()
    migrate_data(ch)

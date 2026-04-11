
import pandas as pd
import yaml
from src.metrics_calculator import TradingMetricsCalculator
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

config = {
    'metrics': {
        'risk_free_rate': 0.065,
        'trading_days_per_year': 252
    }
}

calc = TradingMetricsCalculator(config)
df = pd.read_csv('data/sample_trades.csv')
# Simulate pairing
df['trade_date'] = pd.to_datetime(df['trade_date'])
df['pnl'] = df['trade_value'] * 0.01 
df['price'] = df['price'] 
df['quantity'] = df['quantity']
df['trade_value'] = df['price'] * df['quantity']
# Add missing columns for metrics
df['position_status'] = 'CLOSED'
df['position_type'] = 'LONG-OPEN'

print("Calculating all metrics...")
metrics = calc.calculate_all_metrics(df, None, patterns={})
print("Done!")
print(f"Total Trades: {metrics['total_trades']}")

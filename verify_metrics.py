
import pandas as pd
import yaml
from src.metrics_calculator import TradingMetricsCalculator
import logging

logging.basicConfig(level=logging.INFO)

config = {
    'metrics': {
        'risk_free_rate': 0.065,
        'trading_days_per_year': 252
    }
}

calc = TradingMetricsCalculator(config)
df = pd.read_csv('data/sample_trades.csv')
# Simulate pairing (add pnl, trade_date)
df['trade_date'] = pd.to_datetime(df['trade_date'])
df['pnl'] = df['trade_value'] * 0.01 # Fake PnL
df['pcol'] = 'pnl'

print("Calculating metrics...")
# We need patterns
patterns = {}
metrics = calc.calculate_all_metrics(df, None, patterns=patterns)
print("Done!")
print(f"Total Trades: {metrics['total_trades']}")
print(f"Win Rate: {metrics['win_rate']}")

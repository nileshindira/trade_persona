
import pandas as pd
import yaml
from src.metrics_calculator import TradingMetricsCalculator
from src.data_processor import TradingDataProcessor
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

config = {
    'metrics': {
        'risk_free_rate': 0.065,
        'trading_days_per_year': 252
    }
}

processor = TradingDataProcessor(config)
calc = TradingMetricsCalculator(config)

raw_df = pd.read_csv('data/sample_trades.csv')
df = processor.clean_data(raw_df)

print("Calculating all metrics...")
metrics = calc.calculate_all_metrics(df, None, patterns={})
print("Done!")
print(f"Total Trades: {metrics['total_trades']}")

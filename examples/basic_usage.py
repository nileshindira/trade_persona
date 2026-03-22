#!/usr/bin/env python3
"""
Basic Usage Example
Demonstrates how to use the Trading Persona Analyzer
"""

from src.data_processor import TradingDataProcessor
from src.metrics_calculator import TradingMetricsCalculator
from src.pattern_detector import TradingPatternDetector
from src.llm_analyzer import LLMAnalyzer
from src.report_generator import ReportGenerator
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
data_processor = TradingDataProcessor(config)
metrics_calculator = TradingMetricsCalculator(config)
pattern_detector = TradingPatternDetector(config)
llm_analyzer = LLMAnalyzer(config)
report_generator = ReportGenerator(config)

# Process data
print("Loading and processing data...")
df = data_processor.load_data('data/sample_trades.csv')
df = data_processor.clean_data(df)
df = data_processor.pair_trades(df)

# Calculate metrics
print("Calculating metrics...")
metrics = metrics_calculator.calculate_all_metrics(df)

print(f"\nTrading Metrics:")
print(f"Total Trades: {metrics['total_trades']}")
print(f"Win Rate: {metrics['win_rate']:.2f}%")
print(f"Net P&L: ₹{metrics['total_pnl']:,.2f}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")

# Detect patterns
print("\nDetecting patterns...")
patterns = pattern_detector.detect_all_patterns(df)

print(f"\nDetected Patterns:")
for pattern_name, pattern_data in patterns.items():
    if isinstance(pattern_data, dict) and pattern_data.get('detected'):
        print(f"  ⚠️  {pattern_name.replace('_', ' ').title()}: Detected")

# Generate AI analysis (optional - requires Ollama running)
try:
    print("\nGenerating AI analysis...")
    analysis = llm_analyzer.generate_analysis(metrics, patterns)
    print(f"\nAI Insights:")
    print(f"Trader Profile: {analysis['trader_profile'][:150]}...")
except Exception as e:
    print(f"\nSkipping AI analysis (Ollama not running): {e}")
    analysis = {
        'trader_profile': 'N/A',
        'risk_assessment': 'N/A',
        'behavioral_insights': 'N/A',
        'recommendations': 'N/A',
        'performance_summary': 'N/A'
    }

# Generate report
print("\nGenerating report...")
report = report_generator.generate_report(metrics, patterns, analysis, "Example Trader")

print(f"\nReport Summary:")
print(f"Risk Score: {report['risk_score']}/100")
print(f"Risk Level: {report['executive_summary']['risk_level']}")
print(f"Trading Style: {report['executive_summary']['trading_style']}")

print("\n✅ Analysis complete!")

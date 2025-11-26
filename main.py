#!/usr/bin/env python3
"""
Trading Persona Analyzer - Main Application
Analyze trading patterns and generate comprehensive insights using local LLMs
"""

import yaml
import logging
import argparse
from pathlib import Path
import sys

from src.data_processor import TradingDataProcessor
from src.metrics_calculator import TradingMetricsCalculator
from src.pattern_detector import TradingPatternDetector
from src.llm_analyzer import OllamaAnalyzer
from src.report_generator import ReportGenerator
from src.ema_calculator import EMACalculator
from datetime import datetime, date
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingPersonaAnalyzer:
    """Main application class for trading analysis"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_processor = TradingDataProcessor(self.config)
        self.metrics_calculator = TradingMetricsCalculator(self.config)
        self.pattern_detector = TradingPatternDetector(self.config)
        self.llm_analyzer = OllamaAnalyzer(self.config)
        self.report_generator = ReportGenerator(config=self.config)

        # Initialize EMA calculator
        # try:
            # self.ema_calculator = EMACalculator(self.config)
            # self.ema_enabled = True
            # logger.info("EMA calculator initialized successfully")
        # except Exception as e:
        #     logger.warning(f"EMA calculator initialization failed: {str(e)}. EMA scores will be skipped.")
        #     self.ema_enabled = False



    def analyze(self, data_filepath: str, trader_name: str = "Trader",
                output_dir: str = "data/reports", include_ema: bool = True):
        """Run complete analysis pipeline"""

        logger.info(f"Starting analysis for {trader_name}")

        # Step 1: Load and process data
        logger.info("Loading data...")
        df = self.data_processor.load_data(data_filepath)
        nifty_chart_data = self.data_processor.get_nifty_data(df)

        # Validate data
        is_valid, missing_cols = self.data_processor.validate_data(df)
        if not is_valid:
            logger.error(f"Missing required columns: {missing_cols}")
            return None

        # Clean data
        logger.info("Cleaning data...")
        df = self.data_processor.clean_data(df)

        # Pair trades for P&L
        logger.info("Pairing trades...")
        df = self.data_processor.pair_trades(df, output_dir=output_dir, trader_name=trader_name)


        # # Step 1.5: Add EMA scores (NEW FEATURE)
        # ema_stats = None
        # if include_ema and self.ema_enabled:
        #     try:
        #         logger.info("Calculating EMA allocation scores...")
        #         df = self.ema_calculator.add_ema_scores_to_trades(df)
        #         ema_stats = self.ema_calculator.get_ema_summary_stats(df)
        #         logger.info("EMA scores calculated successfully")
        #     except Exception as e:
        #         logger.error(f"Error calculating EMA scores: {str(e)}")
        #         logger.warning("Continuing analysis without EMA scores")
        #
        # Step 2: Calculate metrics
        logger.info("Calculating metrics...")
        metrics = self.metrics_calculator.calculate_all_metrics(df)

        # # Add EMA stats to metrics if available
        # if ema_stats:
        #     metrics['ema_allocation'] = ema_stats
        #
        # Step 3: Detect patterns
        logger.info("Detecting patterns...")
        patterns = self.pattern_detector.detect_all_patterns(df)

        # Step 4: LLM Analysis
        logger.info("Generating AI analysis...")
        analysis = self.llm_analyzer.generate_analysis(metrics, patterns, df)

        # add nifty data chart in analysis['web_data']['nifty_pnl_timeline']->[date]['values']
        # --- Normalize PNL & NIFTY values to 100 scale

        # PNL cumulative
        pnl_vals = analysis['web_data']['charts']['pnl_timeline']['values']
        if pnl_vals:
            start = pnl_vals[0]
            norm_pnl_vals = [(v - start) + 100 for v in pnl_vals]
        else:
            norm_pnl_vals = []

        # Normalize NIFTY
        nifty_vals = list(nifty_chart_data.values())
        if nifty_vals:
            start_idx = nifty_vals[0]
            norm_nifty_vals = [(v / start_idx) * 100 for v in nifty_vals]
        else:
            norm_nifty_vals = []


        analysis_web_data_charts = {}
        analysis_web_data_charts['dates'] = list(nifty_chart_data.keys())
        analysis_web_data_charts['values'] = list(nifty_chart_data.values())
        analysis['web_data']['charts']['nifty_pnl_timeline'] = analysis_web_data_charts
        analysis['web_data']['charts']['relative_chart'] = {
            "dates": list(nifty_chart_data.keys()),
            "pnl_normalized": norm_pnl_vals,
            "nifty_normalized": norm_nifty_vals,
        }




        # Step 5: Generate report
        logger.info("Generating report...")

        report = self.report_generator.generate_report(
            metrics, patterns, analysis, trader_name
        )

        # Step 6: Export report
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export as JSON
        json_path = output_path / f"{trader_name}_report.json"
        self.report_generator.export_json(report, str(json_path))
        logger.info(f"JSON report saved to {json_path}")

        # Export as HTML
        html_path = output_path / f"{trader_name}_report.html"
        self.report_generator.export_html(report, str(html_path))
        logger.info(f"HTML report saved to {html_path}")

        # # Export enriched CSV with EMA scores
        # if include_ema and self.ema_enabled:
        #     csv_path = output_path / f"{trader_name}_trades_with_ema.csv"
        #     df.to_csv(csv_path, index=False)
        #     logger.info(f"Enriched trades CSV saved to {csv_path}")

        logger.info("Analysis complete!")

        return report

def main():
    parser = argparse.ArgumentParser(description='Trading Persona Analyzer')
    parser.add_argument('data_file', help='Path to trading data CSV file')
    parser.add_argument('--trader-name', default='Trader', help='Trader name for report')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--output-dir', default='data/reports', help='Output directory')
    parser.add_argument('--no-ema', action='store_true', help='Skip EMA score calculation')

    args = parser.parse_args()

    analyzer = TradingPersonaAnalyzer(args.config)
    report = analyzer.analyze(
        args.data_file,
        args.trader_name,
        args.output_dir,
        include_ema=not args.no_ema
    )

    if report:
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print(f"\nTrader: {args.trader_name}")
        print(f"Total Trades: {report['executive_summary']['total_trades']}")
        print(f"Net P&L: ₹{report['executive_summary']['net_pnl']:,.2f}")
        print(f"Win Rate: {report['executive_summary']['win_rate']:.1f}%")
        print(f"Risk Level: {report['executive_summary']['risk_level']}")
        print(f"\nRisk Score: {report['risk_score']}/100")

        # Display EMA stats if available
        if 'ema_allocation' in report.get('metrics', {}):
            print("\n" + "="*50)
            print("EMA ALLOCATION SCORES")
            print("="*50)
            ema_stats = report['metrics']['ema_allocation']
            print(f"Stock EMA (Avg): {ema_stats['stock_ema']['mean']:.2f}")
            print(f"Nifty EMA (Avg): {ema_stats['nifty_ema']['mean']:.2f}")
            print(f"Midcap EMA (Avg): {ema_stats['midcap_ema']['mean']:.2f}")

        print("\nReports generated successfully!")
    else:
        print("Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Trading Persona Analyzer - Main Application
Analyze trading patterns and generate comprehensive insights using local LLMs
"""
import pandas as pd
import yaml
import logging
import argparse
from pathlib import Path
import sys
import webbrowser

from src.data_processor import TradingDataProcessor
from src.metrics_calculator import TradingMetricsCalculator
from src.pattern_detector import TradingPatternDetector
from src.llm_analyzer import LLMAnalyzer
from src.report_generator import ReportGenerator
from src.ema_calculator import EMACalculator
from datetime import datetime, date
import numpy as np
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

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
        self.llm_analyzer = LLMAnalyzer(self.config)
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
                output_dir: str = "data/reports", include_ema: bool = True, pnl_csv: str = False):
        """Run complete analysis pipeline"""

        logger.info(f"Starting analysis for {trader_name}")

        # Step 1: Load and process data
        logger.info("Loading data...")
        df = self.data_processor.load_data(data_filepath)
        nifty_chart_data = {}
        if pnl_csv and str(pnl_csv).lower() != 'false':
            pnl_csv_df = pd.read_csv(pnl_csv)
            date_col = next((c for c in pnl_csv_df.columns if c.lower() == 'date'), 'Date')
            pnl_csv_df['Date'] = pd.to_datetime(pnl_csv_df[date_col], errors='coerce')
            nifty_chart_data = self.data_processor.get_nifty_data(pnl_csv_df)

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
        # Step 2: Detect patterns
        logger.info("Detecting patterns...")
        patterns = self.pattern_detector.detect_all_patterns(df)

        # Step 3: Calculate metrics (now with patterns for trace sheet)
        logger.info("Calculating metrics...")
        metrics = self.metrics_calculator.calculate_all_metrics(df, pnl_csv, nifty_data=nifty_chart_data, patterns=patterns)

        # Step 4: LLM Analysis
        logger.info("Generating AI analysis...")
        analysis = self.llm_analyzer.generate_analysis(metrics, patterns, df, trader_name=trader_name)

        # add nifty data chart in analysis['web_data']['nifty_pnl_timeline']->[date]['values']
        # --- Normalize PNL & NIFTY values to 100 scale
        # PNL cumulative
        pnl_vals = analysis['web_data']['charts']['pnl_timeline']['values']
        # print(pnl_vals)
        if pnl_vals:
            start = pnl_vals[0]
            norm_pnl_vals = [(v - start) + 100 for v in pnl_vals]
        else:
            norm_pnl_vals = []

        # Normalize NIFTY
        nifty_vals = list(nifty_chart_data.values())
        if nifty_vals and nifty_vals[0] != 0:
            start_idx = nifty_vals[0]
            norm_nifty_vals = [(v / start_idx) * 100 for v in nifty_vals]
        else:
            norm_nifty_vals = [100] * len(nifty_vals) if nifty_vals else []



        analysis_web_data_charts = {}
        analysis_web_data_charts['dates'] = list(nifty_chart_data.keys())
        analysis_web_data_charts['values'] = list(nifty_chart_data.values())
        analysis['web_data']['charts']['nifty_pnl_timeline'] = analysis_web_data_charts
        analysis['web_data']['charts']['relative_chart'] = {
            "dates": list(nifty_chart_data.keys()),
            "pnl_normalized": norm_pnl_vals,
            "nifty_normalized": norm_nifty_vals,
        }




        # Step 4.5: Load Proposed Trade File (if exists)
        proposed_trades = []
        proposed_csv_path = Path("TRADE_FILES/Proposed_Trade_File.csv")
        if proposed_csv_path.exists():
            try:
                # Load with pandas to handle comma/separator issues
                pt_df = pd.read_csv(proposed_csv_path)
                
                # 🔥 FIX: Merge PnL from the paired trades dataframe if missing
                if 'pnl' not in pt_df.columns and not df.empty:
                    # Create a lookup map for faster merging: (symbol, date_str) -> pnl
                    try:
                        # Normalize dates to string format found in Proposed file for matching
                        df_lookup = df.copy()
                        df_lookup['dt_match'] = pd.to_datetime(df_lookup['trade_date']).dt.strftime('%d-%m-%Y %H:%M')
                        pnl_map = df_lookup.groupby(['symbol', 'dt_match'])['pnl'].sum().to_dict()
                        
                        def lookup_pnl(row):
                            # Try to find exact match on symbol and date
                            key = (row.get('symbol'), str(row.get('trade_date')).strip())
                            return pnl_map.get(key, 0.0)
                        
                        pt_df['pnl'] = pt_df.apply(lookup_pnl, axis=1)
                        logger.info("Successfully merged PnL into Proposed Trades from paired trade data.")
                    except Exception as e:
                        logger.warning(f"Failed to merge PnL into Proposed Trades: {e}. Defaulting to 0.")
                        pt_df['pnl'] = 0.0
                
                # Final safety: Ensure every row has 'pnl' even if merge failed
                if 'pnl' not in pt_df.columns:
                    pt_df['pnl'] = 0.0

                # Convert to records for Jinja2
                proposed_trades = pt_df.to_dict('records')
                logger.info(f"Loaded {len(proposed_trades)} proposed trades for report.")
            except Exception as e:
                logger.error(f"Error loading Proposed_Trade_File.csv: {e}")

        # Step 5: Generate report
        logger.info("Generating report...")

        report = self.report_generator.generate_report(
            metrics, patterns, analysis, trader_name
        )
        # Add proposed trades to report data
        report["appendix"]["proposed_trades"] = proposed_trades

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

        # Export consolidated trace CSV
        trace_data = metrics.get('consolidated_trace', [])
        if trace_data:
            trace_df = pd.DataFrame(trace_data)
            trace_csv_path = output_path / f"{trader_name}_consolidated_trace.csv"
            trace_df.to_csv(trace_csv_path, index=False)
            logger.info(f"Consolidated Trace Sheet saved to {trace_csv_path}")

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
    parser.add_argument('--pnl-csv', help='Path to P&L CSV file', required=False)
    args = parser.parse_args()

    analyzer = TradingPersonaAnalyzer(args.config)
    pnl_csv = args.pnl_csv

    # Automatic discovery of P&L file if not provided, except for Gourav
    if not pnl_csv and args.trader_name != "Gourav":
        data_path = Path(args.data_file)
        # Try finding [Trader]_MTM.csv in the same directory as the data file
        possible_pnl = data_path.parent / f"{args.trader_name}_MTM.csv"
        if possible_pnl.exists():
            pnl_csv = str(possible_pnl)
            print(f"Automatically discovered P&L file: {pnl_csv}")

    report = analyzer.analyze(
        args.data_file,
        args.trader_name,
        args.output_dir,
        include_ema=not args.no_ema,
        pnl_csv=pnl_csv
    )

    if report:
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print(f"\nTrader: {args.trader_name}")
        print(f"Total Trades: {report['appendix']['metrics'].get('total_trades', 0)}")
        print(f"Net P&L: ₹{report['hero']['topline'].get('net_pnl', 0):,.2f}")
        print(f"Win Rate: {report['hero']['topline'].get('win_rate', 0):.1f}%")
        print(f"Risk Level: {report['metadata'].get('risk_severity', 'N/A')}")
        print(f"\nRisk Score: {report['metadata'].get('risk_score', 'N/A')}/100")

        # Display EMA stats if available
        metrics_dict = report.get('appendix', {}).get('metrics', {})
        if 'ema_allocation' in metrics_dict:
            print("\n" + "="*50)
            print("EMA ALLOCATION SCORES")
            print("="*50)
            ema_stats = metrics_dict['ema_allocation']
            print(f"Stock EMA (Avg): {ema_stats['stock_ema']['mean']:.2f}")
            print(f"Nifty EMA (Avg): {ema_stats['nifty_ema']['mean']:.2f}")
            print(f"Midcap EMA (Avg): {ema_stats['midcap_ema']['mean']:.2f}")

        # Display Multivariate Pattern Matrices
        mvp_data = metrics_dict.get('multivariate_pattern_analysis', [])

        print("\nReports generated successfully!")
        
        # Automatically open the report
        try:
            report_path = Path(args.output_dir).absolute() / f"{args.trader_name}_report.html"
            print(f"Opening report: {report_path}")
            webbrowser.open(f"file://{report_path}")
        except Exception as e:
            print(f"Could not open browser: {e}")
    else:
        print("Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    print(f"time started = {datetime.now()}")
    main()
    print(f"time finished = {datetime.now()}")

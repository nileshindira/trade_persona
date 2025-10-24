#!/usr/bin/env python3
"""
Batch Runner for Trading Persona Analyzer
Runs analysis for each CSV file in a given folder
"""

import os
import sys
from pathlib import Path
from main import TradingPersonaAnalyzer

def run_batch(input_folder: str, config_path: str = "config.yaml", output_root: str = "data/reports"):
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"❌ Input folder not found: {input_folder}")
        sys.exit(1)

    # Create root output directory if missing
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer once
    analyzer = TradingPersonaAnalyzer(config_path)

    csv_files = sorted(input_path.glob("*.csv"))
    if not csv_files:
        print("⚠️ No CSV files found in the folder.")
        sys.exit(0)

    for file_path in csv_files:
        trader_name = file_path.stem  # e.g., 'Dewang' from 'Dewang.csv'
        trader_output_dir = output_root_path / trader_name
        trader_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 Running analysis for: {trader_name}")
        print(f"📁 Input file: {file_path}")
        print(f"📤 Output folder: {trader_output_dir}")

        try:
            analyzer.analyze(
                data_filepath=str(file_path),
                trader_name=trader_name,
                output_dir=str(trader_output_dir),
                include_ema=True
            )
        except Exception as e:
            print(f"❌ Failed for {trader_name}: {str(e)}")
            continue

    print("\n✅ Batch processing complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_folder.py <input_folder> [config.yaml]")
        sys.exit(1)

    input_folder = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"

    run_batch(input_folder, config_path)

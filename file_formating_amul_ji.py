import pandas as pd
import os
from glob import glob

# Folder path containing all CSV files
folder_path = "/home/system-4/PycharmProjects/Trading_Persona_2/trade_analysis_dhan/TRADE_FILES"

# Get all CSV files in the folder
csv_files = glob(os.path.join(folder_path, "*.csv"))

# Load reference column structure
col = pd.read_csv("data/sample_trades.csv")
ref_columns = col.columns

for file_path in csv_files:
    print(f"Processing file: {file_path}")

    # Load CSV (no skiprows here, since your file has headers in row 1)
    data = pd.read_csv(file_path, sep="\t|,", engine="python")  # handle tabs or commas

    # Ensure 'ORDER DATETIME' exists
    if 'ORDER DATETIME' not in data.columns:
        print(f"⚠️ Skipping {file_path}: missing 'ORDER DATETIME' column.")
        continue

    # ✅ Convert to datetime safely (auto-detect format)
    data['Trade Date'] = pd.to_datetime(data['ORDER DATETIME'], errors='coerce')

    # Add missing column for downstream compatibility
    data['charges'] = 0

    # Select and rename relevant columns (make sure these exist in your CSV)
    required_cols = ['Trade Date', 'SCRIP SYMBOL', 'BUY SALE', 'Qty', 'Rate', 'Net Amount', 'charges']
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        print(f"⚠️ Skipping {file_path}: missing required columns {missing_cols}")
        continue

    # Build standardized dataframe
    data1 = data[required_cols]
    data1.columns = ref_columns  # use your reference structure (e.g. trade_date, symbol, etc.)

    # Save processed file
    output_file = os.path.join(folder_path, os.path.basename(file_path))
    data1.to_csv(output_file, index=False)

    print(f"✅ Saved processed file: {output_file}")

print("🎉 All CSV files processed successfully.")

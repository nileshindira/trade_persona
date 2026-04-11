import pandas as pd
import sys

def main():
    try:
        csv_path = '/home/system-4/PycharmProjects/trade_persona/TRADE_FILES/Gourav.csv'
        out_path = '/home/system-4/PycharmProjects/trade_persona/TRADE_FILES/Gourav_MTM.csv'
        print(f"Reading {csv_path}...")
        df = pd.read_csv(csv_path)
        print("Processing...")
        df['date'] = pd.to_datetime(df['trade_date']).dt.date
        
        def calc_val(row):
            if row['transaction_type'] == 'SELL':
                return row['trade_value']
            elif row['transaction_type'] == 'BUY':
                return -row['trade_value']
            else:
                return 0

        df['val'] = df.apply(calc_val, axis=1)
        pnl = df.groupby('date')['val'].sum().reset_index()
        pnl.columns = ['date', 'pnl']
        
        print(f"Writing outcome to {out_path}...")
        pnl.to_csv(out_path, index=False)
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

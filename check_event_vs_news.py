
import pandas as pd
import sys
import os

# Mimic the path setup
sys.path.append(os.getcwd())

def check_data(filepath):
    print(f"--- Checking Data: {filepath} ---")
    try:
        df = pd.read_csv(filepath)
        print(f"Total Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Check Event Column
        if 'is_event' in df.columns:
            event_count = df['is_event'].sum()
            print(f"\n[is_event] True Count: {event_count}")
            if event_count > 0:
                print("Sample Event Rows:")
                print(df[df['is_event'] == True][['symbol', 'trade_date', 'is_event']].head(3))
        else:
            print("\n[is_event] Column NOT FOUND")

        # Check News Column
        if 'is_news' in df.columns:
            news_count = df['is_news'].sum()
            print(f"\n[is_news] True Count: {news_count}")
            if news_count > 0:
                 print("Sample News Rows:")
                 print(df[df['is_news'] == True][['symbol', 'trade_date', 'is_news']].head(3))
        else:
             print("\n[is_news] Column NOT FOUND")
             
        # Cross Check
        if 'is_event' in df.columns and 'is_news' in df.columns:
            overlap = df[(df['is_event'] == True) & (df['is_news'] == True)]
            print(f"\nOverlap (Event AND News): {len(overlap)}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    file_path = "TRADE_FILES/ANISH_NEW.csv"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    check_data(file_path)

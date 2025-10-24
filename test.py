import pandas as pd

data= (pd.read_csv('Dewang.csv').head())
print(data.head())
print(data.columns)

import pandas as pd

# === Load CSV ===
df = pd.read_csv("Dewang.csv")

# === Ensure columns exist ===
# Replace these with your exact column names
# e.g. trade_type could be 'BUY'/'SELL', qty could be 'quantity' or 'qty'
trade_col = 'transaction_type'   # e.g., 'BUY' or 'SELL'
qty_col = 'quantity'       # adjust to your CSV
symbol_col = 'symbol'

# === Calculate buy/sell quantities per symbol ===
summary = df.groupby([symbol_col, trade_col])[qty_col].sum().unstack(fill_value=0)

# === Check equality ===
summary['equal_qty'] = summary.get('BUY', 0) == summary.get('SELL', 0)

# === Show results ===
print(summary)

# === Optionally: filter mismatched symbols ===
unequal = summary[summary['equal_qty'] == False]
if unequal.empty:
    print("\n✅ All symbols have equal BUY and SELL quantities.")
else:
    print("\n⚠️ Symbols with unequal BUY/SELL quantities:")
    print(unequal)

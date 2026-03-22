
import sys
import os
import pandas as pd
from typing import Dict, Any

# Setup paths
sys.path.append(os.getcwd())

from src.data_processor import TradingDataProcessor
from src.metrics_calculator import TradingMetricsCalculator
from src.llm_analyzer import LLMAnalyzer
from src.pattern_detector import TradingPatternDetector

# Mock config
MOCK_CONFIG = {
    "database": {
        "dbname": "trading_db",
        "user": "user",
        "password": "password",
        "host": "localhost",
        "table_name": "stock_data"
    },
    "data": {
        "required_columns": ["symbol", "trade_date", "transaction_type", "quantity", "price"]
    },
    "metrics": {
        "risk_free_rate": 0.05,
        "trading_days_per_year": 252
    },
    "analysis": {
         "min_trades_for_pattern": 5
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama3.2"
    }
}

class DummyLogger:
    def info(self, msg): pass
    def error(self, msg): print(f"DEBUG LOG ERROR: {msg}")
    def warning(self, msg): pass

class ContextAuditAnalyzer(LLMAnalyzer):
    """Subclass to intercept prompts without calling Ollama"""
    
    def __init__(self, output_file="LLM_CONTEXT_AUDIT.md"):
        self.audit_log = []
        self.config = MOCK_CONFIG
        self.logger = DummyLogger()
        self.model = "llama3.2"

    def _call_ollama(self, prompt: str, system_prompt: str) -> str:
        """Intercept and log the prompt instead of calling API"""
        full_context = f"SYSTEM: {system_prompt}\n\nUSER: {prompt}"
        char_len = len(full_context)
        token_est = char_len / 4.0
        
        step_info = {
            "step": self.current_step,
            "char_length": char_len,
            "token_estimate": int(token_est),
            "system_prompt": system_prompt,
            "user_prompt": prompt
        }
        self.audit_log.append(step_info)
        
        # Return dummy JSON or text to allow pipeline to proceed
        if "JSON" in prompt:
             return '{"event": {"verdict": "Mock", "reasoning": "Mock"}, "news": {"verdict": "Mock", "reasoning": "Mock"}, "volume": {"verdict": "Mock", "reasoning": "Mock"}, "trend": {"verdict": "Mock", "reasoning": "Mock"}}'
        return "Mock LLM Response"

    def run_audit(self, metrics, patterns, df, pnl_df=None):
        self.pnl_df = pnl_df if pnl_df is not None else pd.DataFrame()
        
        # We manually trigger the internal methods to capture their prompts
        steps = [
            ("_analyze_trader_profile", self._analyze_trader_profile),
            ("_analyze_risk", self._analyze_risk),
            ("_analyze_behavior", self._analyze_behavior),
            ("_analyze_context_performance", self._analyze_context_performance),
            ("_generate_recommendations", self._generate_recommendations),
            ("_summarize_performance", self._summarize_performance)
        ]
        
        context = self._prepare_context(metrics, patterns, df)
        
        print(f"Base Context Length: {len(context)} chars")
        
        for name, method in steps:
            self.current_step = name
            try:
                method(context)
                print(f"✅ Captured {name}")
            except Exception as e:
                print(f"❌ Failed {name}: {e}")

    def save_report(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# LLM Context & Token Audit Report\n\n")
            f.write(f"**Generated from Real Data Run**\n")
            f.write(f"**Methodology:** Token count estimated as `chars / 4`.\n\n")
            
            total_tokens = sum(x['token_estimate'] for x in self.audit_log)
            f.write(f"### Total Estimated Tokens: **{total_tokens}**\n\n")
            f.write("---\n\n")
            
            for item in self.audit_log:
                f.write(f"## Step: `{item['step']}`\n")
                f.write(f"- **Character Count:** {item['char_length']}\n")
                f.write(f"- **Estimated Tokens:** {item['token_estimate']}\n\n")
                
                f.write("### 🔹 System Prompt Provided\n")
                f.write("```text\n")
                f.write(item['system_prompt'])
                f.write("\n```\n\n")
                
                f.write("### 🔸 User Prompt (Context + Instructions)\n")
                f.write("```text\n")
                f.write(item['user_prompt'])
                f.write("\n```\n\n")
                f.write("---\n")

def main():
    try:
        data_file = "TRADE_FILES/ANISH_NEW.csv"
        pnl_file = "TRADE_FILES/ANISH_MTM.csv"
        
        print("1. Loading Data...")
        processor = TradingDataProcessor(MOCK_CONFIG)
        
        try:
            df = processor.load_data(data_file)
        except Exception as e:
            print(f"DB Load failed ({e}), falling back to partial CSV load...")
            df = pd.read_csv(data_file)
            status_cols = ['is_event', 'is_news', 'is_high_volume', 'nifty50_pct_chg_1w', 'news_category', 'pnl']
            for c in status_cols:
                if c not in df.columns: 
                    if c == 'news_category': df[c] = ""
                    elif c == 'pnl': df[c] = 0.0
                    else: df[c] = False
            
            # Ensure types
            df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0.0)
            df = df.dropna(subset=['trade_date'])

            df['trade_hour'] = df['trade_date'].dt.hour
            df['trade_day_of_week'] = df['trade_date'].dt.dayofweek
            df['trade_month'] = df['trade_date'].dt.month
            # Mock DB columns logic for Options
            if 'strike_price' not in df.columns: df['strike_price'] = 0.0
            if 'option_type' not in df.columns: df['option_type'] = 'EQ'
            if 'close' not in df.columns: df['close'] = df['price']
            
            # Mock OHLC from DB
            if 'high' not in df.columns: df['high'] = df['price']
            if 'low' not in df.columns: df['low'] = df['price']
            if 'open' not in df.columns: df['open'] = df['price']
            if 'volume' not in df.columns: df['volume'] = 1000
            
            # Additional DB flags
            db_defaults = {
                'is_52week_high': False, 'is_52week_low': False, 
                'is_alltime_high': False, 'is_alltime_low': False,
                'atr': 0.0, 'market_behaviour': 'Neutral'
            }
            for k, v in db_defaults.items():
                if k not in df.columns: df[k] = v
            
            if 'chart_charts' not in df.columns:
                df['chart_charts'] = [[] for _ in range(len(df))]

            df['strike_price'] = pd.to_numeric(df['strike_price'], errors='coerce').fillna(0.0)
            df['close'] = pd.to_numeric(df['close'], errors='coerce').fillna(0.0)
        
        print("1.5 Pairing Trades (FIFO)...")
        # Critical for PnL and Holding Period columns
        df = processor.pair_trades(df)

        print("2. Calculating Metrics...")
        calculator = TradingMetricsCalculator(MOCK_CONFIG)
        metrics = calculator.calculate_all_metrics(df, pnl_csv=pnl_file)
        
        print("3. Detecting Patterns...")
        detector = TradingPatternDetector(MOCK_CONFIG)
        patterns = detector.detect_all_patterns(df)
        
        print("4. Running LLM Audit...")
        auditor = ContextAuditAnalyzer()
        auditor.run_audit(metrics, patterns, df, calculator.pnl_df)
        
        outfile = "LLM_CONTEXT_AUDIT.md"
        auditor.save_report(outfile)
        print(f"✅ Audit Report Saved: {outfile}")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

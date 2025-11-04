"""
Pattern Detector Module
Detects trading patterns and behavioral issues
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from collections import Counter

class TradingPatternDetector:
    """Detect trading patterns and behaviors"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_trades = config['analysis']['min_trades_for_pattern']
    
    def detect_all_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect all trading patterns"""
        patterns = {}
        
        patterns['overtrading'] = self.detect_overtrading(df)
        patterns['revenge_trading'] = self.detect_revenge_trading(df)
        patterns['pyramiding'] = self.detect_pyramiding(df)
        patterns['scalping'] = self.detect_scalping(df)
        patterns['hedging'] = self.detect_hedging(df)
        patterns['time_patterns'] = self.detect_time_patterns(df)
        patterns['instrument_clustering'] = self.detect_instrument_clustering(df)

        # 🧠 NEW advanced detections
        patterns['fomo_trading'] = self.detect_fomo_trading(df)
        patterns['chasing_losses'] = self.detect_chasing_losses(df)
        patterns['overconfidence'] = self.detect_overconfidence(df)
        patterns['weekend_exposure'] = self.detect_weekend_exposure(df)

        return patterns
    # ---------- 🧠 NEW DETECTIONS BELOW ----------

    def detect_fomo_trading(self, df: pd.DataFrame) -> Dict:
        """
        Detect FOMO (Fear Of Missing Out) trades:
        Entering immediately after large price moves or missed opportunities.
        """
        if not {'open', 'close', 'trade_date'}.issubset(df.columns):
            return {'detected': False}

        df = df.sort_values('trade_date')
        df['price_change'] = ((df['close'] - df['open']) / df['open']) * 100

        large_moves = df['price_change'].abs() > 2.5  # big intraday moves
        fomo_trades = 0

        for i in range(1, len(df)):
            if large_moves.iloc[i - 1] and df['transaction_type'].iloc[i] == 'BUY':
                fomo_trades += 1

        return {
            'detected': fomo_trades > self.min_trades * 0.2,
            'fomo_trades': int(fomo_trades),
            'percentage': float(fomo_trades / len(df) * 100) if len(df) > 0 else 0
        }

    def detect_chasing_losses(self, df: pd.DataFrame) -> Dict:
        """
        Detect if trader increases position size after losses.
        (A sign of emotional trading or drawdown chasing)
        """
        df = df.sort_values('trade_date')
        chasing_events = 0

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            if prev['pnl'] < 0 and curr['quantity'] > prev['quantity']:
                chasing_events += 1

        return {
            'detected': chasing_events > self.min_trades * 0.3,
            'count': int(chasing_events),
            'percentage': float(chasing_events / len(df) * 100) if len(df) > 0 else 0
        }

    def detect_overconfidence(self, df: pd.DataFrame) -> Dict:
        """
        Detect overconfidence after winning streaks.
        (Increased position size or risk-taking after consecutive wins)
        """
        df = df.sort_values('trade_date')
        win_streak = 0
        overconf_trades = 0

        for i in range(1, len(df)):
            if df.iloc[i - 1]['pnl'] > 0:
                win_streak += 1
                if win_streak >= 3 and df.iloc[i]['quantity'] > df.iloc[i - 1]['quantity']:
                    overconf_trades += 1
            else:
                win_streak = 0

        return {
            'detected': overconf_trades > self.min_trades * 0.2,
            'overconf_trades': int(overconf_trades),
            'percentage': float(overconf_trades / len(df) * 100) if len(df) > 0 else 0
        }

    def detect_weekend_exposure(self, df: pd.DataFrame) -> Dict:
        """
        Detect trades or positions held over the weekend (Friday → Monday).
        High-risk exposure pattern.
        """
        df = df.sort_values('trade_date')
        weekend_holds = 0

        df['trade_day'] = df['trade_date'].dt.day_name()
        friday_trades = df[df['trade_day'] == 'Friday']

        for _, friday_trade in friday_trades.iterrows():
            monday_trades = df[
                (df['symbol'] == friday_trade['symbol']) &
                (df['trade_date'] > friday_trade['trade_date']) &
                (df['trade_date'] - friday_trade['trade_date']).dt.days <= 3
            ]
            if not monday_trades.empty:
                weekend_holds += 1

        return {
            'detected': weekend_holds > self.min_trades * 0.1,
            'weekend_positions': int(weekend_holds),
            'percentage': float(weekend_holds / len(df) * 100) if len(df) > 0 else 0
        }
    def detect_overtrading(self, df: pd.DataFrame) -> Dict:
        """Detect overtrading behavior"""
        daily_trades = df.groupby(df['trade_date'].dt.date).size()
        
        # Thresholds
        excessive_threshold = 10  # More than 10 trades/day
        
        overtrading_days = (daily_trades > excessive_threshold).sum()
        total_days = len(daily_trades)
        
        return {
            'detected': overtrading_days > total_days * 0.3,
            'overtrading_days': int(overtrading_days),
            'avg_trades_per_day': float(daily_trades.mean()),
            'max_trades_per_day': int(daily_trades.max()),
            'severity': 'HIGH' if overtrading_days > total_days * 0.5 else 'MEDIUM' if overtrading_days > total_days * 0.3 else 'LOW'
        }
    
    def detect_revenge_trading(self, df: pd.DataFrame) -> Dict:
        """Detect revenge trading (trading after losses)"""
        df_sorted = df.sort_values('trade_date')
        
        revenge_trades = 0
        
        for i in range(1, len(df_sorted)):
            prev_trade = df_sorted.iloc[i-1]
            curr_trade = df_sorted.iloc[i]
            
            # Check if current trade is within 30 minutes of previous loss
            time_diff = (curr_trade['trade_date'] - prev_trade['trade_date']).total_seconds() / 60
            
            if prev_trade['pnl'] < 0 and time_diff < 30:
                # Check if trade size increased
                if curr_trade['quantity'] > prev_trade['quantity']:
                    revenge_trades += 1
        
        return {
            'detected': revenge_trades > self.min_trades,
            'count': int(revenge_trades),
            'percentage': float(revenge_trades / len(df) * 100) if len(df) > 0 else 0
        }
    
    def detect_pyramiding(self, df: pd.DataFrame) -> Dict:
        """Detect pyramiding (adding to positions)"""
        pyramiding_sequences = 0
        
        for symbol in df['symbol'].unique():
            symbol_trades = df[df['symbol'] == symbol].sort_values('trade_date')
            
            consecutive_buys = 0
            for _, trade in symbol_trades.iterrows():
                if trade['transaction_type'] == 'BUY':
                    consecutive_buys += 1
                else:
                    if consecutive_buys > 1:
                        pyramiding_sequences += 1
                    consecutive_buys = 0
        
        return {
            'detected': pyramiding_sequences > 5,
            'sequences': int(pyramiding_sequences)
        }
    
    def detect_scalping(self, df: pd.DataFrame) -> Dict:
        """Detect scalping behavior"""
        # Scalping = very short holding periods
        avg_holding = df['holding_period_minutes'].mean()
        
        scalping_trades = len(df[df['holding_period_minutes'] < 30])
        
        return {
            'detected': avg_holding < 60,  # Average holding < 1 hour
            'avg_holding_minutes': float(avg_holding),
            'scalping_trades': int(scalping_trades),
            'scalping_percentage': float(scalping_trades / len(df) * 100) if len(df) > 0 else 0
        }
    
    def detect_hedging(self, df: pd.DataFrame) -> Dict:
        """Detect hedging behavior (simultaneous calls and puts)"""
        hedged_positions = 0
        
        # Group by date and symbol base
        for date in df['trade_date'].dt.date.unique():
            day_trades = df[df['trade_date'].dt.date == date]
            
            # Extract base symbol (remove CALL/PUT)
            day_trades = day_trades.copy()  # ensure independent DataFrame
            day_trades.loc[:, 'base_symbol'] = day_trades['symbol'].str.extract(r'(\w+)')[0]
            
            for base_sym in day_trades['base_symbol'].unique():
                sym_trades = day_trades[day_trades['base_symbol'] == base_sym]
                
                has_call = any('CALL' in s for s in sym_trades['symbol'])
                has_put = any('PUT' in s for s in sym_trades['symbol'])
                
                if has_call and has_put:
                    hedged_positions += 1
        
        return {
            'detected': hedged_positions > 5,
            'hedged_days': int(hedged_positions)
        }
    
    def detect_time_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect time-based trading patterns"""
        # Most active hours
        hourly_dist = df['trade_hour'].value_counts()
        
        # Morning vs afternoon
        morning_trades = len(df[df['trade_hour'] < 12])
        afternoon_trades = len(df[df['trade_hour'] >= 12])
        
        return {
            'most_active_hours': hourly_dist.head(3).index.tolist(),
            'morning_trader': morning_trades > afternoon_trades,
            'morning_trades': int(morning_trades),
            'afternoon_trades': int(afternoon_trades)
        }
    
    def detect_instrument_clustering(self, df: pd.DataFrame) -> Dict:
        """Detect instrument preference clustering"""
        # Analyze instrument types
        nifty_trades = len(df[df['symbol'].str.contains('NIFTY', na=False)])
        banknifty_trades = len(df[df['symbol'].str.contains('BANKNIFTY', na=False)])
        
        call_trades = len(df[df['symbol'].str.contains('CALL', na=False)])
        put_trades = len(df[df['symbol'].str.contains('PUT', na=False)])
        
        return {
            'nifty_percentage': float(nifty_trades / len(df) * 100) if len(df) > 0 else 0,
            'banknifty_percentage': float(banknifty_trades / len(df) * 100) if len(df) > 0 else 0,
            'call_percentage': float(call_trades / len(df) * 100) if len(df) > 0 else 0,
            'put_percentage': float(put_trades / len(df) * 100) if len(df) > 0 else 0
        }

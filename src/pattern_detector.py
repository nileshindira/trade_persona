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
        # patterns['instrument_clustering'] = self.detect_instrument_clustering(df)

        # 🧠 Behavioral bias detections
        patterns['fomo_trading'] = self.detect_fomo_trading(df)
        patterns['chasing_losses'] = self.detect_chasing_losses(df)
        # patterns['overconfidence'] = self.detect_overconfidence(df)
        patterns['weekend_exposure'] = self.detect_weekend_exposure(df)
        
        # 🆕 New pattern detections
        patterns['anchor_bias'] = self.detect_anchor_bias(df)
        patterns['loss_averaging'] = self.detect_loss_averaging(df)
        patterns['intraday_vs_overnight'] = self.detect_intraday_vs_overnight_risk(df)
        patterns['absence_of_stop_loss'] = self.detect_absence_of_stop_loss(df)

        return patterns

    def detect_absence_of_stop_loss(self, df: pd.DataFrame) -> Dict:
        """Detect trades where no stop loss was apparently used (deep drawdowns held)"""
        if 'max_drawdown_on_trade' not in df.columns:
            return {'detected': False}
        
        # Consider absence of stop loss if max drawdown > 2.5% or loss > 3x average loss
        avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean()) if not df[df['pnl'] < 0].empty else 1000
        
        def is_asl(row):
            # Parse drawdown string if it's like "-2.5%"
            try:
                dd_val = float(str(row['max_drawdown_on_trade']).replace('%', ''))
            except:
                dd_val = 0
            
            if dd_val < -2.5: return True
            if row['pnl'] < -3 * avg_loss: return True
            return False

        asl_trades = df[df.apply(is_asl, axis=1)].index.tolist()
        
        return {
            'detected': len(asl_trades) > 0,
            'count': len(asl_trades),
            'trades': asl_trades,
            'description': 'Trades held through deep drawdowns without predefined exit triggers'
        }

    # ---------- 🧠 BEHAVIORAL PATTERN DETECTIONS ----------

    def detect_fomo_trading(self, df: pd.DataFrame) -> Dict:
        """
        FOMO = Entering after an EXTENDED CONTINUOUS MOVE with speed and no consolidation.
        """
        if not {'open', 'close', 'high', 'low', 'trade_date'}.issubset(df.columns):
            return {'detected': False}

        df = df.copy().sort_values('trade_date')
        fomo_trades = []
        fomo_examples = []
        
        # Group by symbol to analyze per-stock runs
        for symbol, sym_df in df.groupby('symbol'):
            sym_df = sym_df.sort_values('trade_date')
            if len(sym_df) < 2:
                continue
            
            # Calculate daily returns and runs
            sym_df = sym_df.copy()
            sym_df['daily_return'] = (sym_df['close'] - sym_df['open']) / (sym_df['open'] + 1e-6)
            sym_df['is_green'] = sym_df['daily_return'] > 0
            
            # Detect extended runs (>= 3 consecutive green/red candles)
            run_count = 0
            run_direction = None
            cum_move = 0.0
            
            for idx, (orig_idx, row) in enumerate(sym_df.iterrows()):
                current_dir = 'up' if row['daily_return'] > 0 else 'down'
                
                if current_dir == run_direction:
                    run_count += 1
                    cum_move += abs(row['daily_return'])
                else:
                    run_direction = current_dir
                    run_count = 1
                    cum_move = abs(row['daily_return'])
                
                # Check for fast extended move
                is_extended_run = run_count >= 3
                is_fast_move = cum_move > 0.05  # >5% cumulative move
                
                is_buy = row['transaction_type'] == 'BUY'
                is_sell = row['transaction_type'] in ('SALE', 'SELL')
                
                if is_extended_run or is_fast_move:
                    if (is_buy and run_direction == 'up') or (is_sell and run_direction == 'down'):
                        recent = sym_df.iloc[max(0, idx-3):idx+1]
                        if len(recent) >= 2:
                            ranges = (recent['high'] - recent['low']) / (recent['open'] + 1e-6)
                            range_expanding = ranges.is_monotonic_increasing or ranges.iloc[-1] > ranges.mean()
                            
                            if range_expanding:
                                fomo_trades.append(orig_idx) # Use original index
                                if len(fomo_examples) < 5:
                                    fomo_examples.append({
                                        'symbol': symbol,
                                        'date': str(row['trade_date']),
                                        'type': row['transaction_type'],
                                        'price': float(row['price']),
                                        'consecutive_run': int(run_count),
                                        'cumulative_move_pct': round(cum_move * 100, 2),
                                    })

        fomo_count = len(fomo_trades)
        total_buys = len(df[df['transaction_type'] == 'BUY'])
        
        return {
            'detected': fomo_count > 0,
            'fomo_count': fomo_count,
            'trades': fomo_trades, # Return indices
            'percentage': float(fomo_count / total_buys * 100) if total_buys > 0 else 0,
            'examples': fomo_examples,
            'description': 'Entries made during extended, fast-moving runs without consolidation'
        }


    def detect_anchor_bias(self, df: pd.DataFrame) -> Dict:
        """
        Anchor Bias = Trader anchors to a previous price/level and refuses to adapt.
        
        Detection:
        1. Repeated entries at similar prices after the stock has moved >5%
        2. Round number anchoring — entries/exits clustering at round numbers
        3. Averaging into a losing position at the same level
        """
        df = df.copy().sort_values('trade_date')
        anchor_events = []
        anchor_examples = []
        
        for symbol, sym_df in df.groupby('symbol'):
            buys = sym_df[sym_df['transaction_type'] == 'BUY'].sort_values('trade_date')
            if len(buys) < 2:
                continue
            
            for i in range(1, len(buys)):
                prev = buys.iloc[i-1]
                curr = buys.iloc[i]
                
                # Check if the market has moved significantly but trader buys at similar price
                price_diff_pct = abs(curr['price'] - prev['price']) / (prev['price'] + 1e-6) * 100
                
                # If price at entry is within 2% of previous entry, but high/low shows >5% move
                if price_diff_pct < 2.0:
                    # Check if market moved between these two entries
                    between = sym_df[(sym_df['trade_date'] > prev['trade_date']) & 
                                     (sym_df['trade_date'] <= curr['trade_date'])]
                    if not between.empty and 'high' in between.columns and 'low' in between.columns:
                        high_move = (between['high'].max() - prev['price']) / (prev['price'] + 1e-6) * 100
                        low_move = (prev['price'] - between['low'].min()) / (prev['price'] + 1e-6) * 100
                        
                        if high_move > 5 or low_move > 5:
                            anchor_events.append(curr.name)
                            if len(anchor_examples) < 5:
                                anchor_examples.append({
                                    'symbol': symbol,
                                    'anchored_price': float(prev['price']),
                                    'new_entry_price': float(curr['price']),
                                    'market_moved_pct': round(max(high_move, low_move), 2),
                                    'date': str(curr['trade_date']),
                                })
        
        # Round number anchoring
        round_entries = 0
        for _, row in df.iterrows():
            price = row['price']
            # Check if price is within 0.5% of a round number (100, 500, 1000, etc.)
            for base in [50, 100, 500, 1000, 5000]:
                nearest_round = round(price / base) * base
                if abs(price - nearest_round) / (price + 1e-6) < 0.005:
                    round_entries += 1
                    break
        
        round_pct = (round_entries / len(df) * 100) if len(df) > 0 else 0
        
        return {
            'detected': len(anchor_events) > 3 or round_pct > 30,
            'anchor_events': len(anchor_events),
            'round_number_pct': round(round_pct, 1),
            'examples': anchor_examples,
            'description': 'Repeated entries near the same price despite significant market moves'
        }

    def detect_loss_averaging(self, df: pd.DataFrame) -> Dict:
        """
        Loss Averaging = Buying more of a losing position at declining prices.
        """
        df = df.copy().sort_values('trade_date')
        averaging_events = []
        averaging_trades = []
        paid_off = 0
        compounded = 0
        
        for symbol, sym_df in df.groupby('symbol'):
            buys = sym_df[sym_df['transaction_type'] == 'BUY'].sort_values('trade_date')
            if len(buys) < 2:
                continue
            
            i = 0
            while i < len(buys) - 1:
                sequence = [buys.iloc[i]]
                seq_indices = [buys.index[i]]
                j = i + 1
                while j < len(buys) and buys.iloc[j]['price'] < buys.iloc[j-1]['price']:
                    sequence.append(buys.iloc[j])
                    seq_indices.append(buys.index[j])
                    j += 1
                
                if len(sequence) >= 2:
                    averaging_trades.extend(seq_indices)
                    qty_escalation = sequence[-1]['quantity'] > sequence[0]['quantity']
                    last_avg_date = sequence[-1]['trade_date']
                    sells_after = sym_df[(sym_df['transaction_type'].isin(['SALE', 'SELL'])) & 
                                        (sym_df['trade_date'] > last_avg_date)]
                    avg_buy_price = sum(s['price'] * s['quantity'] for s in sequence) / sum(s['quantity'] for s in sequence)
                    
                    if not sells_after.empty:
                        exit_price = sells_after.iloc[0]['price']
                        outcome = 'paid_off' if exit_price > avg_buy_price else 'compounded_loss'
                        if outcome == 'paid_off':
                            paid_off += 1
                        else:
                            compounded += 1
                    else:
                        outcome = 'still_open'
                    
                    total_capital = sum(s['price'] * s['quantity'] for s in sequence)
                    averaging_events.append({
                        'symbol': symbol,
                        'num_buys': len(sequence),
                        'price_decline_pct': round((1 - sequence[-1]['price'] / sequence[0]['price']) * 100, 2),
                        'qty_escalation': qty_escalation,
                        'total_capital_at_risk': round(total_capital, 2),
                        'outcome': outcome,
                        'avg_buy_price': round(avg_buy_price, 2),
                    })
                i = j
        
        return {
            'detected': len(averaging_events) > 0,
            'events_count': len(averaging_events),
            'trades': averaging_trades,
            'paid_off_count': paid_off,
            'compounded_count': compounded,
            'events': averaging_events[:10],
            'description': 'Buying more of a losing position at declining prices',
            'success_rate': round(paid_off / (paid_off + compounded) * 100, 1) if (paid_off + compounded) > 0 else 0,
        }


    def detect_intraday_vs_overnight_risk(self, df: pd.DataFrame) -> Dict:
        """
        Classify trades as INTRADAY vs OVERNIGHT and compute separate metrics for each.
        Detects overnight gap risk.
        """
        if 'holding_period' not in df.columns:
            return {'detected': False}
        
        df = df.copy()
        
        # Intraday = opened and closed same calendar day (or holding < 390 min = 6.5 hrs)
        # We use holding_period as a proxy
        INTRADAY_THRESHOLD = 390  # ~6.5 hours = one trading session
        
        traded = df[df['pnl'] != 0].copy()
        if traded.empty:
            return {'detected': False}
        
        traded['trade_type'] = traded['holding_period'].apply(
            lambda x: 'INTRADAY' if 0 < x <= INTRADAY_THRESHOLD else ('OVERNIGHT' if x > INTRADAY_THRESHOLD else 'UNKNOWN')
        )
        
        intraday = traded[traded['trade_type'] == 'INTRADAY']
        overnight = traded[traded['trade_type'] == 'OVERNIGHT']
        
        def compute_stats(subset, label):
            if subset.empty:
                return {'count': 0, 'win_rate': 0, 'avg_pnl': 0, 'max_loss': 0, 'total_pnl': 0}
            wins = len(subset[subset['pnl'] > 0])
            return {
                'count': len(subset),
                'win_rate': round(wins / len(subset) * 100, 1),
                'avg_pnl': round(float(subset['pnl'].mean()), 2),
                'max_loss': round(float(subset['pnl'].min()), 2),
                'total_pnl': round(float(subset['pnl'].sum()), 2),
                'avg_holding_minutes': round(float(subset['holding_period'].mean()), 1),
            }
        
        intraday_stats = compute_stats(intraday, 'INTRADAY')
        overnight_stats = compute_stats(overnight, 'OVERNIGHT')
        
        # Detect overnight gap risk: trades that have larger losses overnight
        overnight_risk_higher = (
            overnight_stats.get('max_loss', 0) < intraday_stats.get('max_loss', 0) * 1.5
            if intraday_stats.get('max_loss', 0) < 0 else False
        )
        
        return {
            'detected': True,
            'intraday': intraday_stats,
            'overnight': overnight_stats,
            'overnight_risk_higher': overnight_risk_higher,
            'intraday_pct': round(len(intraday) / len(traded) * 100, 1) if len(traded) > 0 else 0,
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
        df = df.copy().sort_values('trade_date')
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
        """
        Detect overtrading behavior based on quality and volume.
        """
        if df.empty: return {'detected': False}
        daily_stats = df.groupby(df['trade_date'].dt.date).agg({
            'pnl': 'sum',
            'symbol': 'count'
        }).rename(columns={'symbol': 'trade_count', 'pnl': 'daily_pnl'})
        
        global_avg_trades = daily_stats['trade_count'].mean()
        total_trades = daily_stats['trade_count'].sum()
        global_avg_pnl_per_trade = df['pnl'].sum() / total_trades if total_trades > 0 else 0
        
        daily_stats['is_overtrading'] = (
            (daily_stats['trade_count'] > 1.3 * global_avg_trades) & 
            (daily_stats['daily_pnl'] < global_avg_pnl_per_trade * daily_stats['trade_count'])
        )
        
        overtrading_days = daily_stats[daily_stats['is_overtrading']].index.tolist()
        overtrading_trades = df[df['trade_date'].dt.date.isin(overtrading_days)].index.tolist()
        
        return {
            'detected': len(overtrading_days) > 0,
            'overtrading_days': len(overtrading_days),
            'trades': overtrading_trades,
            'avg_trades_per_day': float(global_avg_trades),
            'severity': 'HIGH' if len(overtrading_days) > len(daily_stats) * 0.3 else 'MEDIUM' if len(overtrading_days) > 0 else 'LOW',
            'description': 'Trading volume significantly above your average with sub-par performance output.'
        }
    
    def detect_revenge_trading(self, df: pd.DataFrame) -> Dict:
        """Detect revenge trading (trading after losses)"""
        df_sorted = df.sort_values('trade_date')
        revenge_trades_indices = []
        
        for i in range(1, len(df_sorted)):
            prev_trade = df_sorted.iloc[i-1]
            curr_trade = df_sorted.iloc[i]
            time_diff = (curr_trade['trade_date'] - prev_trade['trade_date']).total_seconds() / 60
            
            if prev_trade['pnl'] < 0 and time_diff < 30:
                if curr_trade['quantity'] > prev_trade['quantity'] or abs(curr_trade['pnl']) > abs(prev_trade['pnl'] * 1.5):
                    revenge_trades_indices.append(curr_trade.name)
        
        return {
            'detected': len(revenge_trades_indices) > 0,
            'count': len(revenge_trades_indices),
            'trades': revenge_trades_indices,
            'percentage': float(len(revenge_trades_indices) / len(df) * 100) if len(df) > 0 else 0
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
        avg_holding = df['holding_period'].mean()
        
        scalping_trades = len(df[df['holding_period'] < 30])
        
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

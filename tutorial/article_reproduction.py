"""
KAMA + ATR Strategy - Reproduction of David Borst's Article
===============================================

This script reproduces the strategy from the article:
"Kaufman Adaptive Moving Average and ATR Long Position Strategy"
by David Borst (Aug 11, 2025)

Key Features:
1. KAMA for adaptive trend following
2. ATR for volatility-based risk control
3. Optuna for parameter optimization
4. Multiple metrics (CAGR, Sharpe, Calmar)

Workflow:
1. Fetch historical data (using yfinance instead of Financial Modeling Prep)
2. Calculate KAMA (Kaufman Adaptive Moving Average)
3. Calculate ATR (Average True Range)
4. Generate signals based on KAMA crossover and ATR filters
5. Backtest the strategy
6. Optimize parameters with Optuna
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from typing import Tuple, Dict
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Try to import Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not installed. Parameter optimization will not be available.")
    print("   Install with: pip install optuna")


# ============================================================================
# 1. DATA FETCHING
# ============================================================================

def fetch_data_yfinance(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data using yfinance (free alternative to FMP API)
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"\n📊 Fetching data for {symbol} from {start_date} to {end_date}...")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    
    # Standardize column names
    df.columns = df.columns.str.lower()
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Reset index to make date a column
    df = df.reset_index()
    
    # Handle both 'date' and 'Date' column names
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'date'}, inplace=True)
    
    # Remove timezone if present
    if pd.api.types.is_datetime64tz_dtype(df['date']):
        df['date'] = df['date'].dt.tz_localize(None)
    
    print(f"✓ Downloaded {len(df)} data points")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


# ============================================================================
# 2. KAMA CALCULATION
# ============================================================================

def kaufman_adaptive_moving_average(
    df: pd.DataFrame, 
    er_period: int, 
    fast_period: int, 
    slow_period: int
) -> pd.DataFrame:
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    
    KAMA adapts to market conditions:
    - In trending markets: moves quickly (high Efficiency Ratio)
    - In choppy markets: moves slowly (low Efficiency Ratio)
    
    Formula:
    1. Direction = |Close[today] - Close[er_period days ago]|
    2. Volatility = Sum(|Close[i] - Close[i-1]|) over er_period
    3. ER (Efficiency Ratio) = Direction / Volatility
    4. SC (Smoothing Constant) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    5. KAMA[today] = KAMA[yesterday] + SC * (Close - KAMA[yesterday])
    
    Args:
        df: DataFrame with 'close' column
        er_period: Period for Efficiency Ratio calculation
        fast_period: Fast EMA period (e.g., 2)
        slow_period: Slow EMA period (e.g., 30)
    
    Returns:
        DataFrame with 'kama' column added
    """
    close = df['close'].values
    n = len(close)
    
    # Calculate Efficiency Ratio (ER)
    direction = np.abs(close[er_period:] - close[:-er_period])
    
    # Calculate volatility (sum of absolute price changes)
    daily_changes = np.abs(np.diff(close))
    volatility = np.array([
        np.sum(daily_changes[i:i+er_period]) 
        for i in range(n - er_period)
    ])
    
    # Avoid division by zero
    volatility = np.where(volatility == 0, 0.000001, volatility)
    
    # Efficiency Ratio
    er = direction / volatility
    
    # Calculate Smoothing Constant (SC)
    sc_fast = 2 / (fast_period + 1)
    sc_slow = 2 / (slow_period + 1)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[:er_period] = close[:er_period]  # Initialize with price
    
    for i in range(er_period, n):
        kama[i] = kama[i-1] + sc[i-er_period] * (close[i] - kama[i-1])
    
    df['kama'] = kama
    
    return df


# ============================================================================
# 3. ATR CALCULATION
# ============================================================================

def average_true_range(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR)
    
    ATR measures market volatility:
    - True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    - ATR = Moving Average of True Range
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        window: Lookback period for ATR calculation
    
    Returns:
        DataFrame with 'atr' column added
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    # True Range is the maximum of the three
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # ATR is the moving average of True Range
    atr = true_range.rolling(window=window).mean()
    
    df['atr'] = atr
    
    return df


# ============================================================================
# 4. SIGNAL GENERATION
# ============================================================================

def generate_signals(
    df: pd.DataFrame, 
    atr_min_pct: float, 
    atr_max_pct: float, 
    atr_exit_pct: float
) -> pd.DataFrame:
    """
    Generate trading signals based on KAMA and ATR
    
    Entry Rules (Buy Signal):
    - Price crosses above KAMA OR
    - (Price > KAMA AND ATR is within acceptable range)
    
    Exit Rules (Sell Signal):
    - Price crosses below KAMA OR
    - ATR exceeds exit threshold (volatility spike)
    
    Args:
        df: DataFrame with 'close', 'kama', 'atr' columns
        atr_min_pct: Minimum ATR% for entry (e.g., 0.5%)
        atr_max_pct: Maximum ATR% for entry (e.g., 5%)
        atr_exit_pct: ATR% threshold for forced exit (e.g., 10%)
    
    Returns:
        DataFrame with 'position' column (1 = long, 0 = cash)
    """
    # Calculate ATR as percentage of price
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Initialize position
    df['position'] = 0
    
    for i in range(1, len(df)):
        # Current state
        price = df.loc[i, 'close']
        kama = df.loc[i, 'kama']
        atr_pct = df.loc[i, 'atr_pct']
        prev_position = df.loc[i-1, 'position']
        
        # Skip if data is invalid
        if pd.isna(price) or pd.isna(kama) or pd.isna(atr_pct):
            df.loc[i, 'position'] = prev_position
            continue
        
        # Entry condition (Buy Signal)
        if price > kama and atr_min_pct <= atr_pct <= atr_max_pct:
            df.loc[i, 'position'] = 1
        
        # Exit condition (Sell Signal)
        elif price < kama or atr_pct > atr_exit_pct:
            df.loc[i, 'position'] = 0
        
        # Otherwise, maintain previous position
        else:
            df.loc[i, 'position'] = prev_position
    
    return df


# ============================================================================
# 5. BACKTESTING
# ============================================================================

def backtest(df: pd.DataFrame, initial_balance: float = 10000) -> Tuple[float, pd.DataFrame]:
    """
    Backtest the strategy
    
    Args:
        df: DataFrame with 'position' column
        initial_balance: Starting capital
    
    Returns:
        (final_balance, trades_df)
    """
    balance = initial_balance
    position = 0  # Number of shares
    entry_price = 0
    trades = []
    
    for i in range(len(df)):
        current_pos = df.loc[i, 'position']
        price = df.loc[i, 'close']
        date = df.loc[i, 'date']
        
        # Entry: Buy signal and currently in cash
        if current_pos == 1 and position == 0:
            position = balance / price
            entry_price = price
            entry_date = date
            balance = 0
        
        # Exit: Sell signal and currently holding
        elif current_pos == 0 and position > 0:
            balance = position * price
            exit_price = price
            exit_date = date
            
            # Record trade
            pnl = (exit_price - entry_price) / entry_price * 100
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl
            })
            
            position = 0
    
    # Close any remaining position
    if position > 0:
        balance = position * df.loc[len(df)-1, 'close']
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    return balance, trades_df


def calculate_metrics(
    df: pd.DataFrame, 
    trades_df: pd.DataFrame, 
    initial_balance: float,
    final_balance: float
) -> Dict:
    """
    Calculate performance metrics
    
    Metrics:
    - Total Return %
    - CAGR (Compound Annual Growth Rate)
    - Sharpe Ratio
    - Max Drawdown
    - Calmar Ratio (CAGR / Max Drawdown)
    - Win Rate
    - Number of Trades
    """
    # Total return
    total_return = (final_balance - initial_balance) / initial_balance * 100
    
    # Number of years
    days = (df['date'].max() - df['date'].min()).days
    years = days / 365.25
    
    # CAGR
    if years > 0 and final_balance > 0:
        cagr = ((final_balance / initial_balance) ** (1 / years) - 1) * 100
    else:
        cagr = 0
    
    # Sharpe Ratio (simplified)
    if len(trades_df) > 0:
        returns = trades_df['pnl_pct'].values
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 / 15)  # Assuming avg 15 days per trade
        else:
            sharpe = 0
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) * 100
        num_trades = len(trades_df)
    else:
        sharpe = 0
        win_rate = 0
        num_trades = 0
    
    # Max Drawdown (simplified - using trade-by-trade equity)
    if len(trades_df) > 0:
        equity_curve = [initial_balance]
        for pnl in trades_df['pnl_pct'].values:
            equity_curve.append(equity_curve[-1] * (1 + pnl / 100))
        
        equity_series = pd.Series(equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0
    
    # Calmar Ratio
    if max_drawdown < 0:
        calmar = cagr / abs(max_drawdown)
    else:
        calmar = 0
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'calmar': calmar,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'days_in_market': (df['position'] == 1).sum(),
        'total_days': len(df)
    }


# ============================================================================
# 6. OPTUNA OPTIMIZATION
# ============================================================================

def optimize_strategy(
    df: pd.DataFrame,
    metric: str = 'sharpe',
    n_trials: int = 100
) -> Dict:
    """
    Optimize strategy parameters using Optuna
    
    Args:
        df: DataFrame with OHLCV data
        metric: Optimization metric ('sharpe', 'cagr', 'calmar')
        n_trials: Number of optimization trials
    
    Returns:
        Dictionary with best parameters and results
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization. Install with: pip install optuna")
    
    print(f"\n🔍 Optimizing parameters (Metric: {metric.upper()}, Trials: {n_trials})...")
    
    def objective(trial):
        # Suggest parameters
        er_period = trial.suggest_int('er_period', 2, 30)
        fast_period = trial.suggest_int('fast_period', 2, 20)
        slow_period = trial.suggest_int('slow_period', 20, 100)
        atr_window = trial.suggest_int('atr_window', 5, 30)
        atr_min_pct = trial.suggest_float('atr_min_pct', 0.5, 3.0)
        atr_max_pct = trial.suggest_float('atr_max_pct', 1.0, 10.0)
        atr_exit_pct = trial.suggest_float('atr_exit_pct', 5.0, 20.0)
        
        # Ensure logical order
        if atr_min_pct >= atr_max_pct:
            return -999999
        
        # Make a copy of the data
        df_copy = df.copy()
        
        try:
            # Calculate indicators
            df_copy = kaufman_adaptive_moving_average(df_copy, er_period, fast_period, slow_period)
            df_copy = average_true_range(df_copy, atr_window)
            
            # Generate signals
            df_copy = generate_signals(df_copy, atr_min_pct, atr_max_pct, atr_exit_pct)
            
            # Backtest
            final_balance, trades_df = backtest(df_copy)
            
            # Calculate metrics
            metrics = calculate_metrics(df_copy, trades_df, 10000, final_balance)
            
            # Return the target metric
            if metric == 'sharpe':
                return metrics['sharpe']
            elif metric == 'cagr':
                return metrics['cagr']
            elif metric == 'calmar':
                return metrics['calmar']
            else:
                return metrics['sharpe']
        
        except Exception as e:
            return -999999
    
    # Run optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n✓ Optimization complete!")
    print(f"  Best {metric.upper()}: {best_value:.4f}")
    
    return {
        'best_params': best_params,
        'best_value': best_value,
        'study': study
    }


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def run_strategy(
    symbol: str,
    start_date: str,
    end_date: str,
    optimize: bool = True,
    optimization_metric: str = 'sharpe',
    n_trials: int = 100
):
    """
    Run the complete KAMA + ATR strategy
    
    Args:
        symbol: Stock ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        optimize: Whether to optimize parameters
        optimization_metric: Metric to optimize ('sharpe', 'cagr', 'calmar')
        n_trials: Number of optimization trials
    """
    print("=" * 80)
    print("KAMA + ATR LONG POSITION STRATEGY".center(80))
    print("=" * 80)
    
    # 1. Fetch data
    df = fetch_data_yfinance(symbol, start_date, end_date)
    
    # 2. Optimize or use default parameters
    if optimize and OPTUNA_AVAILABLE:
        opt_result = optimize_strategy(df, metric=optimization_metric, n_trials=n_trials)
        params = opt_result['best_params']
    else:
        # Default parameters (from the article)
        params = {
            'er_period': 10,
            'fast_period': 2,
            'slow_period': 44,
            'atr_window': 14,
            'atr_min_pct': 1.0,
            'atr_max_pct': 5.0,
            'atr_exit_pct': 10.0
        }
        print("\n📝 Using default parameters (no optimization)")
    
    # 3. Run strategy with best/default parameters
    print("\n" + "=" * 80)
    print("RUNNING STRATEGY WITH PARAMETERS".center(80))
    print("=" * 80)
    
    df = kaufman_adaptive_moving_average(df, params['er_period'], params['fast_period'], params['slow_period'])
    df = average_true_range(df, params['atr_window'])
    df = generate_signals(df, params['atr_min_pct'], params['atr_max_pct'], params['atr_exit_pct'])
    
    # 4. Backtest
    initial_balance = 10000
    final_balance, trades_df = backtest(df, initial_balance)
    
    # 5. Calculate metrics
    metrics = calculate_metrics(df, trades_df, initial_balance, final_balance)
    
    # 6. Calculate Buy & Hold for comparison
    buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    
    # 7. Print results
    print("\n" + "=" * 80)
    print("STRATEGY PARAMETERS".center(80))
    print("=" * 80)
    print(f"  ER Period: {params['er_period']}")
    print(f"  Fast: {params['fast_period']}, Slow: {params['slow_period']}")
    print(f"  ATR Window: {params['atr_window']}")
    print(f"  ATR Min %: {params['atr_min_pct']:.3f}")
    print(f"  ATR Max %: {params['atr_max_pct']:.3f}")
    print(f"  ATR Exit %: {params['atr_exit_pct']:.3f}")
    
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS".center(80))
    print("=" * 80)
    print(f"  Total Profit: {metrics['total_return']:.2f}%")
    print(f"  Buy and Hold Profit: {buy_hold_return:.2f}%")
    print(f"  Total Number of Trades: {metrics['num_trades']}")
    
    if metrics['num_trades'] > 0:
        profitable_trades = (trades_df['pnl_pct'] > 0).sum()
        print(f"  Number of Profitable Trades: {profitable_trades}")
        print(f"  Percentage of Profitable Trades: {metrics['win_rate']:.2f}%")
    
    print(f"  Number of Days Invested: {metrics['days_in_market']}")
    print(f"  Total Number of Days in Dataset: {metrics['total_days']}")
    print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"  CAGR: {metrics['cagr']:.2f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"  Calmar Ratio: {metrics['calmar']:.2f}")
    
    # 8. Show recent trades
    if len(trades_df) > 0:
        print("\n" + "=" * 80)
        print("RECENT TRADES (Last 10)".center(80))
        print("=" * 80)
        recent_trades = trades_df.tail(10)
        for _, trade in recent_trades.iterrows():
            print(f"  {trade['entry_date'].strftime('%Y-%m-%d')} -> {trade['exit_date'].strftime('%Y-%m-%d')}: "
                  f"Entry ${trade['entry_price']:.2f}, Exit ${trade['exit_price']:.2f}, "
                  f"PnL: {trade['pnl_pct']:.2f}%")
    
    print("\n" + "=" * 80)
    print("STRATEGY EXECUTION COMPLETE".center(80))
    print("=" * 80)
    
    return df, trades_df, metrics


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='KAMA + ATR Trading Strategy (Article Reproduction)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters (no optimization)
  python article_reproduction.py --symbol TSLA --no-optimize
  
  # Optimize for Sharpe ratio
  python article_reproduction.py --symbol QQQ --metric sharpe --trials 100
  
  # Optimize for CAGR
  python article_reproduction.py --symbol AAPL --metric cagr --trials 50
  
  # Custom date range
  python article_reproduction.py --symbol NVDA --start 2020-01-01 --end 2024-12-31
        """
    )
    
    parser.add_argument('--symbol', type=str, default='TSLA',
                        help='Stock ticker symbol (default: TSLA)')
    parser.add_argument('--start', type=str, 
                        default=(datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d'),
                        help='Start date YYYY-MM-DD (default: 5 years ago)')
    parser.add_argument('--end', type=str, 
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date YYYY-MM-DD (default: today)')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip optimization and use default parameters')
    parser.add_argument('--metric', type=str, default='sharpe',
                        choices=['sharpe', 'cagr', 'calmar'],
                        help='Optimization metric (default: sharpe)')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of optimization trials (default: 100)')
    
    args = parser.parse_args()
    
    # Run the strategy
    df, trades_df, metrics = run_strategy(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        optimize=not args.no_optimize,
        optimization_metric=args.metric,
        n_trials=args.trials
    )
    
    print("\n✅ All done! The strategy has been executed successfully.")
    print("\n💡 Tips:")
    print("  - Try different symbols: AAPL, NVDA, AMD, GOOGL, MSTR")
    print("  - Experiment with different optimization metrics")
    print("  - Past performance does not guarantee future results!")
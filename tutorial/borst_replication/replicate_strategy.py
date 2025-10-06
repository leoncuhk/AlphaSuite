import pandas as pd
import numpy as np
import yfinance as yf
import optuna
import warnings

warnings.filterwarnings("ignore")

def print_section(title: str):
    """Prints a formatted section title."""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)

def fetch_data_yfinance(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical price data using yfinance."""
    print(f"Fetching data for {symbol} from {start_date} to {end_date} using yfinance...")
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"No data found for symbol {symbol}")
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = df.columns.str.lower()
    print(f"✓ Data fetched successfully: {len(df)} rows")
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]

def kaufman_adaptive_moving_average(df, er_period, fast_period, slow_period):
    """Calculates Kaufman Adaptive Moving Average (KAMA) using pandas."""
    close = df['close']
    change = abs(close.diff(er_period))
    volatility = abs(close.diff()).rolling(window=er_period).sum()
    volatility = volatility.replace(0, 1e-10)
    er = change / volatility
    sc_fast = 2 / (fast_period + 1)
    sc_slow = 2 / (slow_period + 1)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
    sc.fillna((2 / (slow_period + 1))**2, inplace=True)
    kama = np.zeros(len(close))
    kama[0] = close.iloc[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close.iloc[i] - kama[i-1])
    df['KAMA'] = kama
    return df

def average_true_range(df, window=14):
    """Calculates Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    df['ATR'] = atr
    return df

def generate_signals(df, atr_min_pct, atr_max_pct, atr_exit_pct):
    """Generates trading signals based on the OR logic from the article's text."""
    df['atr_pct'] = df['ATR'] / df['close'] * 100
    cross_above_kama = (df['close'] > df['KAMA']) & (df['close'].shift(1) <= df['KAMA'].shift(1))
    atr_in_range_and_price_above = (df['atr_pct'] >= atr_min_pct) & (df['atr_pct'] <= atr_max_pct) & (df['close'] > df['KAMA'])
    buy_signal = cross_above_kama | atr_in_range_and_price_above
    sell_signal = (df['close'] < df['KAMA']) | (df['atr_pct'] > atr_exit_pct)
    position = 0
    positions = []
    for i in range(len(df)):
        if position == 0 and buy_signal.iloc[i]:
            position = 1
        elif position == 1 and sell_signal.iloc[i]:
            position = 0
        positions.append(position)
    df['position'] = positions
    return df

def backtest(df, initial_balance=1000):
    """Performs a looped simulation backtest and returns the final balance."""
    balance = initial_balance
    shares = 0
    for i in range(1, len(df)):
        if shares == 0 and df.at[i, 'position'] == 1 and df.at[i-1, 'position'] == 0:
            shares = balance / df.at[i, 'close']
            balance = 0
        elif shares > 0 and df.at[i, 'position'] == 0 and df.at[i-1, 'position'] == 1:
            balance = shares * df.at[i, 'close']
            shares = 0
    if shares > 0:
        final_balance = shares * df['close'].iloc[-1]
    else:
        final_balance = balance
    return final_balance

def full_backtest_and_metrics(df, initial_balance=1000):
    """Runs a full backtest and calculates detailed performance metrics."""
    balance = initial_balance
    shares = 0
    equity = [initial_balance]
    for i in range(1, len(df)):
        if shares == 0 and df.at[i, 'position'] == 1 and df.at[i-1, 'position'] == 0:
            shares = balance / df.at[i, 'close']
            balance = 0
        elif shares > 0 and df.at[i, 'position'] == 0 and df.at[i-1, 'position'] == 1:
            balance = shares * df.at[i, 'close']
            shares = 0
        if shares > 0:
            current_equity = shares * df.at[i, 'close']
        else:
            current_equity = balance
        equity.append(current_equity)
    if shares > 0:
        final_balance = shares * df['close'].iloc[-1]
    else:
        final_balance = balance
    df['equity_curve'] = equity
    total_return = (final_balance / initial_balance) - 1
    buy_and_hold_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
    trades = (df['position'].diff() > 0).sum()
    equity_returns = df['equity_curve'].pct_change().dropna()
    sharpe_ratio = (equity_returns.mean() / equity_returns.std()) * np.sqrt(252) if equity_returns.std() != 0 else 0
    days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    cagr = ((final_balance / initial_balance) ** (365.0 / days)) - 1 if days > 0 else 0
    rolling_max = df['equity_curve'].cummax()
    daily_drawdown = df['equity_curve'] / rolling_max - 1.0
    max_drawdown = daily_drawdown.min()
    return {
        "final_balance": final_balance,
        "total_profit_pct": total_return * 100,
        "buy_and_hold_profit_pct": buy_and_hold_return * 100,
        "total_trades": trades,
        "sharpe_ratio": sharpe_ratio,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": max_drawdown * 100,
    }

def main():
    """Main function to run the replication."""
    print_section("Replicating Borst's KAMA+ATR Strategy for TSLA with Optuna (using yfinance)")
    start_date = "2018-08-10"
    end_date = "2025-08-10"
    symbol = "TSLA"

    try:
        df_base = fetch_data_yfinance(symbol, start_date, end_date)

        def objective(trial):
            df = df_base.copy()
            er_period = trial.suggest_int('er_period', 2, 30)
            fast_period = trial.suggest_int('fast_period', 2, 20)
            slow_period = trial.suggest_int('slow_period', 20, 100)
            atr_window = trial.suggest_int('atr_window', 5, 30)
            atr_min_pct = trial.suggest_float('atr_min_pct', 0.5, 3)
            atr_max_pct = trial.suggest_float('atr_max_pct', atr_min_pct, 7)
            atr_exit_pct = trial.suggest_float('atr_exit_pct', atr_max_pct, 20)

            df = kaufman_adaptive_moving_average(df, er_period, fast_period, slow_period)
            df = average_true_range(df, atr_window)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            df = generate_signals(df, atr_min_pct, atr_max_pct, atr_exit_pct)
            final_balance = backtest(df)
            return final_balance

        print_section("Starting Optuna Parameter Optimization (100 trials)")
        print("This will take several minutes...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        print_section("Optimization Complete")
        print(f"Best objective value (Final Balance): ${study.best_value:,.2f}")
        print("Best parameters found:")
        for key, val in study.best_params.items():
            print(f"  - {key}: {val}")

        print_section("Running Final Backtest with Best Parameters")
        best_params = study.best_params
        df_final = df_base.copy()
        df_final = kaufman_adaptive_moving_average(df_final, best_params['er_period'], best_params['fast_period'], best_params['slow_period'])
        df_final = average_true_range(df_final, best_params['atr_window'])
        df_final.dropna(inplace=True)
        df_final.reset_index(drop=True, inplace=True)
        df_final = generate_signals(df_final, best_params['atr_min_pct'], best_params['atr_max_pct'], best_params['atr_exit_pct'])
        
        results = full_backtest_and_metrics(df_final)

        print_section("Final Optimized Backtest Results")
        comparison_df = pd.DataFrame([
            {"Metric": "Total Profit", "Optimized Result": f"{results['total_profit_pct']:.2f}%"},
            {"Metric": "Buy and Hold Profit", "Optimized Result": f"{results['buy_and_hold_profit_pct']:.2f}%"},
            {"Metric": "Total Trades", "Optimized Result": results['total_trades']},
            {"Metric": "Sharpe Ratio", "Optimized Result": f"{results['sharpe_ratio']:.2f}"},
            {"Metric": "CAGR", "Optimized Result": f"{results['cagr_pct']:.2f}%"},
            {"Metric": "Max Drawdown", "Optimized Result": f"{results['max_drawdown_pct']:.2f}%"},
        ])
        print(comparison_df.to_string(index=False))

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
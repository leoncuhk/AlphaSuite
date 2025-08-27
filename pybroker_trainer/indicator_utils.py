"""
A utility module for calculating common technical indicators.
This helps to avoid code duplication across different strategy modules.
"""

import pandas as pd
import talib
import numpy as np

def calculate_slope(series: pd.Series) -> float:
    """Calculates the slope of a series using linear regression."""
    y_val = series
    if len(y_val) < 2 or pd.isna(y_val).all(): return np.nan
    x_val = np.arange(len(y_val))
    valid_indices = ~np.isnan(y_val)
    y_clean, x_clean = y_val[valid_indices], x_val[valid_indices]
    if len(y_clean) < 2: return np.nan
    return np.polyfit(x_clean, y_clean, 1)[0]

def add_common_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Adds a comprehensive set of common technical indicators to the DataFrame.
    Strategies can call this to get a baseline set of features.
    """
    # Ensure columns are lowercase for consistency
    df.columns = [x.lower() for x in df.columns]
    
    close = df['close']
    high = df['high']
    low = df['low']
    op = df['open']
    volume = df['volume']

    # Trend Indicators
    df['adx'] = talib.ADX(high, low, close, timeperiod=params.get('adx_period', 14))
    df['+di'] = talib.PLUS_DI(high, low, close, timeperiod=params.get('di_period', 14))
    df['-di'] = talib.MINUS_DI(high, low, close, timeperiod=params.get('di_period', 14))
    df['adxr'] = talib.ADXR(high, low, close, timeperiod=params.get('adxr_period', 14))

    # Momentum & Cycle
    df['rsi'] = talib.RSI(close, timeperiod=params.get('rsi_period', 14))
    df['willr'] = talib.WILLR(high, low, close, timeperiod=params.get('willr_period', 14))
    df['cmo'] = talib.CMO(close, timeperiod=params.get('cmo_period', 14))
    df['cci'] = talib.CCI(high, low, close, timeperiod=params.get('cci_period', 14))
    df['mom'] = talib.MOM(close, timeperiod=params.get('mom_period', 10))
    df['roc'] = talib.ROC(close, timeperiod=params.get('roc_period', 10))
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=params.get('macd_fast_period', 12), slowperiod=params.get('macd_slow_period', 26), signalperiod=params.get('macd_signal_period', 9))
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    df['ppo'] = talib.PPO(close, fastperiod=params.get('ppo_fast', 12), slowperiod=params.get('ppo_slow', 26))

    # Volatility & Volume
    df['atr'] = talib.ATR(high, low, close, timeperiod=params.get('atr_period', 14))
    df['obv'] = talib.OBV(close, volume)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=params.get('bb_period', 20), nbdevup=params.get('bb_std_dev', 2), nbdevdn=params.get('bb_std_dev', 2))
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    df['bb_percentb'] = np.where(df['bb_width'] > 0, (close - bb_lower) / (bb_upper - bb_lower), 0)

    # Custom Features
    short_win = params.get('short_window', 50)
    sma_short = talib.SMA(close, timeperiod=short_win)
    rolling_std = close.rolling(window=short_win).std()
    df['zscore_sma'] = np.where(rolling_std > 0, (close - sma_short) / rolling_std, 0)
    
    # Stock Trend
    trend_sma_ultrashort_p = params.get('trend_sma_ultrashort_window', 20)
    trend_sma_short_p = params.get('trend_sma_short_window', 50)
    trend_sma_long_p = params.get('trend_sma_long_window', 200)
    trend_slope_window_p = params.get('trend_slope_window', 20)
    trend_slope_threshold = params.get('trend_slope_threshold', 0.01)

    sma_ultrashort = talib.SMA(close, timeperiod=trend_sma_ultrashort_p)
    sma_short = talib.SMA(close, timeperiod=trend_sma_short_p)
    sma_long = talib.SMA(close, timeperiod=trend_sma_long_p)
    df[f'sma_{trend_sma_ultrashort_p}'] = sma_ultrashort
    df[f'sma_{trend_sma_short_p}'] = sma_short
    df[f'sma_{trend_sma_long_p}'] = sma_long

    df['sma_short_slope'] = sma_short.rolling(window=trend_slope_window_p).apply(calculate_slope, raw=True)
    df['sma_long_slope'] = sma_long.rolling(window=trend_slope_window_p).apply(calculate_slope, raw=True)

    conditions = [
        (close > sma_long) & (sma_short > sma_long), # Bullish: Price and short MA are above long MA
        (close < sma_long) & (sma_short < sma_long), # Bearish: Price and short MA are below long MA
    ]
    choices = ['bullish', 'bearish']
    df['Stock_Trend'] = np.select(conditions, choices, default='neutral')

    # One-hot encode the 'Stock_Trend' column
    trend_dummies = pd.get_dummies(df['Stock_Trend'], prefix='trend', dtype=int)
    # Ensure all possible trend columns exist, even if one is not present in the data
    for col in ['trend_bullish', 'trend_bearish', 'trend_neutral']:
        if col not in trend_dummies.columns:
            trend_dummies[col] = 0
    for col in trend_dummies.columns:
        df[col] = trend_dummies[col]
    df.drop('Stock_Trend', axis=1, inplace=True)

    return df
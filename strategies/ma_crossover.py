"""
A self-contained module for the classic 'Moving Average Crossover' strategy.

This is a non-ML, rule-based strategy that serves as a baseline and demonstrates
the framework's ability to handle both ML and non-ML systems. It buys when a
short-term moving average crosses above a long-term moving average.
"""

import pandas as pd
import talib

from pybroker_trainer.strategy_sdk import BaseStrategy

class MaCrossoverStrategy(BaseStrategy):
    """Implements the full logic for the MA Crossover strategy."""

    @staticmethod
    def define_parameters():
        """Defines parameters, their types, defaults, and tuning ranges."""
        return {
            'ema_short_window': {'type': 'int', 'default': 13, 'tuning_range': (5, 20)},
            'ema_long_window': {'type': 'int', 'default': 34, 'tuning_range': (25, 55)},
            'atr_period': {'type': 'int', 'default': 14, 'tuning_range': (10, 30)},
            'initial_stop_atr_multiplier': {'type': 'float', 'default': 2.0, 'tuning_range': (1.5, 4.0)},
            'trailing_stop_atr_multiplier': {'type': 'float', 'default': 3.0, 'tuning_range': (2.5, 6.0)},
            'stop_out_window': {'type': 'int', 'default': 60, 'tuning_range': (20, 120)},
            'risk_per_trade_pct': {'type': 'float', 'default': 0.02, 'tuning_range': (0.01, 0.05)},
        }

    @property
    def is_ml_strategy(self) -> bool:
        """This is a rule-based strategy, not an ML one."""
        return False

    def get_feature_list(self) -> list[str]:
        """Non-ML strategies have no features for a model."""
        return []

    def add_strategy_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and adds features unique to this specific strategy.
        The input dataframe will already contain common indicators.
        """
        # Add EMA indicators
        close = data['close']
        ema_short_p = self.params['ema_short_window']
        ema_long_p = self.params['ema_long_window']
        fast_ma_col = f"ema_{ema_short_p}"
        slow_ma_col = f"ema_{ema_long_p}"
        data[fast_ma_col] = talib.EMA(close, timeperiod=ema_short_p)
        data[slow_ma_col] = talib.EMA(close, timeperiod=ema_long_p)
        return data

    def get_setup_mask(self, data: pd.DataFrame) -> pd.Series:
        """Returns a boolean Series indicating the bars where a trade setup occurs."""
        fast_ma_col = f"ema_{self.params['ema_short_window']}"
        slow_ma_col = f"ema_{self.params['ema_long_window']}"
        
        crossover = (data[fast_ma_col] > data[slow_ma_col]) & (data[fast_ma_col].shift(1) <= data[slow_ma_col].shift(1))
        return crossover.fillna(False)

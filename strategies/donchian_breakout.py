"""
A self-contained module for the 'Donchian Channel Breakout' trading strategy.

This strategy identifies stocks in a confirmed uptrend that are breaking out
above their recent price range, as defined by the upper Donchian Channel.
It uses a machine learning model to qualify the breakout, aiming to filter out
false signals and improve the probability of a successful trend-following trade.
"""

import pandas as pd

from pybroker_trainer.strategy_sdk import BaseStrategy

class DonchianBreakoutStrategy(BaseStrategy):
    """Implements the full logic for the Donchian Channel Breakout strategy."""

    @staticmethod
    def define_parameters():
        """Defines parameters, their types, defaults, and tuning ranges."""
        return {
            'donchian_period': {'type': 'int', 'default': 20, 'tuning_range': (15, 50)},
            'atr_period': {'type': 'int', 'default': 14, 'tuning_range': (10, 30)},
            'initial_stop_atr_multiplier': {'type': 'float', 'default': 2.0, 'tuning_range': (1.5, 4.0)},
            'trailing_stop_atr_multiplier': {'type': 'float', 'default': 3.0, 'tuning_range': (2.5, 6.0)},
            'stop_out_window': {'type': 'int', 'default': 60, 'tuning_range': (20, 120)},
            'probability_threshold': {'type': 'float', 'default': 0.60, 'tuning_range': (0.55, 0.80)},
            'risk_per_trade_pct': {'type': 'float', 'default': 0.02, 'tuning_range': (0.01, 0.05)},
        }

    def get_feature_list(self) -> list[str]:
        """Returns the list of feature column names required by the model."""
        return [
            'roc', 'rsi', 'mom', 'ppo', 'cci',
            'sma_short_slope', 'sma_long_slope',
            'adx', '+di', '-di',
            'donchian_upper', 'donchian_lower', 'donchian_middle',
            'atr', 'volume', 'obv',
            *[f'month_{m}' for m in range(1, 13)]
        ]

    def add_strategy_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and adds features unique to this specific strategy.
        The input dataframe will already contain common indicators.
        """
        # Add Donchian Channels
        donchian_period = self.params.get('donchian_period', 20)
        data['donchian_upper'] = data['high'].rolling(window=donchian_period).max()
        data['donchian_lower'] = data['low'].rolling(window=donchian_period).min()
        data['donchian_middle'] = (data['donchian_upper'] + data['donchian_lower']) / 2
        return data

    def get_setup_mask(self, data: pd.DataFrame) -> pd.Series:
        """Returns a boolean Series indicating the bars where a trade setup occurs."""
        is_uptrend = data['trend_bullish'] == 1
        is_breakout = data['high'] > data['donchian_upper'].shift(1)
        raw_setup_mask = is_uptrend & is_breakout
        return raw_setup_mask & ~raw_setup_mask.shift(1).fillna(False)


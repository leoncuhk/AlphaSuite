"""
Defines the Software Development Kit (SDK) for creating new, self-contained
strategy modules.

This module defines the interface that all trading strategies must adhere to.
By creating a class that inherits from `BaseStrategy`, users can create
custom strategies that are automatically compatible with the platform's
backtesting, scanning, and model tuning engines.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any

from pybroker_trainer.indicator_utils import add_common_indicators
from pybroker_trainer.trader import BaseTrader, RuleBasedTrader

class BaseStrategy(ABC):
    """
    The abstract base class for all trading strategies in AlphaSuite.
    It defines the common interface for both rule-based and ML-based strategies.
    """
    def __init__(self, params: Dict[str, Any] | None = None):
        """
        Initializes the strategy with a set of parameters.

        Args:
            params (dict): A dictionary of parameters for the strategy.
        """
        self.params = params if params is not None else {}

    @staticmethod
    @abstractmethod
    def define_parameters() -> Dict[str, Dict[str, Any]]:
        """
        Defines parameters, their types, defaults, and tuning ranges.

        Returns:
            A dictionary where keys are param names and values are dictionaries
            defining the parameter's properties (e.g.,
            {'type': 'int', 'default': 10, 'tuning_range': (5, 20)}).
        """
        pass

    @abstractmethod
    def get_feature_list(self) -> list[str]:
        """
        Returns the list of feature column names required by the strategy's model.
        """
        pass

    @property
    def is_ml_strategy(self) -> bool:
        """
        Specifies if the strategy is ML-based (True) or rule-based (False).
        This property is crucial for the framework to select the correct execution logic.
        Override and return False for purely rule-based strategies. The default is True.
        """
        return True

    def get_model_config(self) -> dict:
        """
        Returns the configuration for the machine learning model.
        Default is a binary classifier. Override for multiclass models.
        """
        return {'objective': 'binary'}

    def get_extra_context_columns_to_register(self) -> list[str]:
        """
        Returns a list of extra columns not included in the feature list
        to register with pybroker's execution context used by the Trader.
        These columns will be accessible via `ctx.column_name`.
        """
        return []

    def calculate_trailing_stop_target(self, data_df: pd.DataFrame, setup_mask: pd.Series, initial_stop_price_series: pd.Series, atr_multiplier_trailing: float, stop_out_window: int) -> pd.Series:
        """
        Calculates a target for a strategy that uses a trailing stop.
        A target is ONLY calculated for days that meet the setup criteria (setup_mask).
        """
        required_columns = ['atr', 'high', 'low', 'close']
        if not all(c in data_df.columns for c in required_columns):
            raise ValueError("Required columns are missing for trailing stop target.")

        target = pd.Series(np.nan, index=data_df.index) # Default to NaN
        
        for i in data_df.loc[setup_mask].index:
            idx_loc = data_df.index.get_loc(i)
            if idx_loc >= len(data_df) - stop_out_window: continue

            entry_price = data_df['close'].loc[i]
            atr_at_entry = data_df['atr'].loc[i]
            if pd.isna(entry_price) or pd.isna(atr_at_entry) or atr_at_entry <= 0: continue

            initial_stop_price = initial_stop_price_series.loc[i]
            trail_amount = atr_at_entry * atr_multiplier_trailing
            trailing_stop_price = entry_price - trail_amount
            
            stopped_out = False
            last_sim_day_idx = -1
            for j in range(idx_loc + 1, idx_loc + 1 + stop_out_window):
                last_sim_day_idx = j
                current_atr = data_df['atr'].iloc[j]
                if pd.notna(current_atr) and current_atr > 0:
                    trail_amount = current_atr * atr_multiplier_trailing
                
                if data_df['low'].iloc[j] <= initial_stop_price or data_df['low'].iloc[j] <= trailing_stop_price:
                    target.loc[i] = 1 if trailing_stop_price > entry_price else 0
                    stopped_out = True
                    break
                
                trailing_stop_price = max(trailing_stop_price, data_df['high'].iloc[j] - trail_amount)
            
            if not stopped_out and last_sim_day_idx != -1:
                exit_price_on_timeout = data_df['close'].iloc[last_sim_day_idx]
                target.loc[i] = 1 if exit_price_on_timeout > entry_price else 0

        print(f"{setup_mask.sum()=}, {target.sum()=}")
        return target

    def calculate_target(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculates the target variable for model training based on the strategy's logic.
        """
        setup_mask = data['setup_mask']
        initial_stop_atr_multiplier = self.params.get('initial_stop_atr_multiplier', 2.0)
        trailing_stop_atr_multiplier = self.params.get('trailing_stop_atr_multiplier', 3.0)
        stop_out_window = self.params.get('stop_out_window', 60)
        initial_stop_price_series = data['close'] - (data['atr'] * initial_stop_atr_multiplier)
        return self.calculate_trailing_stop_target(
            data_df=data,
            setup_mask=setup_mask,
            initial_stop_price_series=initial_stop_price_series,
            atr_multiplier_trailing=trailing_stop_atr_multiplier,
            stop_out_window=stop_out_window
        )

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        A template method that orchestrates the data preparation process.
        
        This concrete method defines a standard sequence for preparing data:
        1. Add a set of common indicators (RSI, BBands, etc.).
        2. Add all strategy-specific indicators and features.
        3. Calculate the 'setup_mask' based on those features.
        4. Calculate the 'target' variable based on the setups.
        
        This ensures consistency and reduces boilerplate in subclasses.
        """
        df = data.copy()
        df = add_common_indicators(df, self.params)
        df = self.add_strategy_specific_features(df)
        df['setup_mask'] = self.get_setup_mask(df)
        df['target'] = self.calculate_target(df)
        return df

    def add_strategy_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and adds features unique to this specific strategy.
        The input dataframe will already contain common indicators.
        """
        return data

    @abstractmethod
    def get_setup_mask(self, data: pd.DataFrame) -> pd.Series:
        """
        Returns a boolean Series indicating the bars where a trade setup occurs.
        This is used for both target calculation and live scanning.
        """
        pass

    def get_trader(self, model_name: Optional[str], params_map: dict):
        """
        Returns an instance of the appropriate trader for this strategy.
        This default implementation selects the trader based on `is_ml_strategy`.
        Individual strategies can override this for custom trader logic if needed.
        """
        if self.is_ml_strategy:
            return BaseTrader(model_name=model_name, params_map=params_map)
        else:
            return RuleBasedTrader(model_name=None, params_map=params_map)
    
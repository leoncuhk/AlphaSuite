"""
Contains the core logic for managing trading positions, including adding,
closing, and checking positions against live market data.
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import pandas as pd
import talib

from load_cfg import WORKING_DIRECTORY
from tools.yfinance_tool import load_ticker_data
# To break the circular dependency during refactoring, we temporarily import from the old file.
# This will be cleaned up in subsequent steps as we continue to refactor pybroker_trainer1.py.
from quant_engine import infer

logger = logging.getLogger(__name__)

POSITIONS_FILE = os.path.join(WORKING_DIRECTORY, "open_positions.json")

@dataclass
class Position:
    """Represents an open trading position."""
    ticker: str
    entry_price: float
    entry_date: str
    shares: float
    strategy: str
    current_stop_loss: Optional[float] = None
    # --- NEW: Parameters aligned with backtesting logic ---
    # These values should be loaded from the strategy's saved parameters when a position is opened.
    tb_atr_stop_multiplier: float = 2.0
    trailing_atr_multiplier: float = 3.0
    atr_period: int = 14

class TradeManager:
    """Manages open trades, including updating stop losses and checking for exits."""

    def __init__(self, positions_file: str = POSITIONS_FILE):
        self.positions_file = positions_file
        self.open_positions: Dict[str, Position] = self.load_positions()

    def load_positions(self) -> Dict[str, Position]:
        """Loads open positions from the state file."""
        if not os.path.exists(self.positions_file):
            return {}
        try:
            with open(self.positions_file, 'r') as f:
                positions_data = json.load(f)
            # Convert dicts back to Position objects
            return {ticker: Position(**data) for ticker, data in positions_data.items()}
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading positions file: {e}")
            return {}

    def save_positions(self):
        """Saves the current open positions to the state file."""
        try:
            with open(self.positions_file, 'w') as f:
                # Convert Position objects to dicts for JSON serialization
                json.dump({ticker: asdict(pos) for ticker, pos in self.open_positions.items()}, f, indent=4)
        except IOError as e:
            logger.error(f"Error saving positions file: {e}")

    def add_position(self, ticker: str, entry_price: float, entry_date: str, shares: float, strategy: str):
        """
        Adds a new position to the manager, loading strategy-specific parameters
        to ensure live management aligns with backtest logic.
        """
        if ticker in self.open_positions:
            logger.warning(f"Position for {ticker} already exists. Not adding new position.")
            return

        # --- Load strategy-specific parameters from the saved JSON file ---
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, 'pybroker_trainer', 'artifacts')
        strategy_params_filename = os.path.join(model_dir, f'{ticker}_{strategy}_strategy_params.json')
        
        try:
            with open(strategy_params_filename, 'r') as f:
                strategy_params = json.load(f)
            logger.info(f"Loaded strategy parameters for {ticker} {strategy} from {strategy_params_filename}")
        except FileNotFoundError:
            logger.error(f"Strategy parameter file not found: {strategy_params_filename}. Using default stop parameters.")
            strategy_params = {}

        # Create the new position with loaded parameters
        new_position = Position(
            ticker=ticker, entry_price=entry_price, entry_date=entry_date, shares=shares, strategy=strategy,
            # Load params, with fallbacks to the dataclass defaults for safety
            tb_atr_stop_multiplier=strategy_params.get('tb_atr_stop_multiplier', 2.0),
            trailing_atr_multiplier=strategy_params.get('trailing_atr_multiplier', 3.0),
            atr_period=strategy_params.get('atr_period', 14)
        )

        self.open_positions[ticker] = new_position
        self.save_positions()
        print(f"ACTION: ADDED new position for {ticker} with strategy {strategy}.")
        logger.info(f"Added new position: {asdict(new_position)}")

    def close_position(self, ticker: str, exit_price: float):
        """Removes a position from the manager."""
        if ticker in self.open_positions:
            del self.open_positions[ticker]
            self.save_positions()
            logger.info(f"Closed position for {ticker} at {exit_price}.")
        else:
            logger.warning(f"Attempted to close a position that does not exist: {ticker}")

    def _get_latest_data(self, ticker: str, history_days: int = 300) -> Optional[pd.DataFrame]:
        """Fetches latest data required for indicator calculation."""
        try:
            # Always refresh data for live management
            data = load_ticker_data(ticker, refresh=True)
            if not data or 'shareprices' not in data or data['shareprices'].empty:
                logger.warning(f"No data loaded for {ticker}")
                return None
            return data['shareprices']
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    def update_stop_loss(self, position: Position, latest_data: pd.DataFrame) -> Optional[float]:
        """
        Calculates and returns the new stop-loss price based on the simple ATR trailing stop logic
        used in the backtest. This ensures consistency between backtesting and live management.
        """
        if latest_data.empty:
            return position.current_stop_loss

        current_high = latest_data.iloc[-1]['High']
        high_prices = latest_data['High'].astype(float)
        low_prices = latest_data['Low'].astype(float)
        close_prices = latest_data['Adj Close'].astype(float)
        atr_series = talib.ATR(high_prices, low_prices, close_prices, timeperiod=position.atr_period)
        if atr_series.empty or pd.isna(atr_series.iloc[-1]):
            logger.warning(f"Could not calculate ATR for {position.ticker}. Cannot update stop loss.")
            return position.current_stop_loss
        
        current_atr = atr_series.iloc[-1]
        if current_atr <= 0:
            return position.current_stop_loss

        if position.current_stop_loss is None:
            position.current_stop_loss = position.entry_price - (current_atr * position.tb_atr_stop_multiplier)
            logger.info(f"[{position.ticker}] Initial stop loss set to: {position.current_stop_loss:.2f}")

        new_potential_stop = current_high - (current_atr * position.trailing_atr_multiplier)

        if new_potential_stop > position.current_stop_loss:
            position.current_stop_loss = new_potential_stop
            logger.info(f"[{position.ticker}] Trailing stop updated to: {position.current_stop_loss:.2f}")

        return position.current_stop_loss

    def check_positions(self):
        """
        Iterates through open positions, checks for exit signals, updates stop losses,
        and prints recommended actions.
        """
        if not self.open_positions:
            print("No open positions to manage.")
            return

        positions_to_close = []
        for ticker, position in self.open_positions.items():
            logger.info(f"--- Checking position for {ticker} ---")
            latest_data = self._get_latest_data(ticker)
            if latest_data is None:
                logger.warning(f"Could not get data for {ticker}. Skipping check.")
                continue

            current_low = latest_data.iloc[-1]['Low']
            current_close = latest_data.iloc[-1]['Adj Close']
            
            self.update_stop_loss(position, latest_data)

            if position.current_stop_loss and current_low <= position.current_stop_loss:
                print(f"ACTION: CLOSE {ticker} position. Reason: Stop loss breached at {position.current_stop_loss:.2f} (Day's Low: {current_low:.2f})")
                positions_to_close.append(ticker)
                continue

            if position.strategy == 'extremum_classification':
                prediction_result = infer(ticker, position.strategy)
                
                if prediction_result:
                    is_sell_prediction = prediction_result.get('prediction') == 2
                    if is_sell_prediction:
                        print(f"ACTION: CLOSE {ticker} position. Reason: Model SELL signal from {position.strategy} (Probabilities: {prediction_result.get('probabilities')})")
                        positions_to_close.append(ticker)
                        continue

            print(f"ACTION: HOLD {ticker}. Current Price: {current_close:.2f}, Updated Stop Loss: {position.current_stop_loss:.2f if position.current_stop_loss else 'Not Set'}")

        if positions_to_close:
            for ticker in positions_to_close:
                self.close_position(ticker, 0) # Exit price not needed for this logic
            print(f"\nPositions closed: {', '.join(positions_to_close)}")
        
        self.save_positions()
        print("\nTrade management check complete.")
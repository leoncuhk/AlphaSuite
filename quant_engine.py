"""
Core quantitative engine for backtesting, training, tuning, and inference of trading strategies.

This module provides a comprehensive suite of tools built on top of the `pybroker`
backtesting library. It is designed to be run from the command line using `click`
for various tasks:

- `train`: Runs a walk-forward analysis for a given strategy and ticker, trains a
  machine learning model on the final data fold, and saves all artifacts (model,
  results, parameters).
- `tune-strategy`: Performs Bayesian optimization on strategy-level parameters
  (e.g., indicator periods, risk levels) to find the optimal combination based on
  a defined objective function.
- `visualize-model`: Loads and displays the out-of-sample performance metrics and
  charts from a completed `train` run.
- `predict`: Loads a trained model and runs inference on the latest market data
  to generate a trading decision (BUY/HOLD/SELL).
- `scan`: Scans a universe of tickers for active trading signals based on one or
  more strategies.
- `batch-train`: A powerful command to automate the tuning and training process
  across a universe of tickers and strategies, logging results to a CSV.

The engine is designed to be extensible, with strategies encapsulated in their own
modules within the `pybroker_trainer` directory.
"""
import csv
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np
import pybroker
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from decimal import Decimal
from skopt import gp_minimize
from skopt.space import Real, Integer
from pybroker import ExecContext, FeeMode, PositionMode, Strategy, StrategyConfig, TestResult
from pybroker.strategy import WalkforwardWindow
from lightgbm import LGBMClassifier
import os
import json
import pickle
import click
from datetime import datetime, timedelta
from dataclasses import asdict

import matplotlib.pyplot as plt
import traceback

from core.db import get_db
from core.model import Company, PriceHistory
from tools.file_wrapper import convert_to_json_serializable
from tools.yfinance_tool import get_earnings_dates, load_ticker_data
from pybroker_trainer.strategy_loader import load_strategy_class, get_strategy_defaults, get_strategy_tuning_space, STRATEGY_CLASS_MAP
from pybroker_trainer.config_loader import load_strategy_config
from load_cfg import WORKING_DIRECTORY
from core.logging_config import setup_logging

# --- Logging Configuration ---
setup_logging('quant_engine.log')
logger = logging.getLogger(__name__)

class PassThroughModel:
    """A dummy model that always predicts a high probability for the positive class."""
    def __init__(self, n_classes=2):
        self.n_classes_ = n_classes
        self.objective = 'binary' if n_classes == 2 else 'multiclass'

    def predict_proba(self, X):
        # For binary, return [0.0, 1.0] to always pass the probability threshold.
        # For multiclass, return neutral probability as a safe default.
        if self.n_classes_ == 2:
            return np.array([[0.0, 1.0]] * len(X))
        else:
            return np.full((len(X), self.n_classes_), 1.0 / self.n_classes_)

    @property
    def feature_name_(self):
        return [] # No features needed
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def _load_price_data(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Loads historical price data for the ticker from the database.
    """
    db_session = next(get_db())
    try:
        company = db_session.query(Company).filter(Company.symbol == ticker).first()
        if not company:
            logger.error(f"Company with ticker {ticker} not found in database.")
            return pd.DataFrame()

        query = db_session.query(PriceHistory).filter(PriceHistory.company_id == company.id)
        if start_date:
            query = query.filter(PriceHistory.date >= start_date)
        if end_date:
            query = query.filter(PriceHistory.date <= end_date)
        query = query.order_by(PriceHistory.date.asc())

        price_data = pd.read_sql(query.statement, db_session.bind) 
        
        if price_data.empty:
            logger.warning(f"No price data found for {ticker} between {start_date} and {end_date}.")
            return pd.DataFrame()

        # Ensure essential columns are present
        for col in ['open', 'high', 'low', 'close', 'adjclose', 'volume']:
            if col not in price_data.columns:
                logger.error(f"Essential column '{col}' missing from price data for {ticker}.")
                return pd.DataFrame()
        
        price_data['date'] = pd.to_datetime(price_data['date'])
        if price_data['date'].dt.tz is not None:
            price_data['date'] = price_data['date'].dt.tz_localize(None)
        price_data.set_index('date', inplace=True)
        
        logger.info(f"Loaded {len(price_data)} data points for {ticker}.")
        return price_data
    except Exception as e:
        logger.error(f"Error loading price data for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        if db_session and db_session.is_active:
            db_session.close()

def _prepare_base_data(ticker: str, start_date: str, end_date: str, strategy_params: dict) -> pd.DataFrame:
    """
    Performs the initial, static data loading and feature calculation.
    This includes loading price data, calculating seasonality, and fetching earnings.
    This part of the process does not depend on tunable strategy parameters.
    """
    logger.info(f"Preparing base data for {ticker} from {start_date} to {end_date}...")
    
    data_df = _load_price_data(ticker, start_date, end_date)
    if data_df.empty:
        raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}")

    # Calculate Buy-and-Hold Performance for the period
    if not data_df.empty:
        buy_and_hold_return_pct = 0.0
        first_price = data_df['adjclose'].iloc[0]
        last_price = data_df['adjclose'].iloc[-1]
        if first_price != 0:
            buy_and_hold_return_pct = ((last_price - first_price) / first_price) * 100
        logger.info(f"Buy-and-Hold Performance for {ticker} between {data_df.index[0].date()} and {data_df.index[-1].date()}: {buy_and_hold_return_pct:.2f}%")

    # --- Add Seasonality Features ---
    data_df['month'] = data_df.index.month
    month_dummies = pd.get_dummies(data_df['month'], prefix='month', dtype=int)
    for m in range(1, 13):
        if f'month_{m}' not in month_dummies.columns:
            month_dummies[f'month_{m}'] = 0
    data_df = pd.concat([data_df, month_dummies], axis=1)
    data_df.drop('month', axis=1, inplace=True)

    include_earnings_dates = strategy_params.get('include_earnings_dates', False)
    if include_earnings_dates:
        try:
            earnings_dates = get_earnings_dates(ticker)
            if isinstance(earnings_dates, dict) and "error" in earnings_dates:
                logger.warning(f"Could not fetch earnings dates for {ticker}. Error: {earnings_dates['error']}. Skipping pre-earnings features.")
            elif earnings_dates is not None and not earnings_dates.empty:
                earnings_dates = earnings_dates.index.tz_localize(None).normalize()
                earnings_series = pd.Series(pd.NaT, index=data_df.index)
                for date in data_df.index:
                    future_earnings = earnings_dates[earnings_dates > date]
                    if not future_earnings.empty:
                        earnings_series.loc[date] = future_earnings.min()
                data_df['days_to_earnings'] = (earnings_series - data_df.index).dt.days
                pre_earnings_window = strategy_params.get('pre_earnings_window', 21)
                data_df['is_pre_earnings_window'] = ((data_df['days_to_earnings'] >= 0) & (data_df['days_to_earnings'] <= pre_earnings_window)).astype(int)
            else:
                data_df['days_to_earnings'], data_df['is_pre_earnings_window'] = -1, 0
        except Exception as e:
            logger.warning(f"Could not fetch earnings dates for {ticker}. Error: {e}. Skipping pre-earnings features.")
            data_df['days_to_earnings'], data_df['is_pre_earnings_window'] = -1, 0
    
    # --- Reformat DataFrame for PyBroker ---
    # PyBroker's Strategy class expects lowercase column names and 'date'/'symbol' as columns.
    df_formatted = data_df.reset_index() # Move 'Date' from index to a column
    df_formatted['symbol'] = ticker
    
    # --- Force pybroker to use adjusted prices for backtesting and features ---
    # This aligns the backtest environment with the training target calculation by
    # replacing the raw OHLC columns with their split/dividend-adjusted values.
    adjustment_factor = df_formatted['adjclose'] / df_formatted['close']
    df_formatted['open'] = df_formatted['open'] * adjustment_factor
    df_formatted['high'] = df_formatted['high'] * adjustment_factor
    df_formatted['low'] = df_formatted['low'] * adjustment_factor
    df_formatted['close'] = df_formatted['adjclose']
    return df_formatted

def plot_equity_curve(result: TestResult, title: str) -> Optional[plt.Figure]:
    """
    Plots the portfolio equity curve from a pybroker TestResult object.
    Returns the matplotlib Figure object.
    """
    if not hasattr(result, 'portfolio') or result.portfolio.empty:
        logger.warning("No portfolio data found in results to plot.")
        return None

    portfolio_df = result.portfolio

    if 'market_value' not in portfolio_df.columns:
        logger.warning("Portfolio DataFrame is missing 'market_value' column.")
        return None

    fig, ax = plt.subplots(figsize=(15, 7))
    portfolio_df['market_value'].plot(ax=ax, label='Portfolio Value')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True)
    return fig

def plot_trades_on_chart(result: TestResult, ticker: str, title: str) -> Optional[plt.Figure]:
    """
    Plots the executed trades on top of the price chart to visualize strategy behavior.
    Returns the matplotlib Figure object.
    """
    if not hasattr(result, 'trades') or result.trades.empty:
        logger.warning("No trade data found in results to plot.")
        return None

    trades_df = result.trades
    
    # Load the price data for the backtest period to plot the price curve
    start_date = result.start_date.strftime('%Y-%m-%d')
    end_date = result.end_date.strftime('%Y-%m-%d')
    
    data_dict = load_ticker_data(ticker, start_date, end_date)
    if not data_dict or 'shareprices' not in data_dict or data_dict['shareprices'] is None:
        logger.error(f"Could not load price data for {ticker} to plot trades.")
        return None
    
    price_data = data_dict['shareprices']
    if price_data.empty:
        logger.error(f"Price data DataFrame is empty for {ticker}.")
        return None
    
    if 'Date' in price_data.columns:
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_data = price_data.set_index('Date')
    elif not isinstance(price_data.index, pd.DatetimeIndex):
        price_data.index = pd.to_datetime(price_data.index)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(price_data.index, price_data['Adj Close'], label=f'{ticker} Price', color='skyblue', alpha=0.7, zorder=1)

    # Separate winning and losing trades for different coloring
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]

    ax.scatter(winning_trades['entry_date'], winning_trades['entry'], marker='^', color='green', s=100, label='Winning Entry', zorder=2)
    ax.scatter(losing_trades['entry_date'], losing_trades['entry'], marker='^', color='red', s=100, label='Losing Entry', zorder=2)
    
    # Connect entry and exit points with lines
    for _, trade in trades_df.iterrows():
        color = 'green' if trade['pnl'] > 0 else 'red'
        ax.plot([trade['entry_date'], trade['exit_date']], [trade['entry'], trade['exit']], color=color, linestyle='--', linewidth=1.5, zorder=3)

    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True)
    return fig

def plot_feature_importance(feature_names, all_importances, top_n=20) -> Optional[plt.Figure]:
    """
    Calculates and plots the average feature importance from all walk-forward folds.
    Returns the matplotlib Figure object.
    """
    if not all_importances or not feature_names:
        logger.warning("No feature importances or feature names were collected to plot.")
        return None

    # Calculate the mean and standard deviation of importances across all folds
    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_importances,
        'std': std_importances
    }).sort_values(by='importance', ascending=False)

    # Plot the top N features
    top_features_df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top_features_df['feature'], top_features_df['importance'], xerr=top_features_df['std'], align='center', capsize=5, color='skyblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Mean Feature Importance (LGBM Gain)')
    ax.set_ylabel('Features')
    ax.set_title(f'Top {top_n} Feature Importances (Averaged Across Walk-Forward Folds)')
    ax.invert_yaxis()  # Highest importance at the top
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjust layout to make room for labels
    return fig

def run_visualize_model(ticker: str, strategy_type: str) -> Optional[Dict]:
    """
    Loads saved model artifacts from a 'train' run and generates visualization assets.

    Args:
        ticker: The stock ticker symbol.
        strategy_type: The strategy type.

    Returns:
        A dictionary containing DataFrames and matplotlib Figures for UI display,
        or None if artifacts cannot be loaded.
    """
    logger.info(f"--- Visualizing Results for {ticker} with {strategy_type} strategy ---")
    model_dir = os.path.join('pybroker_trainer', 'artifacts')
    
    # --- Construct file paths ---
    results_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_results.pkl')
    features_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_features.json')
    importances_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_importances.pkl') # type: ignore
    model_params_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_best_params.json')
    strategy_params_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_best_strategy_params.json')

    # --- Load artifacts, handling missing files gracefully for non-ML strategies ---
    try:
        # The results file is mandatory for any visualization.
        with open(results_filename, 'rb') as f: result = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Could not find required results file: {results_filename}. Please run the 'train' command first.")
        return None

    # Optional artifacts (may not exist for non-ML strategies)
    features = None
    all_importances = None
    best_model_params = None
    best_strategy_params = None

    if os.path.exists(features_filename):
        with open(features_filename, 'r') as f: features = json.load(f)
    if os.path.exists(importances_filename):
        with open(importances_filename, 'rb') as f: all_importances = pickle.load(f)
    if os.path.exists(model_params_filename):
        with open(model_params_filename, 'r') as f: best_model_params = json.load(f)
    if os.path.exists(strategy_params_filename):
        with open(strategy_params_filename, 'r') as f: best_strategy_params = json.load(f)

    # --- Clean DataFrames for Streamlit/Arrow compatibility ---
    # This prevents serialization errors when displaying the DataFrame in the UI.
    # If a column has mixed types (e.g., numbers and strings), pyarrow can fail.
    # Forcing object columns to string is the safest way to ensure they display correctly.
    metrics_df = result.metrics_df.copy()
    for col in metrics_df.columns:
        if metrics_df[col].dtype == 'object':
            metrics_df[col] = metrics_df[col].astype(str)
    trades_df = result.trades.copy()
    for col in trades_df.columns:
        if trades_df[col].dtype == 'object':
            trades_df[col] = trades_df[col].astype(str)

    # --- Generate plots, checking if the necessary data exists ---
    importance_fig = None
    if features and all_importances:
        importance_fig = plot_feature_importance(features, all_importances)

    return {
        "metrics_df": metrics_df,
        "trades_df": trades_df,
        "equity_fig": plot_equity_curve(result, f'Walk-Forward Equity for {ticker} ({strategy_type})'),
        "trades_fig": plot_trades_on_chart(result, ticker, f'Trades for {ticker} ({strategy_type})'),
        "importance_fig": importance_fig,
        "best_model_params": best_model_params,
        "best_strategy_params": best_strategy_params
    }

def _load_strategy_params(ticker: str, strategy_type: str) -> Optional[dict]:
    """
    Loads strategy parameters for a given ticker and strategy, prioritizing
    tuned parameters over last-run parameters, and falling back to defaults.
    """
    strategy_class = load_strategy_class(strategy_type)
    if not strategy_class:
        logger.error(f"Cannot load params: Strategy class for '{strategy_type}' not found.")
        return None
    
    # Start with defaults
    params = get_strategy_defaults(strategy_class)

    model_dir = os.path.join('pybroker_trainer', 'artifacts')
    best_params_path = os.path.join(model_dir, f'{ticker}_{strategy_type}_best_strategy_params.json')
    last_run_params_path = os.path.join(model_dir, f'{ticker}_{strategy_type}_strategy_params.json')

    # Prioritize best tuned params
    if os.path.exists(best_params_path):
        try:
            with open(best_params_path, 'r') as f:
                tuned_params = json.load(f)
            params.update(tuned_params)
            logger.info(f"Loaded BEST strategy parameters for {ticker}-{strategy_type}")
        except Exception as e:
            logger.warning(f"Could not load best strategy params file {best_params_path}. Error: {e}")
    # Fallback to last-run params
    elif os.path.exists(last_run_params_path):
        try:
            with open(last_run_params_path, 'r') as f:
                last_run_params = json.load(f)
            params.update(last_run_params)
            logger.info(f"Loaded last-run strategy parameters for {ticker}-{strategy_type}")
        except Exception as e:
            logger.warning(f"Could not load last-run strategy params file {last_run_params_path}. Error: {e}")
    
    return params

def custom_predict_fn(model_bundle, data):
    """
    Custom predict function for LGBM models.
    Relies solely on the model_bundle returned by train_fn.
    Returns a 2D NumPy array compatible with PyBroker's ctx.preds.
    """
    # Unpack the model, features, and scaler from the bundle.
    model = model_bundle['model']
    features = model_bundle['features']

    if isinstance(model, PassThroughModel):
        return model.predict_proba(data)

    # --- Handle cases where the model was not fitted due to lack of data ---
    # The train_fn returns an unfitted model if the training data for a fold is empty.
    # This check prevents a crash during the prediction phase for that fold.
    if not hasattr(model, 'n_classes_'):
        # This model is unfitted. Determine the expected number of classes from its objective.
        n_classes = 3 if model.objective == 'multiclass' else 2
        logger.warning("custom_predict_fn received an unfitted model. Returning neutral probabilities.")
        # Return a neutral probability for every row in the input data.
        # This ensures the backtest can continue without taking trades in this fold.
        return np.full((len(data), n_classes), 1.0 / n_classes, dtype=np.float64)

    n_classes = model.n_classes_
    expected_features = model.feature_name_ if hasattr(model, 'feature_name_') else features

    if data.empty:
        logger.warning("custom_predict_fn received empty data, returning neutral probabilities.")
        # Return an empty array with the correct number of columns (classes)
        return np.empty((0, n_classes), dtype=np.float64)
    if not all(f in data.columns for f in expected_features):
        missing = set(expected_features) - set(data.columns)
        raise ValueError(f"Missing features: {missing}")

    input_data = data[expected_features]
    probabilities = model.predict_proba(input_data)  # Return full array to match working case
    return probabilities  # Let PyBroker handle multi-row output
        
# --- NEW: Custom Strategy class for Expanding Window Walk-Forward ---
class ExpandingWindowStrategy(pybroker.Strategy):
    """
    A custom Strategy class that overrides the default walk-forward split logic
    to implement an expanding window instead of a sliding one. This is necessary
    for versions of pybroker that do not have a `rolling` parameter.
    """
    def walkforward_split(
        self,
        df: pd.DataFrame,
        windows: int,
        lookahead: int,
        train_size: float,  # This parameter is ignored in this implementation
        shuffle: bool = False,
    ) -> iter:
        """
        Generates train/test splits for an expanding window walk-forward analysis.
        The training data grows with each window to include the previous test set.
        """
        logger.info("Using custom ExpandingWindowStrategy to generate expanding window splits...")
        
        date_col = 'date'  # pybroker lowercases column names
        unique_dates = np.unique(df[date_col])
        n_dates = len(unique_dates)

        if windows <= 0 or n_dates == 0: return

        # The data is divided into `windows + 1` chunks. The first is for initial training.
        test_size_days = n_dates // (windows + 1)
        if test_size_days < 1:
            logger.error(f"Not enough data for {windows} windows. Each test set would have < 1 day.")
            return

        train_end_date_idx = n_dates - (windows * test_size_days) - 1

        for i in range(windows):
            test_start_date_idx = train_end_date_idx + lookahead
            
            # For the last window, extend the test set to the end of the data to include any remainder.
            if i == windows - 1:
                test_end_date_idx = n_dates - 1
            else:
                test_end_date_idx = test_start_date_idx + test_size_days - 1
            if test_end_date_idx >= n_dates: test_end_date_idx = n_dates - 1
            if test_start_date_idx > test_end_date_idx: break

            train_end_date = unique_dates[train_end_date_idx]
            test_start_date = unique_dates[test_start_date_idx]
            test_end_date = unique_dates[test_end_date_idx]

            train_indices = df.index[df[date_col] <= train_end_date].to_numpy()
            test_indices = df.index[(df[date_col] >= test_start_date) & (df[date_col] <= test_end_date)].to_numpy()
            if shuffle: np.random.shuffle(train_indices)
            yield WalkforwardWindow(train_indices, test_indices)
            train_end_date_idx = test_end_date_idx

BASE_CONTEXT_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'target', 'setup_mask', 'atr']

def run_pybroker_walkforward(ticker: str = 'SPY', start_date: str = '2000-01-01', end_date: Optional[str] = None, strategy_type: str = 'trend_following', tune_hyperparameters: bool = True, plot_results: bool = True, save_assets: bool = False, override_params: dict = None, use_tuned_strategy_params: bool = False, disable_inner_parallelism: bool = False, preloaded_data_df: pd.DataFrame = None, preloaded_features: list = None, commission_cost: float = 0.0, calc_bootstrap: bool = True, stop_event_checker=None):
    """
    Runs the full walk-forward analysis for a given ticker.
    """
    features = [] # Ensure features is defined in the outer scope for the finally block
    context_columns_to_register = [] # for the finally block
    last_best_params = None
    last_trained_model = None
    is_ml = True # Assume ML strategy by default
    all_quality_scores = [] # For tracking model quality across folds

    try:
        if stop_event_checker and stop_event_checker():
            logger.warning("Stop event detected before starting walkforward analysis.")
            # Return the expected tuple format
            return None, []

        # --- Unify default end_date handling ---
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        # --- Load strategy config from JSON file ---
        strategy_class = load_strategy_class(strategy_type)
        base_params = get_strategy_defaults(strategy_class) if strategy_class else {}
        base_params.update({ # Add runtime params
            'disable_inner_parallelism': disable_inner_parallelism,
        })

        # Load tuned strategy parameters if requested 
        if use_tuned_strategy_params:
            strategy_params = load_strategy_config(strategy_type, base_params)
        else:
            strategy_params = base_params

        # Allow overriding parameters for optimization 
        if override_params:
            logger.info(f"Overriding strategy parameters with: {override_params}")
            strategy_params.update(override_params)

        if strategy_class:
            strategy_instance = strategy_class(params=strategy_params)
            is_ml = strategy_instance.is_ml_strategy

        # --- OPTIMIZATION: Use pre-loaded data if provided (e.g., from tune_strategy) ---
        if preloaded_data_df is not None and preloaded_features is not None:
            logger.info("Using pre-loaded data for backtest.")
            data_df = preloaded_data_df
            features = preloaded_features
        else:
            if strategy_class:
                base_df = _prepare_base_data(ticker, start_date, end_date, strategy_params)
                data_df = strategy_instance.prepare_data(data=base_df)
                features = strategy_instance.get_feature_list()
                context_columns_to_register = BASE_CONTEXT_COLUMNS + strategy_instance.get_extra_context_columns_to_register()
            else: # Fallback for legacy strategies
                raise NotImplementedError(f"Strategy '{strategy_type}' has not been migrated to the new encapsulated format. Please create a strategy module for it.")

        if is_ml:
            # --- NEW: Add a global check for minimum setups before starting the walk-forward ---
            min_total_setups_for_run = 60 # A reasonable minimum for the entire dataset
            valid_setups = data_df.dropna(subset=['target'])
            if len(valid_setups) < min_total_setups_for_run:
                logger.error(f"Strategy '{strategy_type}' for {ticker} has only {len(valid_setups)} total setups.")
                logger.error(f"This is insufficient for a reliable walk-forward backtest (min: {min_total_setups_for_run}).")
                logger.error("Consider using the 'pre-scan-universe' command to find tickers with more frequent setups.")
                return None, [] # Abort the run early

        # Filter out features that might not have been calculated or don't exist
        # features = [f for f in data_df.columns if f in features]
        # data_df.dropna(subset=features + ['target'], inplace=True)

        if is_ml:
            pybroker.register_columns(features)
        pybroker.register_columns(context_columns_to_register) # Always register context columns

        # Define a function to provide input data for the model.
        # This is required when custom columns are registered. It tells pybroker
        # which columns from the main DataFrame should be fed to the model for prediction.
        def model_input_data_fn(data):
            return data[features]

        # Step 2: Define the training function for the model.
        # This function will be called by PyBroker for each walk-forward window.
        def train_fn(symbol, train_data, test_data, **kwargs):
            nonlocal all_feature_importances, last_best_params, last_trained_model
            from sklearn.model_selection import train_test_split # Local import for this function
            model_config = strategy_instance.get_model_config()
            # --- NEW: Enhanced logging and checks for data validity ---
            train_start = train_data['date'].min().date() if not train_data.empty else 'N/A'
            train_end = train_data['date'].max().date() if not train_data.empty else 'N/A'
            logger.info(f"[{symbol}] Training fold: {train_start} to {train_end}. Initial samples: {len(train_data)}")

            # Use nonlocal to modify the list defined in the outer scope
            nonlocal all_feature_importances, last_best_params, last_trained_model
            pybroker.disable_logging()

            if train_data.empty:
                logger.warning(f"[{symbol}] Training data is empty for this fold. This is expected if the stock did not exist for the full period. Returning untrained model.")
                model = LGBMClassifier(random_state=42, n_jobs=1, **model_config)
                return {'model': model, 'features': features}

            # --- Filter training data to only include rows with valid targets ---
            if 'target' in train_data.columns:
                train_data = train_data.dropna(subset=['target'])

            # --- NEW: More robust check for minimum samples per fold ---
            min_total_samples = 30  # Increased minimum total setups required for training
            min_class_samples = 10  # Minimum required setups for EACH class (win and loss)

            if train_data.empty or len(train_data) < min_total_samples:
                logger.warning(f"[{symbol}] Training data is empty or has insufficient samples ({len(train_data)} < {min_total_samples}). No model will be trained for this fold.")
                model = LGBMClassifier(random_state=42, n_jobs=1, **model_config)
                return {'model': model, 'features': features}
            
            # Check for minimum samples in the minority class
            if 'target' in train_data.columns and not train_data['target'].value_counts().empty:
                if len(train_data['target'].unique()) < 2 or train_data['target'].value_counts().min() < min_class_samples:
                    logger.warning(f"[{symbol}] Minority class has too few samples ({train_data['target'].value_counts().min()} < {min_class_samples}). No model will be trained for this fold.")
                    model = LGBMClassifier(random_state=42, n_jobs=1, **model_config)
                    return {'model': model, 'features': features}

            # --- Determine if tuning is feasible for this specific fold ---
            can_tune = tune_hyperparameters
            if can_tune:
                n_splits = 3 # Must match the cv_splitter below
                min_samples_for_tuning = 20 # A reasonable minimum to attempt tuning
                # Check if any class has fewer samples than n_splits, which would break StratifiedKFold
                if 'target' in train_data.columns and not train_data['target'].value_counts().empty:
                    min_class_count = train_data['target'].value_counts().min()
                    if min_class_count < n_splits or len(train_data) < min_samples_for_tuning:
                        logger.warning(f"[{symbol}] Insufficient samples for tuning (Total: {len(train_data)}, Min Class: {min_class_count}).")
                        logger.warning(f"[{symbol}] Disabling hyperparameter tuning for this fold and using default parameters.")
                        can_tune = False
                else:
                    logger.warning(f"[{symbol}] Target column missing or empty in train_data. Disabling tuning for this fold.")
                    can_tune = False

            if can_tune:
                best_params = _tune_hyperparameters_with_gp_minimize(
                    train_data=train_data,
                    features=features,
                    model_config=model_config,
                    stop_event_checker=stop_event_checker,
                    symbol=symbol
                )
                
                final_model = LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced', **model_config, **best_params)
                final_model.fit(train_data[features], train_data['target'].astype(int))

                last_best_params = best_params
                last_trained_model = final_model
                if hasattr(final_model, 'feature_importances_'):
                    all_feature_importances.append(final_model.feature_importances_)
                return {'model': final_model, 'features': features}
            else:
                # --- Train with default hyperparameters (no tuning) ---
                logger.info("Running with default hyperparameters (no tuning).")
                # --- Train LGBM with default hyperparameters ---
                try:
                    sub_train, sub_val = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['target'])
                except ValueError:
                    logger.warning(f"[{symbol}] Could not create validation split due to class imbalance. Proceeding without quality gate for this fold.")
                    sub_train, sub_val = train_data, pd.DataFrame()

                n_jobs = 1 if strategy_params.get('disable_inner_parallelism') else -1
                default_lgbm_params = {'random_state': 42, 'n_jobs': n_jobs, 'class_weight': 'balanced', 'min_child_samples': 5, **model_config}
                temp_model = LGBMClassifier(**default_lgbm_params)
                temp_model.fit(sub_train[features], sub_train['target'].astype(int))

                # Add a model quality gate to prevent poorly performing models from being used in backtest
                auc_score = 0.5
                if not sub_val.empty and len(np.unique(sub_val['target'])) > 1:
                    val_preds = temp_model.predict_proba(sub_val[features])
                    if model_config.get('objective') == 'binary':
                        auc_score = roc_auc_score(sub_val['target'], val_preds[:, 1])
                    else:
                        auc_score = roc_auc_score(sub_val['target'], val_preds, multi_class='ovr')

                # Track the quality score for this fold
                all_quality_scores.append(auc_score)

                min_auc_threshold = 0.52
                if auc_score < min_auc_threshold:
                    logger.warning(f"[{symbol}] Model quality check failed for this fold (AUC: {auc_score:.3f} < {min_auc_threshold}).")
                    logger.warning(f"[{symbol}] Discarding trained model and using a pass-through model instead.")
                    final_model = PassThroughModel(n_classes=model_config.get('num_class', 2))
                else:
                    logger.info(f"[{symbol}] Model quality check passed (AUC: {auc_score:.3f}). Retraining on full fold data.")
                    final_model = LGBMClassifier(**default_lgbm_params)
                    final_model.fit(train_data[features], train_data['target'].astype(int))
                    if hasattr(final_model, 'feature_importances_'):
                        all_feature_importances.append(final_model.feature_importances_)

                last_trained_model = final_model
                return {'model': final_model, 'features': features}

        # Step 3: Register the model with PyBroker.
        # We don't pass `indicators` because they are already calculated and part of `data_df`.
        model_name = 'binary_classifier'
        model_source = pybroker.model(name=model_name, fn=train_fn, predict_fn=custom_predict_fn, input_data_fn=model_input_data_fn)

        # Step 4: Configure StrategyConfig and instantiate the correct trader
        strategy_config = StrategyConfig(
            position_mode=PositionMode.LONG_ONLY,
            fee_mode=pybroker.FeeMode.PER_SHARE if commission_cost > 0 else None,
            fee_amount=commission_cost
        )
        strategy = ExpandingWindowStrategy(data_source=data_df, start_date=start_date, end_date=end_date, config=strategy_config)
        
        trader = strategy_instance.get_trader(model_name if is_ml else None, {ticker: strategy_params})
        
        if is_ml:
            models_to_use = model_source if isinstance(model_source, list) else [model_source]
            strategy.add_execution(trader.execute, [ticker], models=models_to_use)
        else:
            # Non-ML strategies don't need a model passed to their execution
            strategy.add_execution(trader.execute, [ticker])

        # Step 5: Run the walk-forward analysis
        all_feature_importances = [] # Reset before running
        logger.info("Starting PyBroker walk-forward analysis...")

        # --- REVISED: Walk-Forward Configuration for Compatibility ---
        # The ability to specify `test_size` in walkforward is a feature of newer pybroker versions.
        # To ensure compatibility, we will revert to calculating the number of `windows` based on the
        # total length of the dataset. This is a robust method that works across versions.
        total_years = (data_df['date'].max() - data_df['date'].min()).days / 365.25 if not data_df.empty else 0
        
        if total_years < 4: # Need at least ~4 years for a meaningful split
            logger.error(f"Not enough data ({total_years:.2f} years) for a walk-forward. Minimum 4 years required.")
            return None

        if total_years >= 20:
            windows = 4
        elif total_years >= 15:
            windows = 3
        elif total_years >= 8:
            windows = 2
        else:
            windows = 1
        
        # `train_size` is not used for expanding windows but is a required parameter.
        train_size_prop = 0.7
        logger.info(f"Total data history: {total_years:.2f} years. Using {windows} walk-forward windows.")

        # --- Log the walk-forward split dates for clarity ---
        try:
            # Use the same ExpandingWindowStrategy to visualize the splits that will actually be used.
            temp_strategy_for_split = ExpandingWindowStrategy(data_source=data_df, start_date=start_date, end_date=end_date, config=strategy_config)
            logger.info("--- Walk-Forward Analysis Splits ---")
            for i, (train_idx, test_idx) in enumerate(temp_strategy_for_split.walkforward_split(data_df, windows=windows, train_size=train_size_prop, lookahead=1)):
                try:
                    # --- Add a more robust guard against invalid splits from the generator ---
                    if len(train_idx) == 0 or len(test_idx) == 0:
                        logger.warning(f"Skipping display for Fold {i+1} as it contains an empty train/test split.")
                        continue

                    train_start_date = data_df.loc[train_idx[0]]['date'].date()
                    train_end_date = data_df.loc[train_idx[-1]]['date'].date()
                    test_start_date = data_df.loc[test_idx[0]]['date'].date()
                    test_end_date = data_df.loc[test_idx[-1]]['date'].date()
                    logger.info(f"Fold {i+1}:")
                    logger.info(f"  Train: {train_start_date} to {train_end_date} ({len(train_idx)} bars)")
                    logger.info(f"  Test:  {test_start_date} to {test_end_date} ({len(test_idx)} bars)")
                except IndexError:
                    logger.warning(f"Skipping display for Fold {i+1} due to out-of-bounds indices from pybroker splitter.")
                    traceback.print_exc()
                    continue
            logger.info("------------------------------------")
        except Exception as e:
            logger.warning(f"Could not display walk-forward splits due to an error: {e}")
            traceback.print_exc()

        if stop_event_checker and stop_event_checker():
            logger.warning("Stop event detected before starting pybroker walkforward.")
            return None, []

        result = strategy.walkforward(
            windows=windows,
            train_size=train_size_prop,
            lookahead=1,            
            calc_bootstrap=calc_bootstrap,
            warmup=2,
        )

        if stop_event_checker and stop_event_checker():
            logger.warning("Stop event detected during pybroker walkforward. Results may be incomplete.")
            # Return whatever result we have, but don't save assets.
            return result, all_quality_scores

        if save_assets:
            if stop_event_checker and stop_event_checker():
                logger.warning("Stop event detected before saving assets. Aborting save.")
                return result, all_quality_scores

            _save_walkforward_artifacts(
                ticker=ticker,
                strategy_type=strategy_type,
                result=result,
                is_ml=is_ml,
                last_trained_model=last_trained_model,
                features=features,
                all_feature_importances=all_feature_importances,
                tune_hyperparameters=tune_hyperparameters,
                last_best_params=last_best_params,
                strategy_params=strategy_params
            )

        logger.info(f"\n--- Walk-Forward Analysis Results for {ticker} with {strategy_type} strategy ---")
        logger.info("NOTE: These metrics are calculated on out-of-sample test windows only, providing a more realistic performance estimate.")
        logger.info(result.metrics_df.to_string())
        if plot_results:
            fig_equity = plot_equity_curve(result, f'Walk-Forward Equity Curve for {ticker}')
            if fig_equity: 
                plt.show()
            if all_feature_importances:
                fig_importance = plot_feature_importance(features, all_feature_importances)
                if fig_importance:
                    plt.show()

    except Exception as e:
        logger.error(f"An error occurred during the PyBroker walk-forward analysis for {ticker}: {e}")
        traceback.print_exc()
        return None, []
    finally:
        # Unregister columns to clean up global scope
        if features:
            pybroker.unregister_columns(features)
        if context_columns_to_register:
            pybroker.unregister_columns(context_columns_to_register)
    return result, all_quality_scores # Return the result object and quality scores

def _save_walkforward_artifacts(ticker, strategy_type, result, is_ml, last_trained_model, features, all_feature_importances, tune_hyperparameters, last_best_params, strategy_params):
    """Helper function to save all assets from a walk-forward run."""
    logger.info(f"--- Saving artifacts for {ticker} - {strategy_type} ---")
    model_dir = os.path.join('pybroker_trainer', 'artifacts')
    os.makedirs(model_dir, exist_ok=True)

    # --- Always save core backtest artifacts ---
    # 1. Save the full backtest result object
    results_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_results.pkl')
    with open(results_filename, 'wb') as f:
        pickle.dump(result, f)
    logger.info(f"Saved backtest results to {results_filename}")

    # 2. Save the strategy parameters used for the run
    strategy_params_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_strategy_params.json')
    with open(strategy_params_filename, 'w') as f:
        json.dump(strategy_params, f, indent=4)
    logger.info(f"Saved strategy parameters to {strategy_params_filename}")

    # --- Save ML-specific artifacts only if applicable ---
    if is_ml and last_trained_model:
        # 1. Save the model object
        model_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(last_trained_model, f)
        logger.info(f"Saved final trained model to {model_filename}")

        # 2. Save the features list
        features_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_features.json')
        with open(features_filename, 'w') as f:
            json.dump(features, f, indent=4)
        logger.info(f"Saved feature list to {features_filename}")

        # 3. Save feature importances
        importances_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_importances.pkl')
        with open(importances_filename, 'wb') as f:
            pickle.dump(all_feature_importances, f)
        logger.info(f"Saved feature importances to {importances_filename}")

        # 4. Save the best model hyperparameters if tuning was enabled
        if tune_hyperparameters and last_best_params:
            params_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_best_params.json')
            with open(params_filename, 'w') as f:
                json.dump(convert_to_json_serializable(last_best_params), f, indent=4)
            logger.info(f"Saved best hyperparameters from last fold to {params_filename}")

def _tune_hyperparameters_with_gp_minimize(train_data, features, model_config, stop_event_checker, symbol):
    """
    Performs hyperparameter tuning for the LGBMClassifier using gp_minimize.
    This offers fine-grained control and supports an interruptible callback.
    Returns a dictionary of the best parameters found.
    """
    from sklearn.model_selection import cross_val_score # Local import
    logger.info(f"[{symbol}] Starting hyperparameter tuning with gp_minimize...")
    
    search_spaces_dict = {
        'learning_rate': Real(0.01, 0.3, 'log-uniform'),
        'n_estimators': Integer(50, 500),
        'num_leaves': Integer(20, 100),
        'max_depth': Integer(-1, 50),
        'reg_alpha': Real(0.0, 1.0, 'uniform'), # L1 regularization
        'reg_lambda': Real(0.0, 1.0, 'uniform'), # L2 regularization
    }
    dimensions = list(search_spaces_dict.values())
    dimensions_names = list(search_spaces_dict.keys())

    def hp_objective(params):
        """Objective function for gp_minimize to maximize cross-validated AUC."""
        if stop_event_checker and stop_event_checker():
            logger.warning(f"[{symbol}] Stop event detected in HP tuning objective. Aborting this run.")
            return 1.0 # Return a high value (bad score) for minimization

        param_dict = {name: val for name, val in zip(dimensions_names, params)}
        model = LGBMClassifier(random_state=42, n_jobs=1, class_weight='balanced', **model_config, **param_dict)
        
        cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scoring_metric = 'roc_auc_ovr' if model_config.get('objective') == 'multiclass' else 'roc_auc'
        
        scores = cross_val_score(model, train_data[features], train_data['target'].astype(int), cv=cv_splitter, scoring=scoring_metric, n_jobs=-1)
        
        mean_score = np.mean(scores)
        # We want to maximize the score, so we return its negative for minimization
        return -mean_score

    def skopt_callback(res):
        """Callback for gp_minimize to check for stop event."""
        if stop_event_checker and stop_event_checker():
            logger.warning(f"[{symbol}] Stop event received during hyperparameter tuning. Halting search.")
            return True # Returning True stops the optimization loop.
        return False

    opt_result = gp_minimize(
        func=hp_objective,
        dimensions=dimensions,
        n_calls=32,
        random_state=42,
        callback=skopt_callback
    )

    best_params = {name: val for name, val in zip(dimensions_names, opt_result.x)}
    logger.info(f"[{symbol}] Best hyperparameters found: {best_params}")
    return best_params

def infer(ticker: str, strategy_type: str, data_df: pd.DataFrame = None, strategy_params: dict = None):
    """
    Loads a trained model and its artifacts to make a prediction on the latest data.
    Can accept a pre-prepared DataFrame to avoid redundant data loading.
    """
    # logger.info(f"--- Running Inference for {ticker} with {strategy_type} strategy ---")
    model_dir = os.path.join('pybroker_trainer', 'artifacts')
    # --- Construct file paths ---
    model_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}.pkl')
    features_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_features.json')
    params_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_best_params.json') # Model HPs

    # --- Load artifacts if strategy_params were not passed in ---
    if strategy_params is None:
        strategy_params = _load_strategy_params(ticker, strategy_type)
        if not strategy_params:
            return # Error logged in helper

        # Hyperparameters are part of the model, but we can load them for reference
        if os.path.exists(params_filename):
            with open(params_filename, 'r') as f:
                params = json.load(f)
            logger.info(f"Confirming model was trained with hyperparameters: {params}")
    
    strategy_class = load_strategy_class(strategy_type)
    if not strategy_class:
        logger.error(f"Cannot perform inference: Strategy class for '{strategy_type}' not found.")
        return None
    strategy_instance = strategy_class(params=strategy_params)
    is_ml = strategy_instance.is_ml_strategy
    model = None
    features = []

    try:
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        # logger.info(f"Loaded model from {model_filename}")

        with open(features_filename, 'r') as f:
            features = json.load(f)
        # logger.info(f"Loaded {len(features)} features from {features_filename}")

    except FileNotFoundError as e:
        if is_ml:
            # This is not an error if we intend to fall back to a rule-based check.
            # The 'model' variable will remain None, triggering the rule-based logic below.
            logger.warning(f"ML model artifacts not found for {ticker} - {strategy_type}. Will use rule-based check instead.")
        else:
            logger.warning(f"This is a non-ML strategy. Proceeding with rule-based setup check.")
            model = None # Explicitly set model to None for non-ML strategies
            features = []

    # --- Prepare latest data if not provided ---
    if data_df is None:
        start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        try:
            base_df = _prepare_base_data(ticker, start_date, end_date, strategy_params)
            data_df = strategy_instance.prepare_data(data=base_df)
            if data_df.empty:
                logger.error("No data available for inference after preparation.")
                return None
        except Exception as e:
            logger.error(f"Error preparing data for inference: {e}")
            return None

    # --- Perform Inference ---
    if is_ml and model is not None: 
        # --- Inference for ML-based strategies ---
        if not all(f in data_df.columns for f in features):
            raise ValueError(f"Missing features for inference: {set(features) - set(data_df.columns)}")
        last_data_point = data_df[features].iloc[-1:]
        prediction = model.predict(last_data_point)
        prediction_proba = model.predict_proba(last_data_point)
        
        logger.info(f"--- ML Inference Result for {ticker} on {data_df['date'].iloc[-1].date()} ---")
        logger.info(f"Strategy: {strategy_type}, Predicted Class: {prediction[0]}, Prediction Probabilities: {prediction_proba[0]}")
        
        if len(prediction_proba[0]) > 2: # Multiclass
            final_decision = "HOLD"
            if prediction[0] == 1: final_decision = "BUY"
            elif prediction[0] == 2: final_decision = "SELL"
        else: # Binary
            final_decision = "BUY" if prediction[0] == 1 else "HOLD"

        logger.info(f"Final Decision: {final_decision}")
        return {
            "decision": final_decision,
            "prediction": int(prediction[0]),
            "probabilities": prediction_proba[0].tolist()
        }
    else:
        # --- Inference for rule-based strategies ---
        setup_mask = strategy_instance.get_setup_mask(data_df)

        is_setup = False if setup_mask.empty else setup_mask.iloc[-1]
        decision = "BUY" if is_setup else "HOLD"
        probabilities = [0.0, 1.0] if is_setup else [1.0, 0.0]
        
        logger.info(f"--- Rule-Based Inference Result for {ticker} on {data_df['date'].iloc[-1].date()} ---")
        logger.info(f"Strategy: {strategy_type}, Setup Condition Met: {is_setup}, Final Decision: {decision}")
        return {"decision": decision, "probabilities": probabilities}


@click.group()
def cli():
    """CLI for PyBroker Training and Inference."""
    pass

@cli.command()
@click.option('--ticker', '-t', required=True, help='Stock ticker symbol.')
@click.option('--strategy-type', '-s', required=True, type=click.Choice(list(STRATEGY_CLASS_MAP.keys())), help='The type of strategy to train.')
@click.option('--tune/--no-tune', default=True, help='Enable/disable hyperparameter tuning. Default is enabled.')
@click.option('--plot/--no-plot', default=True, help='Plot results immediately after training. Default is disabled.')
@click.option('--start-date', default='2000-01-01', help='Start date for backtest (YYYY-MM-DD).')
@click.option('--end-date', default=None, help='End date for backtest (YYYY-MM-DD). Defaults to yesterday.')
@click.option('--use-tuned-strategy-params/--no-use-tuned-strategy-params', default=True, help='Use best parameters found by tune-strategy.')
@click.option('--override-param', '-o', 'override_params', multiple=True, help='Override a specific strategy parameter (e.g., -o risk_per_trade_pct=0.02).')
@click.option('--commission', default=0.0, help='Commission cost per share (e.g., 0.005).')
def train(ticker, strategy_type, tune, plot, start_date, end_date, use_tuned_strategy_params, override_params, commission):
    """Runs the walk-forward analysis and saves the final model."""
    override_dict = {}
    if override_params:
        for p in override_params:
            if '=' not in p:
                raise click.BadParameter("Override param must be in key=value format.")
            key, value_str = p.split('=', 1)
            try:
                # Attempt to convert to float/int if possible
                value = float(value_str)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                value = value_str # Keep as string if conversion fails
            override_dict[key.strip()] = value

    run_pybroker_walkforward(
        ticker=ticker.upper(), 
        strategy_type=strategy_type,
        start_date=start_date, 
        end_date=end_date, 
        tune_hyperparameters=tune, 
        plot_results=plot, save_assets=True, 
        use_tuned_strategy_params=use_tuned_strategy_params,
        override_params=override_dict,
        commission_cost=commission
    )

@cli.command(name='visualize-model')
@click.option('--ticker', '-t', required=True, help='Stock ticker symbol.')
@click.option('--strategy-type', '-s', required=True, type=click.Choice(list(STRATEGY_CLASS_MAP.keys())), help='The type of strategy to visualize.')
@click.option('--plot/--no-plot', default=True, help='Plot results. Default is enabled.')
def visualize_model(ticker, strategy_type, plot):
    """
    Loads and visualizes the saved walk-forward backtest results for a trained model.
    This command shows the true out-of-sample performance.
    """
    viz_assets = run_visualize_model(ticker.upper(), strategy_type)

    if not viz_assets:
        logger.error("Visualization failed to produce assets.")
        return

    # --- Print metrics and parameters to the console ---
    logger.info(f"\n--- Walk-Forward Performance Metrics for {ticker} - {strategy_type} ---")
    print(viz_assets["metrics_df"].to_string())

    if viz_assets.get("best_strategy_params"):
        logger.info("\n--- Tuned Strategy Parameters ---")
        print(json.dumps(viz_assets["best_strategy_params"], indent=2))
    
    if viz_assets.get("best_model_params"):
        logger.info("\n--- Tuned Model Hyperparameters (from last fold) ---")
        print(json.dumps(viz_assets["best_model_params"], indent=2))

    if plot:
        logger.info("\nDisplaying plots... Close plot windows to continue.")
        # Iterate through the dictionary and show any matplotlib figures
        for fig_name, fig_obj in viz_assets.items():
            if isinstance(fig_obj, plt.Figure):
                plt.show()
              
@cli.command(name='tune-strategy')
@click.option('--ticker', '-t', required=True, help='Stock ticker symbol.')
@click.option('--strategy-type', '-s', required=True, type=click.Choice(list(STRATEGY_CLASS_MAP.keys())), help='The type of strategy to tune.')
@click.option('--n-calls', default=100, help='Number of optimization iterations.')
@click.option('--start-date', default='2000-01-01', help='Start date for tuning data (YYYY-MM-DD).')
@click.option('--end-date', default=None, help='End date for tuning data (YYYY-MM-DD).')
@click.option('--commission', default=0.0, help='Commission cost per share (e.g., 0.005).')
@click.option('--max-drawdown', default=40.0, help='Maximum acceptable drawdown percentage for tuning objective.')
@click.option('--min-trades', default=20, help='Minimum acceptable trade count for tuning objective.')
@click.option('--min-win-rate', default=40.0, help='Minimum acceptable win rate for tuning objective.')
def tune_strategy(ticker, strategy_type, n_calls, start_date, end_date, commission, max_drawdown, min_trades, min_win_rate):
    """Performs Bayesian optimization on strategy-level parameters."""
    logger.info("running tune strategy...")
    run_tune_strategy(ticker.upper(), strategy_type, n_calls, start_date, end_date, commission, max_drawdown, min_trades, min_win_rate, progress_callback=None, stop_event_checker=None)

def run_tune_strategy(ticker, strategy_type, n_calls, start_date, end_date, commission, max_drawdown, min_trades, min_win_rate, progress_callback=None, stop_event_checker=None):
    """Core logic for Bayesian optimization on strategy-level parameters."""
    logger.info(f"--- Starting Strategy-Level Parameter Tuning for {ticker} with {strategy_type} ---")

    if not end_date:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    strategy_class = load_strategy_class(strategy_type)
    search_space = get_strategy_tuning_space(strategy_class) if strategy_class else []
    if not search_space:
        logger.error(f"Strategy tuning is not configured for strategy_type '{strategy_type}'.")
        return
    strategy_params = get_strategy_defaults(strategy_class)

    # --- Load raw price data ONCE before the tuning loop to avoid repeated I/O ---
    try:
        logger.info("Preparing base data for tuning process... (This may take a moment)")
        base_data_df = _prepare_base_data(ticker, start_date, end_date, strategy_params)
        if base_data_df.empty:
            logger.error("Base data preparation failed for tuning. Aborting.")
            return
    except Exception as e:
        logger.error(f"Failed to prepare base data for tuning: {e}")
        traceback.print_exc()

    # --- Variables to store the metrics of the best run found during optimization ---
    best_run_metrics = {}
    best_score = float('inf')

    # Use a nonlocal counter for progress tracking instead of a global variable.
    iteration_count = 0

    def skopt_callback(res):
        """Callback for skopt to check for stop event."""
        if stop_event_checker and stop_event_checker():
            logger.warning("Stop event received. Halting optimization.")
            # Returning True stops the gp_minimize loop.
            return True
        return False

    # 2. Define the objective function to minimize
    def objective(params):
        nonlocal best_run_metrics, best_score, iteration_count # Allow modification of outer scope variables

        # --- NEW: Check for stop event at the beginning of each iteration ---
        # This makes the stop button feel instantaneous, as it will exit at the start
        # of the next iteration rather than waiting for the current long one to finish.
        if stop_event_checker and stop_event_checker():
            logger.warning("Stop event detected at start of objective function. Terminating this run.")
            return float('inf') # Return a high value to penalize this run. The main callback will then terminate the process.

        # --- Dynamically map the list of params from the optimizer to a dictionary ---
        
        # --- Update progress for UI ---
        iteration_count += 1
        if progress_callback:
            progress_callback(iteration_count / n_calls, f"Running iteration {iteration_count}/{n_calls}...")

        # This is more robust than hardcoding indices.
        param_dict = {dim.name: val for dim, val in zip(search_space, params)}
        
        # --- Prepare data with the specific parameters for this trial ---
        try:
            current_strategy_params = get_strategy_defaults(strategy_class)
            current_strategy_params.update(param_dict)
            
            strategy_instance = strategy_class(params=current_strategy_params)
            # This is the crucial fix: run the full data prep on raw data for each trial.
            data_df = strategy_instance.prepare_data(data=base_data_df.copy())
            features = strategy_instance.get_feature_list()
            context_columns_to_register = BASE_CONTEXT_COLUMNS + strategy_instance.get_extra_context_columns_to_register()
            is_ml_strategy = strategy_instance.is_ml_strategy
        except Exception as e:
            logger.error(f"Error during data preparation in objective function: {e}")
            return 1.0 # Return a poor score

        if is_ml_strategy:
            pybroker.register_columns(features)
        pybroker.register_columns(context_columns_to_register) # Always register context columns

        result, quality_scores = run_pybroker_walkforward(
            ticker=ticker.upper(), strategy_type=strategy_type,
            start_date=start_date, end_date=end_date, tune_hyperparameters=False, plot_results=False, save_assets=False,
            override_params=current_strategy_params, disable_inner_parallelism=True, 
            commission_cost=commission,
            preloaded_data_df=data_df, preloaded_features=features,
            calc_bootstrap=False, # Disable bootstrapping to prevent parallel process errors with UI logging
            stop_event_checker=stop_event_checker # Pass the checker down
        )

        if result is None or result.metrics.trade_count < 5:
            logger.warning("Backtest failed or had < 5 trades. Assigning poor score.")
            return 1.0

        # --- Objective Function Calculation ---
        # This function defines what we want to optimize. It uses hard constraints
        # and a clear primary objective.
        metrics = result.metrics
        penalty = 1000.0

        # Check all hard constraints first. If any are violated, return a large penalty score.
        if metrics.max_drawdown > max_drawdown:
            logger.warning(f"Constraint VIOLATED: Max Drawdown {metrics.max_drawdown:.2f}% > {max_drawdown:.2f}%")
            return penalty + (metrics.max_drawdown - max_drawdown)
        if metrics.trade_count < min_trades:
            logger.warning(f"Constraint VIOLATED: Trade Count {metrics.trade_count} < {min_trades}")
            return penalty + (min_trades - metrics.trade_count)
        if metrics.win_rate < min_win_rate:
            logger.warning(f"Constraint VIOLATED: Win Rate {metrics.win_rate:.2f}% < {min_win_rate:.2f}%")
            return penalty + (min_win_rate - metrics.win_rate)

        # If all constraints are met, the objective is to maximize (Sortino * Profit Factor).
        # Since the optimizer minimizes, we return the negative of this product.
        objective_value = (metrics.sortino or 0) * (metrics.profit_factor or 0)
        score = -objective_value
        
        logger.info(f"--> Params: {param_dict} | Sortino: {metrics.sortino:.4f} | Profit Factor: {metrics.profit_factor:.4f} | Score: {score:.4f}")

        # Add penalty for models with low predictive power (AUC), guiding the optimizer toward more reliable strategies
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # --- FIX: Only apply model quality penalty to ML strategies ---
        if is_ml_strategy and avg_quality < 0.52:
            # If average model quality is below random chance, penalize heavily.
            quality_penalty = (0.52 - avg_quality) * 10  # Penalty increases as quality decreases.
            score += quality_penalty
            logger.info(f"--> POOR MODEL QUALITY (Avg AUC: {avg_quality:.3f}). Applying penalty: {quality_penalty:.4f}. Final Score: {score:.4f}")

        # --- If this is the best score so far, save its metrics for logging later ---
        if score < best_score:
            best_score = score
            best_run_metrics = asdict(result.metrics)

        return score

    # 3. Run the Bayesian optimization
    opt_result = gp_minimize(
        func=objective, 
        dimensions=search_space, 
        n_calls=n_calls, random_state=42, n_jobs=1 if progress_callback else -1,
        callback=skopt_callback if stop_event_checker else None)

    # Check if the process was stopped prematurely
    if stop_event_checker and stop_event_checker():
        logger.info("Tuning process was stopped by the user. Aborting without saving results.")
        return

    # 4. Print the best results
    best_params_dict = {dim.name: val for dim, val in zip(search_space, opt_result.x)}
    logger.info("\n--- Strategy Tuning Complete ---")
    logger.info(f"Best parameters found: ")
    for name, value in best_params_dict.items():
        logger.info(f"  - {name}: {value:.4f}" if isinstance(value, float) else f"  - {name}: {value}")
    logger.info(f"Best score achieved during tuning: {opt_result.fun:.4f}")

    # --- Log the full metrics that corresponded to the best score ---
    if best_run_metrics:
        logger.info("--- Metrics for the Best Run Found During Tuning (Strategy Rules + Default Model) ---")
        # Pretty print the dictionary
        for key, value in best_run_metrics.items():
            if isinstance(value, (float, Decimal)):
                logger.info(f"  {key}: {float(value):.4f}")
            else:
                logger.info(f"  {key}: {value}")
        logger.info("------------------------------------------------------------------------------------")

    # --- Save the best parameters to a file ---
    model_dir = os.path.join('pybroker_trainer', 'artifacts')
    os.makedirs(model_dir, exist_ok=True)
    tuned_params_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_best_strategy_params.json')
    with open(tuned_params_filename, 'w') as f:
        json.dump(best_params_dict, f, indent=4, cls=NumpyEncoder)
    logger.info(f"Saved best strategy parameters to {tuned_params_filename}")

@cli.command(name='predict')
@click.option('--ticker', '-t', required=True, help='Stock ticker symbol.')
@click.option('--strategy-type', '-s', required=True, type=click.Choice(list(STRATEGY_CLASS_MAP.keys())), help='The strategy to use for prediction.')
def predict(ticker, strategy_type):
    """
    Runs inference on the latest data using one or more saved models.
    If multiple strategies are provided, it runs a portfolio inference.
    """
    result = infer(ticker=ticker.upper(), strategy_type=strategy_type)
    if result:
        print(f"Inference Result: {json.dumps(result, indent=2)}")

def run_pybroker_full_backtest(ticker: str = 'SPY', start_date: str = '2000-01-01', end_date: str = '2024-12-31', strategy_type: str = 'trend_following', commission_cost: float = 0.0) -> Optional[dict]:
    """
    Runs a full, in-sample backtest using a pre-trained model saved by the 'train' command.
    This function loads the saved model and its associated parameters, then runs a backtest
    over the specified historical period.
    Returns a dictionary of backtest artifacts (result, features, model).
    """
    features: list[str] = []
    context_columns_to_register: list[str] = []
    model_dir = os.path.join('pybroker_trainer', 'artifacts')
    model = None
    strategy_params: dict = {}

    try:
        # --- Step 1: Load strategy class and determine its type (ML vs. non-ML) ---
        strategy_class = load_strategy_class(strategy_type)
        if not strategy_class:
            logger.error(f"Could not load strategy class for {strategy_type}. Aborting.")
            return None
        
        # Load the best available strategy parameters. This prioritizes tuned params.
        strategy_params = _load_strategy_params(ticker, strategy_type)
        if not strategy_params:
            logger.error(f"Could not load strategy parameters for {ticker} - {strategy_type}. Aborting backtest.")
            return None

        strategy_instance = strategy_class(params=strategy_params) # type: ignore
        is_ml = strategy_instance.is_ml_strategy
        
        # --- Step 2: If it's an ML strategy, load all saved model artifacts ---
        if is_ml:
            try:
                model_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}.pkl')
                with open(model_filename, 'rb') as f: model = pickle.load(f)

                features_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_features.json')
                with open(features_filename, 'r') as f: features = json.load(f)
                
                logger.info(f"Loaded model and artifacts for {ticker} {strategy_type}")

            except FileNotFoundError as e:
                logger.error(f"Error loading model artifacts for ML strategy: {e}. Please run 'train' command first.")
                return None
        else:
            logger.info(f"Running backtest for non-ML strategy: {strategy_type}")
            features = strategy_instance.get_feature_list() # Should be empty, but get it anyway

        # --- Step 3: Prepare data for the full backtest range ---
        base_df = _prepare_base_data(ticker, start_date, end_date, strategy_params)
        data_df = strategy_instance.prepare_data(data=base_df)
        context_columns_to_register = BASE_CONTEXT_COLUMNS + strategy_instance.get_extra_context_columns_to_register()
        
        if data_df.empty:
            logger.error(f"No data available for {ticker} in the specified range. Aborting.")
            return

        # --- Step 4: Configure pybroker ---
        if is_ml:
            pybroker.register_columns(features)
        pybroker.register_columns(context_columns_to_register)
        pybroker.disable_logging()

        model_name = 'in_sample_model'
        model_source = None
        if is_ml:
            model_bundle = {'model': model, 'features': features, 'scaler': None}
            def train_fn_dummy(symbol, train_data, test_data, **kwargs):
                return model_bundle
            def model_input_data_fn(data): 
                return data[features]
            model_source = pybroker.model(name=model_name, fn=train_fn_dummy, predict_fn=custom_predict_fn, input_data_fn=model_input_data_fn, pretrained=True)

        # --- Step 5: Configure and run the backtest ---
        strategy_config = StrategyConfig(
            position_mode=PositionMode.LONG_ONLY, 
            exit_on_last_bar=True,
            fee_mode=pybroker.FeeMode.PER_SHARE if commission_cost > 0 else None,
            fee_amount=commission_cost
        )
        
        trader = strategy_instance.get_trader(model_name if is_ml else None, {ticker: strategy_params})
        strategy = Strategy(data_source=data_df, start_date=start_date, end_date=end_date, config=strategy_config)
        if is_ml:
            strategy.add_execution(trader.execute, [ticker], models=[model_source])
        else:
            strategy.add_execution(trader.execute, [ticker])

        logger.info("Starting backtest with saved model...")
        if len(data_df) < 4:
            logger.error(f"Not enough data ({len(data_df)} bars) to run a backtest. Minimum 4 required.")
            return None
        
        # Use a minimal train_size to force pybroker to call the train_fn,
        # which loads our pre-trained model. This is a workaround for pybroker's
        # internal check that skips model loading if train_data is empty.
        min_train_size = 2 / (len(data_df) - 2) if len(data_df) > 2 else 0.99
        result = strategy.walkforward(windows=1, train_size=min_train_size)

        # --- Step 6: Display results ---
        return {
            'result': result,
            'features': features,
            'model': model
        }

    except Exception as e:
        logger.error(f"An error occurred during the full backtest for {ticker}: {e}")
        traceback.print_exc()
        return None
    finally:
        # Unregister columns to clean up global scope
        if features:
            pybroker.unregister_columns(features)
        if context_columns_to_register:
            pybroker.unregister_columns(context_columns_to_register)
    return None

def run_pybroker_portfolio_backtest(tickers: list[str], strategy_type: str, start_date: str, end_date: str, plot_results: bool = True, use_tuned_strategy_params: bool = False, max_open_positions: int = 5, commission_cost: float = 0.0):
    """
    Runs a single walk-forward backtest on a portfolio of tickers.
    This is used to validate scanning-based strategies like RSI Divergence.
    """
    all_data_dfs = []
    features = []
    context_columns_to_register = []
    params_map: Dict[str, dict] = {}

    try:
        logger.info(f"--- Preparing data for portfolio backtest across {len(tickers)} tickers... ---")
        strategy_class = load_strategy_class(strategy_type)
        if not strategy_class:
            logger.error(f"Could not load strategy class for {strategy_type}. Aborting.")
            return None
        is_ml = strategy_class(params={}).is_ml_strategy

        for ticker in tickers:
            try:
                # --- Load tuned parameters for each ticker if requested ---
                current_strategy_params = get_strategy_defaults(strategy_class) if strategy_class else {}

                if use_tuned_strategy_params:
                    model_dir = os.path.join('pybroker_trainer', 'artifacts')
                    tuned_params_filename = os.path.join(model_dir, f'{ticker}_{strategy_type}_best_strategy_params.json')
                    if os.path.exists(tuned_params_filename):
                        with open(tuned_params_filename, 'r') as f:
                            tuned_params = json.load(f)
                        logger.info(f"Loaded tuned strategy parameters for {ticker}.")
                        current_strategy_params.update(tuned_params)
                    else:
                        logger.warning(f"Tuned params file not found for {ticker}. Using defaults.")
                
                params_map[ticker] = current_strategy_params
                strategy_instance = strategy_class(params=current_strategy_params)
                base_df = _prepare_base_data(ticker, start_date, end_date, current_strategy_params)
                data_df = strategy_instance.prepare_data(data=base_df)
                if not data_df.empty:
                    all_data_dfs.append(data_df)
                    if not features:  # Capture feature list from the first successful ticker
                        features = strategy_instance.get_feature_list()
                else:
                    logger.warning(f"No data for {ticker}, skipping.")
            except Exception as e:
                logger.error(f"Failed to prepare data for {ticker}: {e}")

        if not all_data_dfs:
            logger.error("No data could be prepared for any tickers. Aborting portfolio backtest.")
            return None

        portfolio_df = pd.concat(all_data_dfs, ignore_index=True)
        logger.info(f"Combined portfolio data shape: {portfolio_df.shape}")

        # --- Setup PyBroker for Portfolio Backtest ---
        pybroker.register_columns(features)
        context_columns_to_register = BASE_CONTEXT_COLUMNS + strategy_instance.get_extra_context_columns_to_register()
        pybroker.register_columns(context_columns_to_register)

        def model_input_data_fn(data):
            return data[features]

        # The train_fn will be called by pybroker for each symbol in the portfolio.
        # We can reuse the same logic as the single-ticker backtest.
        def train_fn(symbol, train_data, test_data, **kwargs):
            # This function is a placeholder for ML strategies.
            # For non-ML, it won't be used, but pybroker requires it to be defined.
            train_start = train_data['date'].min().date() if not train_data.empty else 'N/A'
            train_end = train_data['date'].max().date() if not train_data.empty else 'N/A'
            logger.info(f"[{symbol}] Training fold: {train_start} to {train_end}. Initial samples: {len(train_data)}")

            pybroker.disable_logging()
            if train_data.empty:
                logger.warning(f"[{symbol}] Training data is empty for this fold. This is expected if the stock did not exist for the full period. Returning untrained model.")
                model_config = strategy_instance.get_model_config()
                model = LGBMClassifier(random_state=42, n_jobs=1, **model_config)
                return {'model': model, 'features': features}
            
            # --- FIX: Drop rows where the target is NaN before training ---
            # This is crucial for strategies like RSI Divergence that only label specific setup days.
            if 'target' in train_data.columns:
                train_data = train_data.dropna(subset=['target'])

            # --- NEW: More robust check for minimum samples per fold ---
            min_total_samples = 30  # Increased minimum total setups required for training
            min_class_samples = 10  # Minimum required setups for EACH class (win and loss)

            if train_data.empty or len(train_data) < min_total_samples:
                logger.warning(f"[{symbol}] Training data is empty or has insufficient samples ({len(train_data)} < {min_total_samples}) for portfolio fold. Returning untrained model.")
                model_config = strategy_instance.get_model_config()
                model = LGBMClassifier(random_state=42, n_jobs=1, **model_config)
                return {'model': model, 'features': features}
            
            if 'target' in train_data.columns and not train_data['target'].value_counts().empty:
                if len(train_data['target'].unique()) < 2 or train_data['target'].value_counts().min() < min_class_samples:
                    logger.warning(f"[{symbol}] Minority class has too few samples ({train_data['target'].value_counts().min()} < {min_class_samples}) for portfolio fold. Returning untrained model.")
                    model_config = strategy_instance.get_model_config()
                    model = LGBMClassifier(random_state=42, n_jobs=1, **model_config)
                    return {'model': model, 'features': features}

            logger.info(f"[{symbol}] Training model on {len(train_data)} valid setup samples...")
            model = LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced', **strategy_instance.get_model_config())
            model.fit(train_data[features], train_data['target'].astype(int))
            return {'model': model, 'features': features}

        model_name = f"{strategy_type}_portfolio_model"
        model_source = pybroker.model(name=model_name, fn=train_fn, predict_fn=custom_predict_fn, input_data_fn=model_input_data_fn)

        strategy_config = StrategyConfig(
            position_mode=PositionMode.LONG_ONLY, 
            exit_on_last_bar=True,
            max_long_positions=max_open_positions,
            fee_mode=pybroker.FeeMode.PER_SHARE if commission_cost > 0 else None,
            fee_amount=commission_cost
        )
        strategy = ExpandingWindowStrategy(data_source=portfolio_df, start_date=start_date, end_date=end_date, config=strategy_config)
        
        trader = strategy_instance.get_trader(model_name if is_ml else None, params_map)
        # Add a single execution for all tickers. PyBroker will handle the rest.
        if is_ml:
            strategy.add_execution(trader.execute, tickers, models=[model_source])
        else:
            strategy.add_execution(trader.execute, tickers)

        # --- Run the Walk-Forward Analysis ---
        logger.info("Starting portfolio walk-forward analysis...")
        total_years = (portfolio_df['date'].max() - portfolio_df['date'].min()).days / 365.25
        windows = 4 if total_years >= 20 else 2 if total_years >= 10 else 1
        
        result = strategy.walkforward(windows=windows, train_size=0.7, lookahead=1, calc_bootstrap=True)

        logger.info(f"\n--- Portfolio Walk-Forward Results for {strategy_type} strategy ---")
        logger.info(result.metrics_df.to_string())
        if plot_results:
            plot_equity_curve(result, f"Portfolio ({strategy_type})")

    except Exception as e:
        logger.error(f"An error occurred during the portfolio backtest: {e}")
        traceback.print_exc()
        return None
    finally:
        if features: pybroker.unregister_columns(features)
        if context_columns_to_register: pybroker.unregister_columns(context_columns_to_register)
    return result

ETFs = ['SPY', 'QQQ', 'IWM', 'DBC', 'GLD', 'USO', 'UNG', 'GBTC', 'ETHE', 
        'XLK', 'XLF', 'XLE', 'XLY', 'XLP', 'XLI', 'XLV', 'XLU', 'XLB', 
        'DLR.TO', 'GLXY.TO', 'XEF.TO', 'XEG.TO', 'XFN.TO', 'XIC.TO', 'XIU.TO', 'XRE.TO', 'ZAG.TO', 'ZEB.TO']

from tools.scanner_tool import get_candidate_companies

def get_default_tickers(source: int = 2, market: str = 'us', min_volume: int = 500000, limit: int = 50):
    if source > 0:
        companies = get_candidate_companies(market=market, min_avg_volume=min_volume, limit=limit)
        symbols = [c.symbol for c in companies]
        if source == 1:
            tickers = symbols
        else:
            tickers = ETFs + symbols
    else:
        tickers = ETFs
    logger.info(f"Candidates to scan: {len(tickers)}")
    return tickers

@cli.command(name='pre-scan-universe')
@click.option('--tickers', '-t', help='Comma-separated list of candidate tickers to scan.')
@click.option('--strategy-type', '-s', required=True, type=click.Choice(list(STRATEGY_CLASS_MAP.keys())), help='The strategy to check for setups.')
@click.option('--min-setups', default=60, help='Minimum number of historical setups required for a ticker to be included.')
@click.option('--start-date', default='2000-01-01', help='Start date for historical data.')
@click.option('--end-date', default=None, help='End date for historical data (defaults to yesterday).')
def pre_scan_universe(tickers, strategy_type, min_setups, start_date, end_date):
    """Scans tickers for a minimum number of historical setups."""
    valid_tickers, _ = run_pre_scan_universe(tickers, strategy_type, min_setups, start_date, end_date, stop_event_checker=None)
    if valid_tickers:
        logger.info("Use the following string for the --tickers option in other commands:")
        print(",".join(valid_tickers))

def run_pre_scan_universe(tickers, strategy_type, min_setups, start_date, end_date, progress_callback=None, stop_event_checker=None):
    """Core logic to scan tickers for a minimum number of historical setups."""
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
    else:
        ticker_list = get_default_tickers(limit=100)
    
    if not start_date:
        start_date = '2000-01-01'
    if not end_date:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    logger.info(f"Scanning {len(ticker_list)} tickers for a minimum of {min_setups} '{strategy_type}' setups...")

    strategy_class = load_strategy_class(strategy_type)
    if not strategy_class:
        logger.error(f"Could not load strategy class for {strategy_type}. Aborting.")
        return None

    ticker_setup_count = {}
    for i, ticker in enumerate(ticker_list):
        if stop_event_checker and stop_event_checker():
            logger.warning("Stop event detected during pre-scan. Aborting.")
            break

        if progress_callback:
            progress_callback(i / len(ticker_list), f"Scanning {ticker} ({i+1}/{len(ticker_list)})")

        try:
            ticker_setup_count[ticker] = 0
            strategy_params = get_strategy_defaults(strategy_class)

            # Instantiate the strategy to call its get_setup_mask method
            strategy_instance = strategy_class(params=strategy_params)
            base_df = _prepare_base_data(ticker, start_date, end_date, strategy_params)
            data_df = strategy_instance.prepare_data(data=base_df)
            if data_df.empty:
                logger.warning(f"[{ticker}] No data available. Skipping.")
                continue
            
            # Count the number of non-NaN values in the 'target' column.
            # This represents the number of setups that could actually be labeled and used for training.
            setup_count = data_df['target'].notna().sum()
            ticker_setup_count[ticker] = setup_count

        except Exception as e:
            logger.error(f"Error processing {ticker} during pre-scan: {e}")

    if progress_callback:
        progress_callback(1.0, "Pre-scan complete.")

    logger.info("\n--- Pre-scan Complete ---")
    print(ticker_setup_count)
    valid_tickers = [ticker for ticker, count in ticker_setup_count.items() if count >= min_setups]
    if valid_tickers:
        logger.info(f"Found {len(valid_tickers)} valid tickers out of {len(ticker_list)}.")
    else:
        logger.warning("No tickers met the minimum setup criteria.")
    return valid_tickers, ticker_setup_count

@cli.command(name='portfolio-backtest')
@click.option('--strategy-type', '-s', required=True, type=click.Choice(list(STRATEGY_CLASS_MAP.keys())), help='The type of strategy to backtest on a portfolio.')
@click.option('--tickers', '-t', required=True, help='Comma-separated list of tickers for the portfolio.')
@click.option('--start-date', default='2000-01-01', help='Start date for backtest (YYYY-MM-DD).')
@click.option('--end-date', default=None, help='End date for backtest (YYYY-MM-DD).')
@click.option('--max-open-positions', default=5, help='Maximum number of concurrent long positions in the portfolio.')
@click.option('--plot/--no-plot', default=True, help='Plot results. Default is enabled.')
@click.option('--use-tuned-strategy-params/--no-use-tuned-strategy-params', default=False, help='Use best parameters found by tune-strategy for each stock.')
@click.option('--commission', default=0.0, help='Commission cost per share (e.g., 0.005).')
def portfolio_backtest(strategy_type, tickers, start_date, end_date, max_open_positions, plot, use_tuned_strategy_params, commission):
    """Runs a portfolio-level backtest to validate a scanning strategy."""
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
    else:
        ticker_list = get_default_tickers()

    if not start_date:
        start_date = '2000-01-01'
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    run_pybroker_portfolio_backtest(ticker_list, strategy_type, start_date, end_date, plot, use_tuned_strategy_params, max_open_positions, commission)

def run_scan(ticker_list: list[str], strategy_list: list[str], progress_callback=None, stop_event_checker=None) -> list[dict]:
    """
    Scans a list of tickers for stocks showing bullish setups.
    This is the core logic, callable from other scripts.
    """
    found_signals = []
    total_items = len(ticker_list)

    for i, ticker in enumerate(ticker_list):
        if stop_event_checker and stop_event_checker():
            logger.warning("Stop event detected during scan. Aborting.")
            break

        if progress_callback:
            progress_callback(i / total_items, f"Scanning {ticker} ({i+1}/{total_items})")

        try:
            # --- OPTIMIZATION: Prepare base data ONCE per ticker ---
            start_date_str = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')
            end_date_str = datetime.now().strftime('%Y-%m-%d')
            base_df = _prepare_base_data(ticker, start_date_str, end_date_str, {'include_earnings_dates': True})
            
            if base_df.empty:
                logger.warning(f"Could not load base data for {ticker}. Skipping.")
                continue

            for strategy_type in strategy_list:
                try:
                    # --- Load best available params for the specific strategy ---
                    strat_params = _load_strategy_params(ticker, strategy_type)
                    if not strat_params:
                        continue # Error logged in helper
                
                    strategy_class = load_strategy_class(strategy_type) # type: ignore
                    strategy_instance = strategy_class(params=strat_params)
                    # Use a copy of base_df to prevent cross-contamination of indicators between strategies
                    data_df = strategy_instance.prepare_data(data=base_df.copy())
                    if data_df.empty:
                        logger.warning(f"[{ticker}] No data after preparation for {strategy_type}. Skipping.")
                        continue

                    result = infer(ticker, strategy_type, data_df=data_df, strategy_params=strat_params)

                    if result and result.get('decision') == "BUY":
                        found_signals.append({
                            "ticker": ticker,
                            "strategy": strategy_type,
                            "date": data_df['date'].iloc[-1].strftime('%Y-%m-%d'),
                            "close": data_df['close'].iloc[-1],
                            "probabilities": result.get('probabilities') or result.get('primary_probabilities')
                        })
                except Exception as strat_e:
                    logger.error(f"Error scanning {ticker} with strategy {strategy_type}: {strat_e}")
                    traceback.print_exc()

        except Exception as e:
            logger.error(f"Error scanning {ticker}: {e}")
            continue

    # --- Save results to a JSON file for the Streamlit app ---
    scan_results_file = os.path.join(WORKING_DIRECTORY, 'scan_results.json')
    if found_signals:
        logger.info("\n--- Scan Results ---")
        for signal in found_signals:
            logger.info(f"Ticker: {signal['ticker']}, Strategy: {signal['strategy']}, Date: {signal['date']}, Close: {signal['close']:.2f}, Probs: {signal.get('probabilities')}")
        
        try:
            with open(scan_results_file, 'w') as f:
                json.dump(found_signals, f, indent=4)
            logger.info(f"Scan results saved to {scan_results_file}")
        except Exception as e:
            logger.error(f"Failed to save scan results to JSON: {e}")
    else:
        logger.info("\n--- No signals found during scan. ---")
        # --- Create an empty file if no signals are found to prevent stale data in UI ---
        with open(scan_results_file, 'w') as f:
            json.dump([], f)
        logger.info(f"Cleared/created empty scan results file at {scan_results_file}")

    if progress_callback:
        progress_callback(1.0, "Scan complete.")
        
    return found_signals

@cli.command()
@click.option('--tickers', '-t', help='Comma-separated list of tickers to scan. Defaults to a list of liquid stocks/ETFs.')
@click.option('--strategies', '-s', default=','.join(list(STRATEGY_CLASS_MAP.keys())), help='Comma-separated list of strategies to process. Defaults to all available strategies.')
def scan(tickers, strategies):
    """
    Scans a list of tickers for stocks showing bullish setups.
    Saves results to scan_results.json for the dashboard.
    """
    logger.info(f"--- Scanning for setups ---")
    ticker_list = [t.strip().upper() for t in tickers.split(',')] if tickers else get_default_tickers(limit=100)
    strategy_list = [s.strip() for s in strategies.split(',')]
    
    run_scan(ticker_list, strategy_list, stop_event_checker=None)

def _process_batch_job(ticker, strat_type, n_calls, start_date, end_date, commission, max_drawdown, min_trades, min_win_rate, stop_event_checker=None):
    """Helper function to process a single tune-and-train job and return its metrics."""
    try:
        if stop_event_checker and stop_event_checker():
            logger.info(f"Stop event detected before processing {ticker} - {strat_type}. Skipping job.")
            return "STOPPED"

        # First, tune strategy parameters
        logger.info(f"Tuning strategy parameters for {ticker} - {strat_type}...")
        run_tune_strategy(
            ticker=ticker, strategy_type=strat_type, n_calls=n_calls,
            start_date=start_date, end_date=end_date, commission=commission,
            max_drawdown=max_drawdown, min_trades=min_trades, min_win_rate=min_win_rate,
            stop_event_checker=stop_event_checker
        )

        # Then, train the model with the best strategy parameters
        # Check for stop event again before starting the next long process
        logger.info(f"Training model for {ticker} - {strat_type} with tuned strategy parameters...")
        result, _ = run_pybroker_walkforward(
            ticker=ticker, strategy_type=strat_type, start_date=start_date, end_date=end_date,
            tune_hyperparameters=True, plot_results=False, save_assets=True,
            use_tuned_strategy_params=True, commission_cost=commission,
            stop_event_checker=stop_event_checker
        )
        
        if stop_event_checker and stop_event_checker():
            return "STOPPED"

        if result:
            logger.info(f"Successfully trained and tuned {ticker} for {strat_type}.")
            return asdict(result.metrics)
        else:
            logger.warning(f"Training failed for {ticker} - {strat_type}. No result to log.")
            return None

    except Exception as e:
        logger.error(f"Error processing {ticker} with {strat_type}: {e}")
        traceback.print_exc()
        return None

def run_batch_train(tickers, strategies, n_calls, start_date, end_date, commission, max_drawdown, min_trades, min_win_rate, min_setups, stop_event_checker=None):
    """Core logic for running a batch of tuning and training jobs."""
    logger.info(f"--- Starting Batch Training Process ---")

    if not end_date:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    strategy_list = [s.strip() for s in strategies.split(',')]
    
    # --- Step 1: Determine the list of jobs (ticker, strategy) to run ---
    jobs_to_run = []
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        for ticker in ticker_list:
            for strat_type in strategy_list:
                jobs_to_run.append((ticker, strat_type))
    else:
        logger.info("No tickers provided. Discovering tickers via pre-scan...")
        for strat_type in strategy_list:
            logger.info(f"Pre-scanning for {strat_type} setups...")
            valid_tickers_for_strat, _ = run_pre_scan_universe(None, strat_type, min_setups, start_date, end_date)
            for ticker in valid_tickers_for_strat:
                jobs_to_run.append((ticker, strat_type))
    
    if not jobs_to_run:
        logger.error("No valid ticker/strategy pairs found to process. Aborting.")
        return

    logger.info(f"Found {len(jobs_to_run)} total jobs to process.")

    results_log_file = os.path.join(WORKING_DIRECTORY, 'batch_train_summary.csv')

    # --- NEW: Check for already completed jobs to make the process resumable ---
    completed_jobs = set()
    if os.path.exists(results_log_file):
        try:
            log_df = pd.read_csv(results_log_file)
            if 'ticker' in log_df.columns and 'strategy_type' in log_df.columns:
                completed_jobs = set(zip(log_df['ticker'], log_df['strategy_type']))
                logger.info(f"Found {len(completed_jobs)} completed jobs in {results_log_file}. These will be skipped.")
        except Exception as e:
            logger.warning(f"Could not read or parse existing log file {results_log_file}. Will not skip any jobs. Error: {e}")

    # --- Step 2: Process jobs and log results to CSV ---
    write_header = not os.path.exists(results_log_file) or os.path.getsize(results_log_file) == 0

    with open(results_log_file, 'a', newline='') as f:
        writer = None
        for i, (ticker, strat_type) in enumerate(jobs_to_run):
            if stop_event_checker and stop_event_checker():
                logger.warning("Stop event received. Halting batch train process.")
                break

            # --- NEW: Skip job if already completed ---
            if (ticker, strat_type) in completed_jobs:
                logger.info(f"--- Skipping Job {i+1}/{len(jobs_to_run)}: {ticker} with {strat_type} (already completed) ---")
                continue

            logger.info(f"\n--- Processing Job {i+1}/{len(jobs_to_run)}: {ticker} with {strat_type} ---")
            
            metrics = _process_batch_job(
                ticker, strat_type, n_calls, start_date, end_date, 
                commission, max_drawdown, min_trades, min_win_rate,
                stop_event_checker=stop_event_checker
            )

            if metrics == "STOPPED":
                logger.warning("Batch process halted by stop event inside a job.")
                break

            if metrics:
                # Add job identifiers to the metrics dict
                metrics['ticker'] = ticker
                metrics['strategy_type'] = strat_type
                metrics['run_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                if writer is None:
                    # Use the keys from the first successful result to write the header
                    fieldnames = list(metrics.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                
                # Ensure all keys are present, fill with None if not
                row_to_write = {field: metrics.get(field) for field in writer.fieldnames}
                writer.writerow(row_to_write)
                f.flush() # Ensure data is written immediately

    logger.info(f"\n--- Batch Training Process Complete ---")
    logger.info(f"Summary of all runs saved to: {results_log_file}")

@cli.command(name='batch-train')
@click.option('--tickers', '-t', help='Comma-separated list of tickers to process. If omitted, tickers will be discovered via pre-scan.')
@click.option('--strategies', '-s', default=','.join(list(STRATEGY_CLASS_MAP.keys())), help='Comma-separated list of strategies to process. Defaults to all available strategies.')
@click.option('--n-calls', default=100, help='Number of tuning iterations per job.')
@click.option('--start-date', default='2000-01-01', help='Start date for training data.')
@click.option('--end-date', default=None, help='End date for training data (defaults to yesterday).')
@click.option('--commission', default=0.005, help='Commission cost per share (e.g., 0.005).')
@click.option('--max-drawdown', default=40.0, help='Maximum acceptable drawdown percentage for tuning objective.')
@click.option('--min-trades', default=20, help='Minimum acceptable trade count for tuning objective.')
@click.option('--min-win-rate', default=40.0, help='Minimum acceptable win rate for tuning objective.')
@click.option('--min-setups', default=60, help='Minimum historical setups required for a ticker to be included if pre-scanning.')
def batch_train(tickers, strategies, n_calls, start_date, end_date, commission, max_drawdown, min_trades, min_win_rate, min_setups):
    """
    Runs a batch of tuning and training jobs and logs the results to a CSV.
    If --tickers is not provided, it will pre-scan to find suitable tickers for each strategy.
    """
    run_batch_train(tickers, strategies, n_calls, start_date, end_date, commission, max_drawdown, min_trades, min_win_rate, min_setups, stop_event_checker=None)

if __name__ == '__main__':
    cli()

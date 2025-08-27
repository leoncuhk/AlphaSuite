"""
Command-line script and library for downloading, updating, and processing stock market data.

This module provides functions to:
- Download and update company information and historical price data from Yahoo Finance.
- Run a daily pipeline to update prices and calculate common technical indicators.
- Perform ad-hoc calculations for specific markets or tickers.
- Run scanners to find stocks meeting specific criteria (e.g., strongest stocks in strongest industries).

This script uses a command-line interface with sub-commands for different actions:
`pipeline`, `download`, `calculate`, `scan`, and `refresh`.
"""
import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import traceback
from dotenv import load_dotenv

from core.db import close_database, initialize_database_schema
from core import model # Ensure models are registered with Base
from core.logging_config import setup_logging
from tools.scanner_tool import calculate_and_save_common_values_for_scanner, find_strongest_stocks_in_strongest_industries
from tools.yfinance_tool import load_ticker_data, save_or_update_company_data

# --- Logging Configuration ---
setup_logging('download_data.log')
logger = logging.getLogger(__name__)

# --- Forcing real-time logging to stdout for subprocesses ---
# When this script is run as a subprocess for the UI, stdout is block-buffered.
# To ensure logs appear in the UI in real-time, we must explicitly flush the
# stream after each log message. We achieve this by replacing the default
# StreamHandler with one that automatically flushes.

class _FlushingStreamHandler(logging.StreamHandler):
    """A stream handler that flushes the stream after each record is emitted."""
    def emit(self, record):
        super().emit(record)
        self.flush()

# Ensure there is always a flushing handler for stdout. This is crucial for
# streaming logs to the UI when this script is run as a subprocess.
root_logger = logging.getLogger()
stdout_handler_found = False
for handler in list(root_logger.handlers):
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        # If a stdout handler already exists, replace it with our flushing version.
        formatter = handler.formatter
        level = handler.level
        root_logger.removeHandler(handler)
        flushing_handler = _FlushingStreamHandler(sys.stdout)
        flushing_handler.setFormatter(formatter)
        flushing_handler.setLevel(level)
        root_logger.addHandler(flushing_handler)
        stdout_handler_found = True
        break

if not stdout_handler_found:
    # If no stdout handler was found, add one. This is common when the initial
    # config omits a console logger if it detects a non-interactive session.
    flushing_handler = _FlushingStreamHandler(sys.stdout)
    flushing_handler.setFormatter(logging.Formatter('%(message)s'))
    flushing_handler.setLevel(logging.INFO)
    root_logger.addHandler(flushing_handler)

def initialize_database():
    """Initializes the database by calling the central setup function."""
    # The core logic is now centralized in core.db for consistency.
    initialize_database_schema()

def run_daily_pipeline():
    """
    Runs the full daily pipeline:
    1. Updates prices for CA and US markets.
    2. Recalculates common scanner values.
    """
    logger.info("--- Starting Daily Data & Scan Pipeline ---")

    markets = ['ca', 'us']

    for market in markets:
        logger.info(f"--- Processing {market.upper()} Market ---")

        logger.info(f"Updating {market.upper()} market prices...")
        save_or_update_company_data(market=market, existing_tickers_action='only', update_prices_action='last_day', batch_size=1000)
        
        logger.info(f"Calculating {market.upper()} common values...")
        calculate_and_save_common_values_for_scanner(market=market)
    logger.info("--- Daily Data & Scan Pipeline Finished ---")

def run_full_download(market, exchange, quote_types, ticker_file, batch_size, start_date, end_date, existing_tickers_action, update_prices_action):
    """Wraps the main save_or_update_company_data function to be callable."""
    logger.info(f"--- Starting Full Download/Update for market: {market} ---")
    # The main function already has extensive logging, so we just call it.
    save_or_update_company_data(
        market=market, exchange=exchange, quote_types=quote_types,
        ticker_file=ticker_file, batch_size=batch_size,
        start_date=start_date, end_date=end_date,
        existing_tickers_action=existing_tickers_action,
        update_prices_action=update_prices_action
    )
    logger.info("--- Full Download/Update Finished ---")

def run_range_download(market, exchange, quote_types, ticker_file, batch_size, start_date, end_date, existing_tickers_action, update_prices_action):
    """
    Downloads data for a specific date range.
    """
    logger.info(f"--- Starting Range Download for market: {market}, from {start_date} to {end_date} ---")
    # The main function already has extensive logging, so we just call it.
    save_or_update_company_data(
        market=market, exchange=exchange, quote_types=quote_types,
        ticker_file=ticker_file, batch_size=batch_size,
        start_date=start_date, end_date=end_date,
        existing_tickers_action=existing_tickers_action,
        update_prices_action=update_prices_action
    )
    logger.info("--- Range Download Finished ---")

def run_calculation(market, tickers):
    """Wraps the calculate_and_save_common_values_for_scanner function."""
    logger.info(f"--- Starting Calculation for market: {market}, tickers: {tickers or 'All'} ---")
    if tickers:
        calculate_and_save_common_values_for_scanner(market='', tickers=tickers)
    else:
        calculate_and_save_common_values_for_scanner(market=market)
    logger.info("--- Calculation Finished ---")

def run_scanner(scanner_name, market, min_avg_volume):
    """Wraps the various scanner functions."""
    import json

    logger.info(f"--- Running Scanner '{scanner_name}' for market: {market} ---")
    passing_stocks = []
    if scanner_name == 'strongest_industries':
        passing_stocks = find_strongest_stocks_in_strongest_industries(market=market)
    else:
        logger.info(f"Unknown scanner name: {scanner_name}")
        # Print empty result for subprocess parsing
        print(f"SCANNER_RESULT_JSON:{json.dumps([])}")
        return []

    logger.info(f"Found {len(passing_stocks)} passing stocks.")
    logger.info("--- Scanner Finished ---")
    # For subprocess communication, print the final result as a JSON string
    # with a specific prefix that the UI can look for. This allows the UI
    # to capture the structured data from the process's stdout.
    print(f"SCANNER_RESULT_JSON:{json.dumps(passing_stocks)}")
    return passing_stocks

def run_refresh_ticker(ticker):
    """Wraps the load_ticker_data function for a single ticker refresh."""
    logger.info(f"--- Refreshing data for ticker: {ticker} ---")
    result = load_ticker_data(ticker, refresh=True)
    if not result or 'shareprices' not in result or result['shareprices'].empty:
        logger.info(f"No data found or failed to refresh for {ticker}")
    else:
        logger.info(f"Successfully refreshed data for {ticker}. Last date: {result['shareprices'].index[-1].date()}")
    logger.info("--- Ticker Refresh Finished ---")

def add_common_download_args(parser):
    """Adds common arguments used by download commands to a parser."""
    parser.add_argument("--market", type=str, help="The market to download (e.g., 'us', 'ca').", default="us")
    parser.add_argument("--exchange", type=str, help="The specific exchange.", default=None)
    parser.add_argument("--quote_types", type=str, help="Comma-separated list of quote types.", default="EQUITY,ETF")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing.", default=50)

def main():
    """
    Parses command-line arguments and executes the corresponding function.
    This function encapsulates the main execution logic of the script.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Safeguard to prevent running download scripts in demo mode
    if os.getenv("DEMO_MODE", "false").lower() == "true":
        print("DEMO_MODE is active. Data download and manipulation scripts are disabled.")
        print("To run these commands, set DEMO_MODE=false in your .env file.")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Download, update, and process stock market data.",
        formatter_class=argparse.RawTextHelpFormatter # To preserve formatting in help messages
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- 'init-db' command ---
    parser_init_db = subparsers.add_parser('init-db', help='Initialize the database by creating all necessary tables.')
    parser_init_db.set_defaults(func=initialize_database)

    # --- 'pipeline' command ---
    parser_pipeline = subparsers.add_parser('pipeline', help='Run the full daily pipeline (update prices and calculate scanner values).')
    parser_pipeline.set_defaults(func=run_daily_pipeline)

    # --- 'download' command ---
    parser_download = subparsers.add_parser('download', help='Run a full download or update of company and price data.')
    add_common_download_args(parser_download)
    parser_download.add_argument("--ticker_file", type=str, help="File with tickers.", default="yhallsym.json")
    parser_download.add_argument("--start_date", type=str, help="Start date (YYYY-MM-DD).", default="2000-01-01")
    parser_download.add_argument("--end_date", type=str, help="End date (YYYY-MM-DD).", default=None)
    parser_download.add_argument("--existing_tickers_action", type=str, help="Action for existing tickers.", default="skip")
    parser_download.add_argument("--update_prices_action", type=str, help="Action for updating prices.", default='yes')
    parser_download.set_defaults(func=run_full_download)

    # --- 'range_download' command ---
    parser_range_download = subparsers.add_parser('range_download', help='Download data for a specific date range.')
    add_common_download_args(parser_range_download)
    parser_range_download.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD).")
    parser_range_download.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD).")
    parser_range_download.add_argument("--existing_tickers_action", type=str, help="Action for existing tickers.", default="only")
    parser_range_download.add_argument("--update_prices_action", type=str, help="Action for updating prices.", default='only')
    parser_range_download.set_defaults(func=run_range_download)

    # --- 'calculate' command ---
    parser_calc = subparsers.add_parser('calculate', help='Calculate and save common values for the scanner.')
    parser_calc.add_argument("--market", type=str, help="Market to calculate for (e.g., 'us', 'ca').", default="us")
    parser_calc.add_argument("--tickers", type=str, help="Comma-separated list of specific tickers to calculate for.", default=None)
    parser_calc.set_defaults(func=run_calculation)

    # --- 'scan' command ---
    parser_scan = subparsers.add_parser('scan', help='Run a scanner to find stocks meeting criteria.')
    parser_scan.add_argument("--scanner_name", type=str, help="The name of the scanner to run.", default='strongest_industries')
    parser_scan.add_argument("--market", type=str, help="Market to scan (e.g., 'us', 'ca').", default="us")
    parser_scan.add_argument("--min_avg_volume", type=int, help="Minimum average volume for scanners.", default=50000)
    parser_scan.set_defaults(func=run_scanner)

    # --- 'refresh' command ---
    parser_refresh = subparsers.add_parser('refresh', help='Force a re-download of all data for a single ticker.')
    parser_refresh.add_argument("--ticker", type=str, required=True, help="The ticker symbol to refresh.")
    parser_refresh.set_defaults(func=run_refresh_ticker)

    args = parser.parse_args()

    # Create a dictionary of arguments to pass to the function, excluding 'func'
    func_args = vars(args)
    command_func = func_args.pop('func', None)
    # Split comma-separated string arguments into lists
    if 'quote_types' in func_args:
        func_args['quote_types'] = func_args['quote_types'].split(',')
    if 'tickers' in func_args and func_args.get('tickers'):
        func_args['tickers'] = func_args['tickers'].split(',')

    try:
        if command_func:
            # Pass arguments as keyword arguments
            command_func(**func_args)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        close_database()

if __name__ == "__main__":
    main()

"""
Example usages:

Run the full daily pipeline (price updates + calculations):
python download_data.py pipeline

Initial download of a market:
python download_data.py download --market=us

Update company info only (no prices):
python download_data.py download --market=ca --existing_tickers_action=only --update_prices_action=no

Daily update of recent prices:
python download_data.py download --market=us --existing_tickers_action=only --update_prices_action=last_day --batch_size=1000

Daily update of calculated values for a whole market:
python download_data.py calculate --market=us

Download prices for a specific date range:
python download_data.py range_download  --market=us --start_date=2024-01-01 --end_date=2024-01-31 --existing_tickers_action=only --update_prices_action=only --batch_size=1000

Run an individual scanner:
python download_data.py scan --scanner_name=strongest_industries --market=us

Refresh data for a single ticker:
python download_data.py refresh --ticker=AAPL

"""

import os
import streamlit as st
import logging
import logging.handlers
from datetime import datetime
import threading
import queue
import time

from load_cfg import DEMO_MODE
from pybroker_trainer.strategy_loader import get_strategy_class_map
from quant_engine import (
    run_pybroker_walkforward,
    run_tune_strategy,
    run_visualize_model
)

st.set_page_config(page_title="Model Training & Tuning", layout="wide")
st.title("🛠️ Model Training & Tuning")

if DEMO_MODE:
    st.warning(
        "**Demo Mode Active:** All model tuning and training operations are disabled. "
        "To enable these features, set `DEMO_MODE = False` in `load_cfg.py`.",
        icon="🔒"
    )

# --- Add session state and stop checker for long-running processes ---
if 'tuning_in_progress' not in st.session_state:
    st.session_state.tuning_in_progress = False
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'tuning_thread' not in st.session_state:
    st.session_state.tuning_thread = None
if 'training_thread' not in st.session_state:
    st.session_state.training_thread = None
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = None
if 'progress_queue' not in st.session_state:
    st.session_state.progress_queue = None
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None
if 'completion_message_tune' not in st.session_state:
    st.session_state.completion_message_tune = None
if 'completion_message_train' not in st.session_state:
    st.session_state.completion_message_train = None

def run_tuning_in_thread(params, log_q, progress_q, stop_event):
    """Wrapper to run the tuning process in a separate thread and communicate via queues."""
    # Setup QueueHandler to route logs from the thread to the main UI
    qh = logging.handlers.QueueHandler(log_q)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt='%H:%M:%S')
    qh.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(qh)

    def progress_callback(progress, text):
        progress_q.put((progress, text))

    try:
        # The stop_event_checker is a simple lambda that checks the thread-safe Event object.
        run_tune_strategy(
            ticker=params['ticker'],
            strategy_type=params['strategy_type'],
            n_calls=params['n_calls'],
            start_date=params['start_date'].strftime('%Y-%m-%d'),
            end_date=params['end_date'].strftime('%Y-%m-%d'),
            commission=params['commission'],
            max_drawdown=params['max_drawdown'],
            min_trades=params['min_trades'],
            min_win_rate=params['min_win_rate'],
            progress_callback=progress_callback,
            stop_event_checker=lambda: stop_event.is_set()
        )
    finally:
        root_logger.removeHandler(qh)

st.markdown("""
This page provides a user interface for the core backend commands.
- **Tune Strategy:** Use Bayesian Optimization to find the best high-level parameters for a strategy (e.g., stop-loss multipliers, indicator periods).
- **Train Model:** Run a full walk-forward backtest. This process trains and tests a model on different time windows and saves the final trained model and its performance metrics.
""")

strategy_options = list(get_strategy_class_map().keys())

tune_tab, train_tab, visualize_tab = st.tabs(["Tune Strategy", "Train Model", "Visualize Model Performance"])

with tune_tab:
    st.header("Tune Strategy Parameters")

    # Display completion message if it exists from a previous run
    if st.session_state.get('completion_message_tune'):
        message, msg_type = st.session_state.completion_message_tune
        if msg_type == "success":
            st.success(message)
        elif msg_type == "warning":
            st.warning(message)
        st.session_state.completion_message_tune = None # Clear the message

    # Show form only if tuning is not in progress
    if not st.session_state.tuning_in_progress:
        with st.form("tune_form"):
            c1, c2, c3 = st.columns(3)
            ticker_tune = c1.text_input("Ticker", "SPY").upper()
            strategy_type_tune = c2.selectbox("Strategy Type", strategy_options, key="tune_strat")
            n_calls_tune = c3.number_input("Tuning Iterations (n_calls)", min_value=10, max_value=200, value=50)

            c1, c2 = st.columns(2)
            start_date_tune = c1.date_input("Start Date", datetime(2000, 1, 1), key="tune_start")
            end_date_tune = c2.date_input("End Date", datetime.now(), key="tune_end")

            st.subheader("Tuning Constraints")
            c1, c2, c3, c4 = st.columns(4)
            commission_tune = c1.number_input("Commission ($ per share)", value=0.0, format="%.4f")
            max_drawdown_tune = c2.number_input("Max Drawdown (%)", value=40.0)
            min_trades_tune = c3.number_input("Min Trades", value=20)
            min_win_rate_tune = c4.number_input("Min Win Rate (%)", value=40.0)

            run_tuning = st.form_submit_button("Run Tuning", use_container_width=True, disabled=DEMO_MODE)

            if run_tuning:
                st.session_state.tuning_in_progress = True
                # Store params in session state to use them after the rerun
                st.session_state.tune_params = {
                    "ticker": ticker_tune, "strategy_type": strategy_type_tune, "n_calls": n_calls_tune,
                    "start_date": start_date_tune, "end_date": end_date_tune, "commission": commission_tune,
                    "max_drawdown": max_drawdown_tune, "min_trades": min_trades_tune, "min_win_rate": min_win_rate_tune,
                }
                st.rerun()

    # This block runs when tuning_in_progress is True
    if st.session_state.tuning_in_progress:
        # If the thread object doesn't exist, it means we need to start it.
        if st.session_state.tuning_thread is None:
            st.session_state.log_queue = queue.Queue()
            st.session_state.progress_queue = queue.Queue()
            st.session_state.log_messages = [] # Clear old logs
            st.session_state.stop_event = threading.Event()

            thread = threading.Thread(
                target=run_tuning_in_thread,
                args=(st.session_state.tune_params, st.session_state.log_queue, st.session_state.progress_queue, st.session_state.stop_event),
                daemon=True # This is crucial to allow the main app to exit even if the thread is running
            )
            st.session_state.tuning_thread = thread
            thread.start()

        if st.button("Stop Tuning Process", use_container_width=True, type="primary"):
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            st.warning("Stop signal sent. The process will halt after the current iteration.")

        # --- UI Monitoring Section ---
        params = st.session_state.tune_params
        st.info(f"Tuning in progress for {params['ticker']} with {params['strategy_type']}...")
        log_container = st.expander("Tuning Log", expanded=True)
        log_area = log_container.empty()
        progress_bar = st.progress(0, "Initializing...")

        # Update UI from queues
        while not st.session_state.log_queue.empty():
            st.session_state.log_messages.append(st.session_state.log_queue.get().getMessage())
        log_area.code("\n".join(st.session_state.log_messages[-150:]))

        progress_val, progress_text = 0, "Running..."
        while not st.session_state.progress_queue.empty():
            progress_val, progress_text = st.session_state.progress_queue.get()
        progress_bar.progress(progress_val, text=progress_text)

        # Check thread status
        if st.session_state.tuning_thread and st.session_state.tuning_thread.is_alive():
            time.sleep(1) # Poll every second
            st.rerun()
        else:
            # Thread is finished, set message, clean up, and rerun
            was_stopped = st.session_state.stop_event and st.session_state.stop_event.is_set()
            if was_stopped:
                st.session_state.completion_message_tune = ("Tuning process was stopped by the user.", "success")
            else:
                st.session_state.completion_message_tune = ("Tuning complete! Best parameters have been saved.", "success")

            # Cleanup
            st.session_state.tuning_in_progress = False
            st.session_state.tuning_thread = None
            st.session_state.log_queue = None
            st.session_state.progress_queue = None
            st.session_state.stop_event = None
            st.session_state.log_messages = []
            st.rerun()

def run_training_in_thread(params, log_q, stop_event):
    """Wrapper to run the training process in a separate thread and log via a queue."""
    # Setup QueueHandler to route logs from the thread to the main UI
    qh = logging.handlers.QueueHandler(log_q)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt='%H:%M:%S')
    qh.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(qh)
    try:
        run_pybroker_walkforward(
            ticker=params['ticker'],
            strategy_type=params['strategy_type'],
            start_date=params['start_date'].strftime('%Y-%m-%d'),
            end_date=params['end_date'].strftime('%Y-%m-%d'),
            tune_hyperparameters=params['tune_hyperparams'],
            plot_results=False, # We will plot in Streamlit
            save_assets=True,
            use_tuned_strategy_params=params['use_tuned_params'],
            commission_cost=params['commission'],
            stop_event_checker=lambda: stop_event.is_set()
        )
    finally:
        # Important to remove the handler to avoid duplicate logs on subsequent runs
        root_logger.removeHandler(qh)

with train_tab:
    st.header("Train Final Model")

    # Display completion message if it exists from a previous run
    if st.session_state.get('completion_message_train'):
        message, msg_type = st.session_state.completion_message_train
        if msg_type == "success":
            st.success(message)
        elif msg_type == "warning":
            st.warning(message)
        st.session_state.completion_message_train = None # Clear the message

    # Show form only if training is not in progress
    if not st.session_state.training_in_progress:
        with st.form("train_form"):
            c1, c2, c3 = st.columns(3)
            ticker_train = c1.text_input("Ticker", "SPY", key="train_ticker").upper()
            strategy_type_train = c2.selectbox("Strategy Type", strategy_options, key="train_strat")
            commission_train = c3.number_input("Commission ($ per share)", value=0.0, format="%.4f", key="train_comm")
    
            c1, c2 = st.columns(2)
            start_date_train = c1.date_input("Start Date", datetime(2000, 1, 1), key="train_start")
            end_date_train = c2.date_input("End Date", datetime.now(), key="train_end")
    
            c1, c2 = st.columns(2)
            use_tuned_params_train = c1.checkbox("Use Tuned Strategy Params?", value=True)
            tune_model_hyperparams = c2.checkbox("Tune Model Hyperparameters?", value=True)
    
            run_training = st.form_submit_button("Run Training", use_container_width=True, disabled=DEMO_MODE)
    
            if run_training:
                st.session_state.training_in_progress = True
                st.session_state.train_params = {
                    "ticker": ticker_train, "strategy_type": strategy_type_train, "commission": commission_train,
                    "start_date": start_date_train, "end_date": end_date_train,
                    "use_tuned_params": use_tuned_params_train, "tune_hyperparams": tune_model_hyperparams
                }
                st.rerun()
    
    # This block runs when training_in_progress is True
    if st.session_state.training_in_progress:
        # If the thread object doesn't exist, it means we need to start it.
        if st.session_state.training_thread is None:
            st.session_state.log_queue = queue.Queue()
            st.session_state.log_messages = [] # Clear old logs
            st.session_state.stop_event = threading.Event()

            thread = threading.Thread(
                target=run_training_in_thread,
                args=(st.session_state.train_params, st.session_state.log_queue, st.session_state.stop_event),
                daemon=True # This is crucial to allow the main app to exit even if the thread is running
            )
            st.session_state.training_thread = thread
            thread.start()

        if st.button("Stop Training Process", use_container_width=True, type="primary"):
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            st.warning("Stop signal sent. The process will halt at the next available checkpoint.")

        # --- UI Monitoring Section ---
        params = st.session_state.train_params
        st.info(f"Training in progress for {params['ticker']} with {params['strategy_type']}...")
        log_container = st.expander("Training Log", expanded=True)
        log_area = log_container.empty()

        # Update UI from queue
        while not st.session_state.log_queue.empty():
            st.session_state.log_messages.append(st.session_state.log_queue.get().getMessage())
        log_area.code("\n".join(st.session_state.log_messages[-150:]))

        # Check thread status
        if st.session_state.training_thread and st.session_state.training_thread.is_alive():
            time.sleep(1) # Poll every second
            st.rerun()
        else:
            # Thread is finished, set message, clean up, and rerun
            was_stopped = st.session_state.stop_event and st.session_state.stop_event.is_set()
            if was_stopped:
                st.session_state.completion_message_train = ("Training process was stopped by the user.", "success")
            else:
                st.session_state.completion_message_train = ("Training complete! Check logs for status. You can now visualize the model in the next tab.", "success")

            # Cleanup
            st.session_state.training_in_progress = False
            st.session_state.training_thread = None
            st.session_state.log_queue = None
            st.session_state.stop_event = None
            st.session_state.log_messages = []
            st.rerun()

with visualize_tab:
    st.header("Visualize Trained Model Performance")
    st.markdown("Load the saved artifacts from a `train` run to visualize its out-of-sample performance, trade executions, and feature importances.")

    if 'viz_results' not in st.session_state:
        st.session_state.viz_results = None

    def clear_viz_results():
        """Callback to clear visualization results when the selection changes."""
        if 'viz_results' in st.session_state:
            st.session_state.viz_results = None

    # Get a list of all trained models from the pybroker_trainer artifacts directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'pybroker_trainer', 'artifacts')
    trained_models = []
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            # The `_results.pkl` file is the one artifact guaranteed to exist for any trained strategy (ML or non-ML).
            # We use this file to discover all available backtest results.
            if f.endswith('_results.pkl'):
                # Example filename: SPY_ma_crossover_results.pkl
                parts = f.replace('_results.pkl', '').split('_')
                if len(parts) >= 2:
                    ticker = parts[0]
                    strategy = "_".join(parts[1:])
                    trained_models.append(f"{ticker} - {strategy}")

    if not trained_models:
        st.warning("No trained models found in the pybroker_trainer artifacts directory. Please run the 'train' command first.")
        selected_model = None
    else:
        selected_model = st.selectbox(
            "Select a Trained Model to Visualize",
            options=sorted(list(set(trained_models))), # Use set to remove duplicates
            key="selected_model_viz", # Add a key for state management
            on_change=clear_viz_results # Clear old results when selection changes
        )

    visualize_button = st.button("Load and Visualize", use_container_width=True, disabled=(not selected_model))

    if visualize_button and selected_model:
        ticker, strategy_type = selected_model.split(' - ', 1)
        with st.spinner(f"Loading artifacts and generating plots for {ticker} - {strategy_type}..."):
            try:
                viz_assets = run_visualize_model(ticker, strategy_type)
                st.session_state.viz_results = viz_assets
                if viz_assets is None:
                    st.error(f"Could not load visualization assets for {ticker} - {strategy_type}. Check the logs. A common cause is missing artifact files (e.g., _results.pkl).")
                else:
                    st.success("Visualization assets loaded successfully.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.viz_results = None

    if st.session_state.viz_results:
        # Add a button to clear results and go back to the selection form
        if st.button("Clear Visualization & Select New Model", use_container_width=True):
            st.session_state.viz_results = None
            st.rerun()

        results = st.session_state.viz_results

        st.subheader("Best Parameters Found")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Strategy Parameters (from `tune-strategy`)")
            if results.get("best_strategy_params"):
                st.json(results["best_strategy_params"])
            else:
                st.info("No tuned strategy parameters file (`_best_strategy_params.json`) found.")
        with col2:
            st.markdown("##### Model Hyperparameters (from `train`)")
            if results.get("best_model_params"):
                st.json(results["best_model_params"])
            else:
                st.info("No tuned model hyperparameters file (`_best_params.json`) found (tuning may have been disabled).")

        st.subheader("Walk-Forward Performance Metrics")
        st.dataframe(results["metrics_df"])

        st.subheader("Walk-Forward Equity Curve")
        if results.get("equity_fig"): st.pyplot(results["equity_fig"])

        st.subheader("Trade Executions on Price Chart")
        if results.get("trades_fig"): st.pyplot(results["trades_fig"])

        with st.expander("View Detailed Trades Data"):
            trades_df = results.get("trades_df")
            if trades_df is not None and not trades_df.empty:
                st.dataframe(trades_df)
            else:
                st.info("No trades were executed in this backtest, or trade data is unavailable.")

        st.subheader("Feature Importances (Averaged Across Folds)")
        if results.get("importance_fig"): st.pyplot(results["importance_fig"])

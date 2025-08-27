import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

from load_cfg import DEMO_MODE, WORKING_DIRECTORY
from pybroker_trainer.strategy_loader import get_strategy_class_map
from quant_engine import run_scan, get_default_tickers

st.set_page_config(page_title="Market Scanner", layout="wide")

st.title("🔍 Market Scanner Results")

if DEMO_MODE:
    st.warning(
        "**Demo Mode Active:** Running new interactive scans is disabled. "
        "To enable this feature, set `DEMO_MODE = False` in `load_cfg.py`.",
        icon="🔒"
    )

# --- Session State ---
if 'scanner_results_df' not in st.session_state:
    st.session_state.scanner_results_df = None
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None
if 'scan_source' not in st.session_state:
    st.session_state.scan_source = "file" # 'file' or 'interactive'

SCAN_RESULTS_FILE = os.path.join(WORKING_DIRECTORY, 'scan_results.json')

# --- Helper to load from file ---
def load_from_file():
    if os.path.exists(SCAN_RESULTS_FILE):
        try:
            last_modified_time = datetime.fromtimestamp(os.path.getmtime(SCAN_RESULTS_FILE))
            with open(SCAN_RESULTS_FILE, 'r') as f:
                data = json.load(f)
            st.session_state.scanner_results_df = pd.DataFrame(data) if data else pd.DataFrame()
            st.session_state.last_scan_time = last_modified_time
            st.session_state.scan_source = "file"
        except (json.JSONDecodeError, IOError) as e:
            st.error(f"Error loading cached scan results file: {e}")
            st.session_state.scanner_results_df = pd.DataFrame()
            st.session_state.last_scan_time = None
    else:
        st.session_state.scanner_results_df = pd.DataFrame()
        st.session_state.last_scan_time = None

# --- UI for Interactive Scan ---
with st.expander("🔬 Run a New Scan", expanded=False):
    with st.form("scan_form"):
        strategy_options = list(get_strategy_class_map().keys())
        default_tickers_str = ",".join(get_default_tickers(source=2, limit=100))
        
        tickers_input = st.text_area("Tickers (comma-separated)", value=default_tickers_str, height=100)
        strategies_input = st.multiselect("Strategies to Scan", options=strategy_options, default=strategy_options)
        
        run_interactive_scan = st.form_submit_button("Run Interactive Scan", disabled=DEMO_MODE)

if run_interactive_scan:
    if not tickers_input:
        st.error("Please enter at least one ticker.")
    elif not strategies_input:
        st.error("Please select at least one strategy.")
    else:
        tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        
        # --- Add a progress bar for better user feedback ---
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def update_progress(progress, text):
            progress_bar.progress(progress)
            progress_text.text(text)

        try:
            interactive_results = run_scan(
                ticker_list=tickers_list,       
                strategy_list=strategies_input, 
                progress_callback=update_progress # NEW: Pass the callback function
            )
            st.session_state.scanner_results_df = pd.DataFrame(interactive_results) if interactive_results else pd.DataFrame()
            st.session_state.last_scan_time = datetime.now()
            st.session_state.scan_source = "interactive"
            st.success("Interactive scan complete!")
        except Exception as e:
            st.error(f"Error during interactive scan: {e}")
            st.session_state.scanner_results_df = pd.DataFrame()
            st.session_state.last_scan_time = None

if st.button("Refresh from File", help="Load the latest scan results from the saved file."):
    # --- FIX: A more robust way to force a reload from file ---
    # By clearing the state and re-running, we ensure the initial load logic is triggered.
    st.session_state.scanner_results_df = None
    st.session_state.last_scan_time = None
    st.session_state.scan_source = "file"
    st.rerun() # Force an immediate rerun to display the new data

# --- Initial Load and Display Logic ---
# Only auto-load from file if the results have never been populated.
if st.session_state.scanner_results_df is None:
    load_from_file()

results_df = st.session_state.scanner_results_df
last_scan_time = st.session_state.last_scan_time

if last_scan_time:
    st.info(f"Scan results last updated: **{last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}**")
else:
    st.warning("Scan results file not found. Please run the scanner first: `python quant_engine.py scan`")

if results_df is not None:
    if results_df.empty:
        st.success("✅ No active trading signals found in the latest scan.")
    else:
        st.subheader("Active Trading Signals")

        # --- Data Cleaning and Formatting ---
        def format_probs(p):
            if isinstance(p, list) and len(p) > 1:
                return p[1] # For binary classification [prob_loss, prob_win], show prob_win
            return p # Fallback for other formats

        results_df['probability'] = results_df['probabilities'].apply(format_probs)
        # --- NEW: Add a column for the Yahoo Finance link ---
        results_df['yahoo_link'] = "https://finance.yahoo.com/quote/" + results_df['ticker']
        results_df = results_df.sort_values(by='probability', ascending=False)

        # --- Display DataFrame ---
        st.dataframe(
            results_df[['ticker', 'strategy', 'date', 'close', 'probability', 'yahoo_link']],
            column_config={
                "ticker": st.column_config.TextColumn("Ticker"),
                "strategy": st.column_config.TextColumn("Strategy"),
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "close": st.column_config.NumberColumn("Close Price", format="$%.2f"),
                "probability": st.column_config.ProgressColumn(
                    "Win Probability",
                    help="Model's predicted probability of a winning trade.",
                    format="%.2f", min_value=0, max_value=1,
                ),
                "yahoo_link": st.column_config.LinkColumn(
                    "Yahoo Finance",
                    help="Link to the ticker's Yahoo Finance page.",
                    display_text="🔗 Link"
                )
            },
            use_container_width=True, hide_index=True
        )

        # --- NEW: Market Regime Analysis Section ---
        st.subheader("Market Regime Analysis")
        st.markdown("""
        This analysis counts the number of signals generated by each strategy in the latest scan.
        The prevalence of certain strategy types can provide insights into the current market's character.
        A high number of momentum signals (`donchian_breakout`) suggests a different environment than a high number of mean-reversion signals (`bb_extreme_reversal`).
        """)

        strategy_counts = results_df['strategy'].value_counts()

        strategy_interpretations = {
            'uptrend_pullback': "Indicates a **Healthy Uptrend** where stocks are experiencing normal pullbacks to moving averages.",
            'bb_extreme_reversal': "Suggests a **Choppy or Ranging Market** where prices are oscillating between support and resistance.",
            'donchian_breakout': "Points to a **Strong Trending / Momentum Market** where buying new highs is being rewarded.",
            'rsi_divergence': "Highlights potential **Trend Exhaustion or Reversal** points, often at the end of a strong move.",
            'bb_squeeze_breakout': "Signals a period of **Low Volatility followed by a Breakout**, indicating a potential new trend is starting.",
            'gap_and_go': "Reflects a **High Momentum, News-Driven Market** where significant overnight news causes large price gaps.",
            'selling_climax': "Indicates potential **Capitulation or Market Bottoming**, where panic selling creates a reversal opportunity.",
            'wyckoff_spring': "Suggests a **Shakeout or Accumulation Phase**, where weak hands are forced out before a potential uptrend resumes.",
            'consolidation_breakout': "Identifies stocks breaking out from a **Consolidation or Basing Pattern**, suggesting the start of a new directional move after a period of indecision.",
            'ma_crossover': "Signals a potential **Shift in Trend Direction**, where a shorter-term trend is overtaking a longer-term trend, often indicating the start of a new sustained move.",
        }

        c1, c2 = st.columns(2)
        with c1:
            st.write("##### Signal Counts by Strategy")
            st.bar_chart(strategy_counts)
        with c2:
            st.write("##### Strategy Interpretations")
            interpretation_df = pd.DataFrame(
                [(strat, strategy_interpretations.get(strat, "No interpretation available.")) for strat in strategy_counts.index],
                columns=['Strategy', 'Potential Market Regime']
            )
            st.dataframe(interpretation_df, hide_index=True, use_container_width=True)
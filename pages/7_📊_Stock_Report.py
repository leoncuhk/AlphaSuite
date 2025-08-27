import base64
import os
import streamlit as st
import traceback

from tools.canslim_analysis_tool import CanslimReportGenerator
from tools.yfinance_tool import is_ticker_active
from load_cfg import DEMO_MODE, LLM_PROVIDER, get_llm

st.set_page_config(page_title="Stock Report Generator", layout="wide")

st.title("📊 Stock Report Generator")

if DEMO_MODE:
    st.warning(
        "**Demo Mode Active:** Live report generation is disabled. Sample reports will be shown instead.",
        icon="🔒"
    )

st.markdown(f"""
Enter a stock ticker to generate a report. Choose from:
- **CANSLIM Analysis**: A detailed evaluation based on William O'Neil's CANSLIM investing principles.
- **Comprehensive Report**: View a sample report. *(Note: This feature is for demonstration only. Contact the author for details on the full version.)*

The currently configured LLM provider is: **{LLM_PROVIDER.upper()}**
""")

# --- State Management to preserve the report and ticker ---
if 'report' not in st.session_state:
    st.session_state.report = ""
if 'ticker' not in st.session_state:
    st.session_state.ticker = "META"  # Default ticker
if 'report_bytes' not in st.session_state:
    st.session_state.report_bytes = None

def clear_report_state():
    """Clears the generated report and its bytes from the session state."""
    st.session_state.report = ""
    st.session_state.report_bytes = None

def generate_and_display_report(ticker_symbol, report_type):
    """Generates and displays the selected report, or a sample for the comprehensive one."""
    report_file = None
    try:
        if report_type == "Comprehensive Report":
            st.info("Displaying a sample Comprehensive Report. This feature is for demonstration purposes.")
            # Construct path to the sample report in the 'samples' directory at the project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            report_file = os.path.join(project_root, "samples", "META_comprehensive_report_20250824.pdf")

            if not os.path.exists(report_file):
                st.error("Sample report file not found. Please ensure 'samples/META_comprehensive_report_20250824.pdf' exists.")
                clear_report_state()
                return

        elif report_type == "CANSLIM Analysis":
            if DEMO_MODE:
                st.info("Displaying a sample CANSLIM Analysis Report. This feature is for demonstration purposes.")
                # Construct path to the sample report in the 'samples' directory at the project root
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                report_file = os.path.join(project_root, "samples", "META_canslim_report_20250824.pdf")

                if not os.path.exists(report_file):
                    st.error("Sample report file not found. Please ensure 'samples/META_canslim_report_20250824.pdf' exists.")
                    clear_report_state()
                    return
            else: # Original logic for non-demo mode
                if not is_ticker_active(ticker_symbol):
                    st.error(f"Ticker {ticker_symbol} is not actively traded or could not be found.")
                    clear_report_state()
                    return

                spinner_text = f"Generating {report_type.lower()} for {ticker_symbol}... This may take some time."
                with st.spinner(spinner_text):
                    llm = get_llm()
                    generator = CanslimReportGenerator(llm)
                    report_file = generator.generate_report(ticker_symbol)
        else:
            st.error("Invalid report type selected.")
            clear_report_state()
            return

        if report_file:
            st.session_state.report = report_file
            with open(report_file, "rb") as f:
                st.session_state.report_bytes = f.read()
            st.success(f"Report '{os.path.basename(st.session_state.report)}' loaded successfully.")

    except Exception as e:
        st.error(f"An error occurred while generating the report for {ticker_symbol}: {e}")
        clear_report_state()
        traceback.print_exc()

# --- UI Components ---
ticker_input = st.text_input(
    "Enter Stock Ticker:", 
    value=st.session_state.ticker, 
    on_change=clear_report_state,
    disabled=DEMO_MODE
).upper()

report_type = st.radio(
    "Select Report Type:",
    ("CANSLIM Analysis", "Comprehensive Report"),
    key="report_type_selection",
    on_change=clear_report_state
)

if st.button("Generate Report"):
    st.session_state.ticker = ticker_input
    generate_and_display_report(st.session_state.ticker, report_type)

# --- Download Report ---
if st.session_state.report_bytes:
    st.download_button(
        label="Download Report",
        data=st.session_state.report_bytes,
        file_name=os.path.basename(st.session_state.report),
        mime="application/pdf"
    )

# --- Display Report ---
if st.session_state.report and st.session_state.report_bytes:
    st.subheader("Generated Report Preview")
    try:
        base64_pdf = base64.b64encode(st.session_state.report_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not display PDF preview: {e}")

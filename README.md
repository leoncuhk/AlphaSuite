# AlphaSuite

AlphaSuite is a comprehensive suite of tools for quantitative financial analysis, model training, backtesting, and trade management. It's designed for traders and analysts who want to build, validate, and deploy data-driven trading strategies.

## ✨ Key Features

*   **Strategy Development & Backtesting**:
    *   **Model Training & Tuning**: Fine-tune strategy parameters using Bayesian optimization and train final models with walk-forward analysis.
    *   **Performance Visualization**: Visualize a tuned model's out-of-sample performance, trade executions, and feature importances.
    *   **Portfolio Analysis**: Discover which stocks are suitable for a strategy and run portfolio-level backtests to validate your ideas.
    *   **Interactive Backtester**: Visualize the in-sample performance of a saved model on historical data.
*   **Live Analysis & Trading**:
    *   **Market Scanner**: Scan the market for trading signals based on pre-trained models or run an interactive scan on-demand.
    *   **Portfolio Manager**: Manually add, view, and manage your open trading positions.
*   **Data & Research**:
    *   **Data Management**: Control the entire data pipeline, from downloading market data to running rule-based scanners.
    *   **AI-Powered Stock Reports**: Generate a comprehensive fundamental and technical analysis report or CANSLIM analysis for any stock.
    *   **News Intelligence**: Scans recent news, generates a detailed market briefing, and analyzes it against economic risk profiles.
*   **Robust Data Pipeline**: Fetches and stores comprehensive company data, price history, financials, and analyst estimates from Yahoo Finance into a PostgreSQL database.
*   **Interactive Web UI**: A Streamlit-based dashboard for managing data, training models, and analyzing results.

## 📖 Articles & Case Studies

Check out these articles to see how AlphaSuite can be used to develop and test sophisticated trading strategies from scratch:

*   **[From Chaos Theory to a Profitable Trading Strategy in 30 Minutes](https://medium.com/@rshu/from-chaos-theory-to-a-profitable-trading-strategy-in-30-minutes-d247cba4bbbd)**: A step-by-step guide on building a rule-based strategy using concepts from chaos theory.
*   **[From Chaos to Alpha, Part 2: Supercharging a Machine Learning Strategy with Lorenz Features](https://rshu.medium.com/from-chaos-to-alpha-part-2-supercharging-a-machine-learning-strategy-with-lorenz-features-794acfd3f88c)**: Demonstrates how to enhance an ML-based strategy with custom features and optimize it using walk-forward analysis.

## 🛠️ Tech Stack

*   **Backend**: Python
*   **Web Framework**: Streamlit
*   **Data Analysis**: Pandas, NumPy, SciPy
*   **Financial Data**: yfinance, TA-Lib
*   **Database**: PostgreSQL with SQLAlchemy
*   **AI/LLM**: LangChain, Google Gemini, Ollama

## 🚀 Getting Started

Follow these steps to set up and run AlphaSuite on your local machine.

### 1. Prerequisites

*   Python 3.9+
*   PostgreSQL Server
*   Git

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AlphaSuite.git
    cd AlphaSuite
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    *   **TA-Lib**: This library has a C dependency that must be installed first. Follow the official [TA-Lib installation instructions](https://github.com/mrjbq7/ta-lib) for your operating system.
    *   Install the remaining Python packages:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Set up the Database:**
    *   Ensure your PostgreSQL server is running.
    *   Create a new database (e.g., `alphasuite`).
    *   The application will create the necessary tables on its first run.

5.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Open the `.env` file and edit the variables:
        *   `DATABASE_URL`: Set this to your PostgreSQL connection string.
        *   `LLM_PROVIDER`: Set to `gemini` or `ollama` to choose your provider.
        *   `GEMINI_API_KEY`: Required if `LLM_PROVIDER` is `gemini`.
        *   `OLLAMA_URL`: The URL for your running Ollama instance (e.g., `http://localhost:11434`). Required for `ollama`.
        *   `OLLAMA_MODEL`: The name of the model you have pulled in Ollama (e.g., `llama3`).

### 3. Usage

1.  **Initial Data Download:**
    Before running the app, you need to populate the database with market data. Run the download script from your terminal. This may take a long time for the initial run.
    ```bash
    # Download data for the US market (recommended for first run)
    python download_data.py --run_daily_pipeline=true
    ```

2.  **Run the Streamlit Web Application:**
    ```bash
    streamlit run 🏠_Home.py
    ```
    Open your web browser to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Follow the In-App Workflow:**
    1.  **Populate Data:** Go to the **Data Management** page and run the "Daily Pipeline" or a "Full Download".
    2.  **Scan for Signals:** Use the **Market Scanner** to find live trading signals.
    3.  **Tune & Train:** To build custom models, navigate to the **Model Training & Tuning** page.
    4.  **Analyze & Backtest:** Use the **Portfolio Analysis** page to validate your strategies.
    5.  **Deep Research:** Use the **Stock Report** page for in-depth analysis of specific stocks.

## ⚖️ License

This project is licensed under the MIT License - see the LICENSE file for details.
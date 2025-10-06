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

*   **[The Institutional Edge: How We Boosted a Strategy’s Return with Volume Profile](https://medium.com/codex/the-institutional-edge-how-we-boosted-a-strategys-return-from-162-to-223-with-one-indicator-eef74cadae91)**: A deep dive into using Volume Profile to enhance a classic trend-following strategy, demonstrating a significant performance boost.
*   **[I Was Paralyzed by Uncertainty, So I Built My Own Quant Engine](https://medium.com/codex/i-was-paralyzed-by-stock-market-uncertainty-so-i-built-my-own-quant-engine-176a6706c451)**: The story behind AlphaSuite's creation and its mission to empower data-driven investors. Also available as a [video narration](https://youtu.be/NXk7bXPYGP8).
*   **[From Chaos Theory to a Profitable Trading Strategy](https://medium.com/codex/from-chaos-theory-to-a-profitable-trading-strategy-in-30-minutes-d247cba4bbbd)**: A step-by-step guide on building a rule-based strategy using concepts from chaos theory.
*   **[Supercharging a Machine Learning Strategy with Lorenz Features](https://medium.com/codex/from-chaos-to-alpha-part-2-supercharging-a-machine-learning-strategy-with-lorenz-features-794acfd3f88c)**: Demonstrates how to enhance an ML-based strategy with custom features and optimize it using walk-forward analysis.

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

---

## 🍎 macOS 详细安装指南

本节提供针对 macOS 用户的详细安装步骤和经验分享。

### 前置要求

确保已安装以下工具：
- **Homebrew**: 如未安装，运行 `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- **uv**: Python 包管理工具，运行 `brew install uv`（如已用其他方式安装可跳过）

### 详细安装步骤

#### 1. 安装系统依赖

```bash
# 安装 PostgreSQL 数据库
brew install postgresql@15

# 安装 TA-Lib C 库（技术分析库的底层依赖）
brew install ta-lib

# 启动 PostgreSQL 服务
brew services start postgresql@15
```

**注意**：PostgreSQL@15 是 keg-only 的，这意味着它不会自动添加到 PATH。如果需要直接使用 `psql` 命令，可以添加到 PATH：
```bash
echo 'export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### 2. 创建虚拟环境（使用 uv）

```bash
# 进入项目目录
cd AlphaSuite

# 创建 Python 3.11 虚拟环境
uv venv --python 3.11

# 激活虚拟环境
source .venv/bin/activate
```

#### 3. 安装 Python 依赖

```bash
# 使用 uv 安装所有依赖（包括 TA-Lib Python 包装器）
uv pip install -r requirements.txt
```

**重要提示**：由于已通过 Homebrew 安装了 TA-Lib C 库，Python 的 TA-Lib 包装器会自动找到并链接到系统库，无需手动下载 wheel 文件。

#### 4. 创建数据库

```bash
# 使用默认用户创建数据库
/opt/homebrew/opt/postgresql@15/bin/createdb alphasuite
```

#### 5. 配置环境变量

```bash
# 复制示例配置文件
cp .env.example .env

# 编辑 .env 文件，修改数据库连接字符串
# 将 DATABASE_URL 改为：postgresql://你的用户名@localhost:5432/alphasuite
# macOS 默认 PostgreSQL 用户名是你的系统用户名，不需要密码
```

示例 `.env` 配置：
```bash
DATABASE_URL=postgresql://leon@localhost:5432/alphasuite
DEMO_MODE=false
WORKING_DIRECTORY=./work/
LLM_PROVIDER=gemini
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

#### 6. 验证安装

运行测试脚本确保所有组件正常工作：
```bash
python test_installation.py
```

成功的输出应显示所有测试通过：
- ✓ 库导入
- ✓ TA-Lib 功能
- ✓ 数据库连接

### 常见问题

#### Q1: TA-Lib 安装失败
如果遇到 TA-Lib 相关错误：
1. 确保先安装了 C 库：`brew install ta-lib`
2. 确认 Homebrew 路径正确（Apple Silicon 使用 `/opt/homebrew`，Intel Mac 使用 `/usr/local`）
3. 尝试重新安装：`uv pip install --force-reinstall TA-Lib`

#### Q2: PostgreSQL 连接失败
- 确认服务正在运行：`brew services list | grep postgresql`
- 检查数据库是否存在：`/opt/homebrew/opt/postgresql@15/bin/psql -l`
- 确保 `.env` 中的用户名与系统用户名一致

#### Q3: 虚拟环境激活问题
使用 uv 创建的虚拟环境需要使用标准方式激活：
```bash
source .venv/bin/activate
```

### 性能优化建议

1. **使用 uv 代替 pip**：uv 是用 Rust 编写的极快的 Python 包管理器，安装速度比 pip 快 10-100 倍
2. **SSD 存储**：建议将项目和数据库存放在 SSD 上以获得最佳性能
3. **内存配置**：如处理大量数据，建议至少 16GB RAM

### 下一步

安装完成后，按照以下顺序开始使用：

1. **初始化数据**（首次运行需要较长时间）：
   ```bash
   python download_data.py --run_daily_pipeline=true
   ```

2. **启动 Web 应用**：
   ```bash
   streamlit run 🏠_Home.py
   ```

3. **访问应用**：
   在浏览器中打开 `http://localhost:8501`

### 卸载说明

如需完全卸载：
```bash
# 停止并移除 PostgreSQL 服务
brew services stop postgresql@15
brew uninstall postgresql@15 ta-lib

# 删除数据库数据（可选）
rm -rf /opt/homebrew/var/postgresql@15

# 删除虚拟环境
rm -rf .venv

# 删除项目目录
cd .. && rm -rf AlphaSuite
```
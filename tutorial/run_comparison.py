"""
KAMA + ATR 策略 - 模型对比示例
对比 LightGBM 和 RandomForest 的性能

这个脚本的目的：
1. 展示如何在现有策略上轻松切换和测试不同的ML模型。
2. 对比两种主流模型在相同数据和任务上的分类性能。
3. 对比它们的模拟回测效果，看哪个模型能带来更好的投资回报。
4. 分析并理解模型差异背后的原因。
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
import yfinance as yf
from typing import Optional

# --- 路径和模块导入 ---
warnings.filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入教学模块
from tutorial.feature_engineering import prepare_data_for_ml, get_feature_columns
from tutorial.labeling import label_with_triple_barrier

# 导入模型
try:
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    MODELS_AVAILABLE = True
except ImportError:
    print("错误：请确保 lightgbm, scikit-learn 已安装。 pip install lightgbm scikit-learn")
    MODELS_AVAILABLE = False

# --- 辅助函数 ---
def print_section(title: str):
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)

def download_data(symbol: str = "QQQ", period: str = "5y") -> pd.DataFrame:
    print_section(f"下载 {symbol} 数据 ({period})")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    df.columns = df.columns.str.lower()
    df = df[['open', 'high', 'low', 'close', 'volume']]
    print(f"✓ 数据下载成功: {len(df)} 行")
    return df

# --- 修改后的ML策略类 ---

class ComparableMLStrategy:
    """
    可对比的机器学习策略类
    支持 LightGBM 和 RandomForest
    """
    def __init__(
        self,
        model_type: str = 'lightgbm', # 'lightgbm' 或 'random_forest'
        profit_target_mult: float = 3.0,
        stop_loss_mult: float = 3.0,
        eval_bars: int = 15,
        test_size: float = 0.3,
        random_state: int = 42
    ):
        if not MODELS_AVAILABLE:
            raise ImportError("必要的ML库未安装。")
        
        self.model_type = model_type
        self.profit_target_mult = profit_target_mult
        self.stop_loss_mult = stop_loss_mult
        self.eval_bars = eval_bars
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.feature_importance = None

        print_section(f"初始化策略: {self.model_type.upper()}")

    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        print("1. 准备特征和标签...")
        df = prepare_data_for_ml(df)
        df = label_with_triple_barrier(
            df,
            profit_target_mult=self.profit_target_mult,
            stop_loss_mult=self.stop_loss_mult,
            eval_bars=self.eval_bars
        )
        df_clean = df.dropna(subset=['label']).copy()
        
        print("\n2. 分割训练集和测试集...")
        split_idx = int(len(df_clean) * (1 - self.test_size))
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        
        feature_cols = get_feature_columns()
        X_train = train_df[feature_cols]
        y_train = train_df['label']
        X_test = test_df[feature_cols]
        y_test = test_df['label']
        
        print(f"  训练集: {len(X_train)} | 测试集: {len(X_test)}")
        return X_train, X_test, y_train, y_test, test_df

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        print(f"\n3. 训练 {self.model_type.upper()} 模型...")
        
        if self.model_type == 'lightgbm':
            params = {
                'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
                'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 5,
                'min_data_in_leaf': 20, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
                'verbose': -1, 'random_state': self.random_state
            }
            train_data = lgb.Dataset(X_train, label=y_train)
            self.model = lgb.train(params, train_data, num_boost_round=200)
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importance()
            }).sort_values('importance', ascending=False)

        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,       # 100棵树
                max_depth=10,           # 限制深度防止过拟合
                min_samples_leaf=20,    # 叶节点最小样本数
                random_state=self.random_state,
                n_jobs=-1               # 使用所有CPU核心
            )
            self.model.fit(X_train, y_train)
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print("✓ 训练完成。")

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        if self.model is None:
            raise ValueError("模型尚未训练！")
        
        # Scikit-learn 和 LightGBM 的 predict_proba API 略有不同
        pred_proba = self.model.predict_proba(X)[:, 1] if self.model_type == 'random_forest' else self.model.predict(X)
        return (pred_proba >= threshold).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        print("\n4. 评估模型性能 (样本外测试集)...")
        y_pred = self.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if self.model_type == 'random_forest' else self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        for name, value in metrics.items():
            print(f"  - {name.capitalize():<10}: {value:.3f}")
        return metrics

    def backtest_simple(self, test_df: pd.DataFrame, signals: np.ndarray) -> dict:
        print("\n5. 运行简化回测 (样本外测试集)...")
        
        # 只关注模型发出“买入”信号的交易
        trade_indices = np.where(signals == 1)[0]
        
        if len(trade_indices) == 0:
            print("  - 模型未产生任何买入信号。")
            return {'num_trades': 0}

        # 从 test_df 中提取这些交易的预计算回报
        # 注意：'return_pct' 是由三重障碍法在标记时计算的
        trade_returns = test_df.iloc[trade_indices]['return_pct']
        
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns <= 0]
        
        # 假设每笔交易投资相同金额
        total_return_pct = trade_returns.sum()
        
        results = {
            'num_trades': len(trade_returns),
            'win_rate': len(winning_trades) / len(trade_returns) if len(trade_returns) > 0 else 0,
            'avg_trade_return': trade_returns.mean(),
            'avg_win_return': winning_trades.mean(),
            'avg_loss_return': losing_trades.mean(),
            'total_return_approx': total_return_pct / len(signals) # 简单平均到整个周期
        }
        
        print(f"  - 信号数量: {results['num_trades']}")
        print(f"  - 胜率 (Precision): {results['win_rate']:.3f}")
        print(f"  - 平均每笔交易回报: {results['avg_trade_return']:.3f}%")
        
        return results

def main():
    """主函数：运行模型对比"""
    if not MODELS_AVAILABLE:
        return 1

    df = download_data("QQQ", "5y")

    # --- 运行 LightGBM ---
    lgbm_strategy = ComparableMLStrategy(model_type='lightgbm')
    X_train, X_test, y_train, y_test, test_df = lgbm_strategy.prepare_training_data(df)
    lgbm_strategy.train(X_train, y_train)
    lgbm_metrics = lgbm_strategy.evaluate(X_test, y_test)
    lgbm_signals = lgbm_strategy.predict(X_test)
    lgbm_backtest = lgbm_strategy.backtest_simple(test_df, lgbm_signals)
    lgbm_importance = lgbm_strategy.feature_importance

    # --- 运行 RandomForest ---
    rf_strategy = ComparableMLStrategy(model_type='random_forest')
    # 使用完全相同的数据分割
    rf_strategy.train(X_train, y_train)
    rf_metrics = rf_strategy.evaluate(X_test, y_test)
    rf_signals = rf_strategy.predict(X_test)
    rf_backtest = rf_strategy.backtest_simple(test_df, rf_signals)
    rf_importance = rf_strategy.feature_importance

    # --- 总结报告 ---
    print_section("模型对比总结报告")

    # 合并指标
    summary_df = pd.DataFrame({
        'Metric': list(lgbm_metrics.keys()) + ['num_trades', 'win_rate', 'avg_trade_return'],
        'LightGBM': list(lgbm_metrics.values()) + [lgbm_backtest.get('num_trades', 0), lgbm_backtest.get('win_rate', 0), lgbm_backtest.get('avg_trade_return', 0)],
        'RandomForest': list(rf_metrics.values()) + [rf_backtest.get('num_trades', 0), rf_backtest.get('win_rate', 0), rf_backtest.get('avg_trade_return', 0)]
    })
    summary_df['LightGBM'] = summary_df['LightGBM'].apply(lambda x: f"{x:.3f}")
    summary_df['RandomForest'] = summary_df['RandomForest'].apply(lambda x: f"{x:.3f}")
    
    print("--- 性能指标对比 ---")
    print(summary_df.to_string(index=False))

    print("\n--- 特征重要性对比 (Top 5) ---")
    importance_summary = pd.DataFrame({
        'LGBM_Feature': lgbm_importance['feature'].head(5).values,
        'LGBM_Importance': lgbm_importance['importance'].head(5).values,
        'RF_Feature': rf_importance['feature'].head(5).values,
        'RF_Importance': rf_importance['importance'].head(5).values,
    })
    print(importance_summary.to_string(index=False))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

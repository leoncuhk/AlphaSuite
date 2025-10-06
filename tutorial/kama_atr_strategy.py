"""
KAMA + ATR 策略实现
包含规则策略和机器学习策略两个版本

规则策略：基于固定规则的简单策略（容易过拟合）
ML策略：使用机器学习模型的自适应策略（更稳健）
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# 尝试导入LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM未安装，ML策略将不可用")

from feature_engineering import prepare_data_for_ml, get_feature_columns
from labeling import label_with_triple_barrier


class RuleBasedKAMAStrategy:
    """
    基于规则的KAMA + ATR策略
    
    进场规则：
    1. 价格 > KAMA（上升趋势）
    2. ATR% 在"平静"区间内（波动性适中）
    3. RSI 在合理区间（不要在极端超买/超卖时进场）
    
    出场规则：
    1. 价格 < KAMA（趋势反转）
    2. ATR% 飙升（波动性激增，风险增加）
    3. 达到止损或止盈
    """
    
    def __init__(
        self,
        atr_pct_min: float = 0.01,  # ATR%最小值（波动性下限）
        atr_pct_max: float = 0.03,  # ATR%最大值（波动性上限）
        rsi_min: float = 40,         # RSI最小值
        rsi_max: float = 70,         # RSI最大值
        stop_loss_mult: float = 3.0, # 止损倍数（ATR）
        profit_target_mult: float = 3.0  # 止盈倍数（ATR）
    ):
        """
        初始化规则策略参数
        
        这些参数通常是通过过度优化得到的，在样本外表现会很差！
        """
        self.atr_pct_min = atr_pct_min
        self.atr_pct_max = atr_pct_max
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.stop_loss_mult = stop_loss_mult
        self.profit_target_mult = profit_target_mult
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        参数:
            df: 包含技术指标和特征的DataFrame
        
        返回:
            添加了'signal'列的DataFrame（1=买入，0=持有，-1=卖出）
        """
        df = df.copy()
        
        # 初始化信号列
        df['signal'] = 0
        
        # 进场条件（买入信号）
        buy_condition = (
            (df['close'] > df['kama']) &  # 趋势向上
            (df['feature_atr_pct'] >= self.atr_pct_min) &  # 波动性足够
            (df['feature_atr_pct'] <= self.atr_pct_max) &  # 波动性不过高
            (df['rsi'] >= self.rsi_min) &  # 不在超卖区
            (df['rsi'] <= self.rsi_max)    # 不在超买区
        )
        
        # 出场条件（卖出信号）
        sell_condition = (
            (df['close'] < df['kama']) |  # 趋势反转
            (df['feature_atr_pct'] > self.atr_pct_max * 1.5)  # 波动性激增
        )
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df
    
    def backtest_simple(self, df: pd.DataFrame, initial_capital: float = 10000) -> dict:
        """
        简单回测（不考虑滑点、手续费等）
        
        这个简单回测仅用于快速验证策略逻辑，不适合作为最终评估。
        专业回测请使用PyBroker或类似框架。
        
        参数:
            df: 包含信号的DataFrame
            initial_capital: 初始资金
        
        返回:
            包含回测结果的字典
        """
        df = self.generate_signals(df)
        
        # 初始化
        capital = initial_capital
        position = 0  # 0=空仓，1=持仓
        entry_price = 0
        trades = []
        equity_curve = [initial_capital]
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # 处理买入信号
            if row['signal'] == 1 and position == 0:
                position = 1
                entry_price = row['close']
                entry_date = df.index[i]
                
            # 处理卖出信号
            elif row['signal'] == -1 and position == 1:
                exit_price = row['close']
                exit_date = df.index[i]
                
                # 计算收益
                trade_return = (exit_price - entry_price) / entry_price
                capital *= (1 + trade_return)
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return * 100
                })
                
                position = 0
            
            equity_curve.append(capital)
        
        # 计算统计指标
        if len(trades) > 0:
            trade_returns = [t['return'] for t in trades]
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
            avg_return = np.mean(trade_returns)
            
            # 夏普比率（简化版）
            if np.std(trade_returns) > 0:
                sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(252 / 15)  # 假设平均持仓15天
            else:
                sharpe = 0
        else:
            win_rate = 0
            avg_return = 0
            sharpe = 0
        
        total_return = (capital - initial_capital) / initial_capital * 100
        
        return {
            'total_return': total_return,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_trade_return': avg_return,
            'sharpe_ratio': sharpe,
            'final_capital': capital,
            'trades': trades,
            'equity_curve': equity_curve
        }


class MLKAMAStrategy:
    """
    基于机器学习的KAMA + ATR策略
    
    核心思想：
    不使用固定规则，而是让模型从数据中学习哪些市场条件下应该交易。
    模型会自动发现最重要的特征组合和阈值。
    """
    
    def __init__(
        self,
        profit_target_mult: float = 3.0,
        stop_loss_mult: float = 3.0,
        eval_bars: int = 15,
        test_size: float = 0.3,  # 样本外测试集比例
        random_state: int = 42
    ):
        """
        初始化ML策略参数
        
        参数:
            profit_target_mult: 止盈倍数
            stop_loss_mult: 止损倍数
            eval_bars: 最大持仓天数
            test_size: 测试集比例
            random_state: 随机种子（保证可重现）
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM未安装，请运行: pip install lightgbm")
        
        self.profit_target_mult = profit_target_mult
        self.stop_loss_mult = stop_loss_mult
        self.eval_bars = eval_bars
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        
    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """
        准备训练数据
        
        步骤：
        1. 计算技术指标和特征
        2. 使用三重障碍法标记
        3. 分割训练集和测试集
        
        参数:
            df: 原始OHLCV数据
        
        返回:
            (X_train, X_test, y_train, y_test, df_labeled)
        """
        print("=" * 60)
        print("准备训练数据")
        print("=" * 60)
        
        # 步骤1：特征工程
        print("\n1. 计算技术指标和特征...")
        df = prepare_data_for_ml(df)
        
        # 步骤2：标记数据
        print("\n2. 使用三重障碍法标记交易...")
        df = label_with_triple_barrier(
            df,
            profit_target_mult=self.profit_target_mult,
            stop_loss_mult=self.stop_loss_mult,
            eval_bars=self.eval_bars
        )
        
        # 移除未标记的行
        df_clean = df.dropna(subset=['label']).copy()
        
        # 步骤3：分割数据
        print("\n3. 分割训练集和测试集...")
        split_idx = int(len(df_clean) * (1 - self.test_size))
        
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        
        feature_cols = get_feature_columns()
        
        X_train = train_df[feature_cols]
        y_train = train_df['label']
        X_test = test_df[feature_cols]
        y_test = test_df['label']
        
        print(f"\n训练集大小: {len(X_train)} 样本")
        print(f"测试集大小: {len(X_test)} 样本")
        print(f"特征数量: {len(feature_cols)}")
        print(f"\n训练集成功率: {y_train.mean()*100:.1f}%")
        print(f"测试集成功率: {y_test.mean()*100:.1f}%")
        
        return X_train, X_test, y_train, y_test, df_clean
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        训练LightGBM模型
        
        LightGBM参数说明：
        - objective='binary': 二分类任务
        - metric='auc': 使用AUC作为评估指标
        - learning_rate=0.05: 较慢的学习率，避免过拟合
        - num_leaves=31: 树的复杂度
        - max_depth=5: 限制树的深度，防止过拟合
        - min_data_in_leaf=20: 叶子节点最小样本数
        - feature_fraction=0.8: 每次迭代使用80%的特征（随机性）
        - bagging_fraction=0.8: 每次迭代使用80%的样本（随机性）
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
        """
        print("\n" + "=" * 60)
        print("训练机器学习模型")
        print("=" * 60)
        
        # LightGBM参数
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 5,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state
        }
        
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # 训练模型
        print("\n开始训练...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data],
            valid_names=['train'],
            callbacks=[lgb.log_evaluation(period=50)]
        )
        
        # 特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        print("\n" + "=" * 60)
        print("特征重要性（前10）")
        print("=" * 60)
        print(self.feature_importance.head(10).to_string(index=False))
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        预测交易信号
        
        参数:
            X: 特征DataFrame
            threshold: 分类阈值（默认0.5，可以调整以平衡精确率和召回率）
        
        返回:
            预测标签数组（1=买入，0=不买入）
        """
        if self.model is None:
            raise ValueError("模型尚未训练！请先调用train()方法。")
        
        # 获取预测概率
        pred_proba = self.model.predict(X)
        
        # 应用阈值
        predictions = (pred_proba >= threshold).astype(int)
        
        return predictions
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        评估模型性能
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
        
        返回:
            包含评估指标的字典
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        print("\n" + "=" * 60)
        print("模型评估（样本外测试集）")
        print("=" * 60)
        
        # 预测
        y_pred = self.predict(X_test)
        y_pred_proba = self.model.predict(X_test)
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"\n准确率 (Accuracy): {metrics['accuracy']:.3f}")
        print(f"精确率 (Precision): {metrics['precision']:.3f}")
        print(f"召回率 (Recall): {metrics['recall']:.3f}")
        print(f"F1分数: {metrics['f1_score']:.3f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        
        # 混淆矩阵
        true_positives = ((y_test == 1) & (y_pred == 1)).sum()
        true_negatives = ((y_test == 0) & (y_pred == 0)).sum()
        false_positives = ((y_test == 0) & (y_pred == 1)).sum()
        false_negatives = ((y_test == 1) & (y_pred == 0)).sum()
        
        print(f"\n混淆矩阵:")
        print(f"  真阳性 (TP): {true_positives}")
        print(f"  真阴性 (TN): {true_negatives}")
        print(f"  假阳性 (FP): {false_positives}")
        print(f"  假阴性 (FN): {false_negatives}")
        
        return metrics


if __name__ == "__main__":
    # 测试代码
    import yfinance as yf
    
    print("=" * 60)
    print("KAMA + ATR 策略测试")
    print("=" * 60)
    
    # 下载数据
    print("\n下载QQQ数据（3年）...")
    ticker = yf.Ticker("QQQ")
    df = ticker.history(period="3y")
    df.columns = df.columns.str.lower()
    print(f"数据范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"数据点数: {len(df)}")
    
    # ============ 测试1：规则策略 ============
    print("\n" + "=" * 60)
    print("测试1：规则策略")
    print("=" * 60)
    
    rule_strategy = RuleBasedKAMAStrategy()
    df_with_features = prepare_data_for_ml(df)
    results = rule_strategy.backtest_simple(df_with_features)
    
    print(f"\n规则策略回测结果：")
    print(f"  总收益: {results['total_return']:.2f}%")
    print(f"  交易次数: {results['num_trades']}")
    print(f"  胜率: {results['win_rate']*100:.1f}%")
    print(f"  平均交易收益: {results['avg_trade_return']:.2f}%")
    print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
    
    # ============ 测试2：ML策略 ============
    if LIGHTGBM_AVAILABLE:
        print("\n" + "=" * 60)
        print("测试2：机器学习策略")
        print("=" * 60)
        
        ml_strategy = MLKAMAStrategy()
        
        # 准备数据
        X_train, X_test, y_train, y_test, df_labeled = ml_strategy.prepare_training_data(df)
        
        # 训练模型
        ml_strategy.train(X_train, y_train)
        
        # 评估模型
        metrics = ml_strategy.evaluate(X_test, y_test)
        
        print("\n✅ 策略测试完成！")
    else:
        print("\n⚠️  跳过ML策略测试（LightGBM未安装）")
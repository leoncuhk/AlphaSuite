"""
KAMA + ATR 机器学习策略
基于Kaufman自适应移动平均线和波动率的趋势跟踪策略

这是对病毒式传播文章的正确实现，使用AlphaSuite框架进行专业回测。

核心思想：
1. KAMA用于识别趋势（自适应，在趋势市场快速反应，震荡市场慢速反应）
2. ATR用于波动率过滤（在低波动期交易，避免高波动风险期）
3. 机器学习模型学习最佳的特征组合和阈值
4. Walk-forward analysis确保样本外验证

参考文章：
"We Backtested a Viral Trading Strategy. The Results Will Teach You a Lesson."
by Richard Shu
"""

import sys
import os

# 添加项目根目录到路径（用于独立运行）
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import talib

from pybroker_trainer.strategy_sdk import BaseStrategy


class KamaAtrMLStrategy(BaseStrategy):
    """
    KAMA + ATR 机器学习策略实现
    
    这个策略集成到AlphaSuite框架中，可以使用quant_engine进行：
    - Walk-forward optimization
    - 贝叶斯参数调优
    - 专业回测（包括最大回撤、夏普比率等）
    - 实时扫描和信号生成
    """
    
    @staticmethod
    def define_parameters():
        """
        定义策略参数及其调优范围
        
        参数说明：
        - kama_period: KAMA周期（较小=更敏感）
        - kama_fast: 快速EMA常数（用于计算自适应因子）
        - kama_slow: 慢速EMA常数（用于计算自适应因子）
        - atr_period: ATR计算周期
        - initial_stop_atr_multiplier: 初始止损（ATR的倍数）
        - trailing_stop_atr_multiplier: 跟踪止损（ATR的倍数）
        - stop_out_window: 最大持仓天数
        - probability_threshold: ML模型预测概率阈值
        - risk_per_trade_pct: 每笔交易风险（账户权益的百分比）
        """
        return {
            'kama_period': {'type': 'int', 'default': 10, 'tuning_range': (5, 30)},
            'kama_fast': {'type': 'int', 'default': 2, 'tuning_range': (2, 5)},
            'kama_slow': {'type': 'int', 'default': 30, 'tuning_range': (20, 50)},
            'atr_period': {'type': 'int', 'default': 14, 'tuning_range': (10, 30)},
            'sma_long_period': {'type': 'int', 'default': 200, 'tuning_range': (100, 250)},
            'initial_stop_atr_multiplier': {'type': 'float', 'default': 3.0, 'tuning_range': (2.0, 5.0)},
            'trailing_stop_atr_multiplier': {'type': 'float', 'default': 3.0, 'tuning_range': (2.0, 6.0)},
            'stop_out_window': {'type': 'int', 'default': 15, 'tuning_range': (10, 30)},
            'probability_threshold': {'type': 'float', 'default': 0.60, 'tuning_range': (0.50, 0.75)},
            'risk_per_trade_pct': {'type': 'float', 'default': 0.02, 'tuning_range': (0.01, 0.05)},
        }
    
    def get_feature_list(self) -> list[str]:
        """
        返回机器学习模型所需的特征列表
        
        特征设计原则（关键！）：
        1. 标准化：所有特征必须标准化，使其在不同价格水平下可比
        2. 无前瞻性：只使用历史数据，不能包含未来信息
        3. 经济意义：每个特征都应该有明确的交易逻辑
        
        特征说明：
        - feature_atr_pct: ATR占价格的百分比（波动性）
        - feature_price_kama_dist: 价格与KAMA的距离百分比（趋势位置）
        - feature_price_sma200_dist: 价格与200日均线的距离（长期趋势）
        - feature_rsi: 标准化RSI（动量，0-1范围）
        - feature_macdhist: 标准化MACD柱状图（动量加速度）
        - feature_bb_position: 布林带位置（0-1，相对位置）
        - feature_momentum_5: 5日收益率（短期动量）
        - feature_momentum_20: 20日收益率（中期动量）
        - feature_volume_ratio: 成交量比率（相对于20日平均）
        - month_*: 月份哑变量（季节性因素）
        """
        return [
            'feature_atr_pct',           # 波动性（最重要！）
            'feature_price_kama_dist',   # KAMA趋势
            'feature_price_sma200_dist', # 长期趋势（第二重要）
            'feature_rsi',               # 动量
            'feature_macdhist',          # 动量加速度
            'feature_bb_position',       # 布林带位置
            'feature_momentum_5',        # 短期动量
            'feature_momentum_20',       # 中期动量
            'feature_volume_ratio',      # 成交量确认
            *[f'month_{m}' for m in range(1, 13)]  # 季节性
        ]
    
    def add_strategy_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算策略特定的技术指标和标准化特征
        
        这是特征工程的核心！
        
        步骤：
        1. 计算原始技术指标（KAMA、ATR、RSI、MACD等）
        2. 标准化特征（除以价格或归一化到0-1）
        3. 确保特征在不同时期可比较
        """
        # 获取参数
        kama_period = self.params.get('kama_period', 10)
        kama_fast = self.params.get('kama_fast', 2)
        kama_slow = self.params.get('kama_slow', 30)
        atr_period = self.params.get('atr_period', 14)
        sma_long_period = self.params.get('sma_long_period', 200)
        
        # ============ 1. 计算KAMA（Kaufman自适应移动平均线） ============
        # KAMA是一种智能的移动平均线，会根据市场效率自动调整速度
        # 在趋势市场中快速反应，在震荡市场中慢速过滤噪音
        data['kama'] = talib.KAMA(data['close'], timeperiod=kama_period)
        
        # ============ 2. 计算ATR（平均真实波动范围） ============
        # ATR衡量市场波动性，是风险管理的核心指标
        data['atr'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=atr_period)
        
        # ============ 3. 计算长期均线（趋势确认） ============
        data['sma_200'] = talib.SMA(data['close'], timeperiod=sma_long_period)
        
        # ============ 4. 计算其他技术指标 ============
        # RSI - 相对强弱指标（动量）
        data['rsi'] = talib.RSI(data['close'], timeperiod=14)
        
        # MACD - 移动平均收敛散度（趋势和动量）
        data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(
            data['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # 布林带 - 波动性指标
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(
            data['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # ============ 5. 创建标准化特征（关键！） ============
        # 这是整个策略最重要的部分
        
        # 特征1：ATR百分比（波动性特征）
        # 最重要的特征！模型会学习在低波动时期交易
        # 为什么标准化？ATR=5在$50的股票和$500的股票意义完全不同
        data['feature_atr_pct'] = data['atr'] / data['close']
        
        # 特征2：价格与KAMA的距离（趋势特征）
        # 衡量价格相对于自适应趋势线的位置
        # 正值=价格在KAMA之上（上升趋势），负值=价格在KAMA之下
        data['feature_price_kama_dist'] = (data['close'] / data['kama']) - 1
        
        # 特征3：价格与200日均线的距离（长期趋势）
        # 模型的第二重要特征：避免逆势交易
        data['feature_price_sma200_dist'] = (data['close'] / data['sma_200']) - 1
        
        # 特征4：标准化RSI（动量特征）
        # RSI已经在0-100范围内，归一化到0-1使其与其他特征一致
        data['feature_rsi'] = data['rsi'] / 100.0
        
        # 特征5：MACD柱状图标准化（动量加速度）
        # 衡量动量的变化率，对时机选择很重要
        data['feature_macdhist'] = data['macd_hist'] / data['close']
        
        # 特征6：布林带位置（波动性+超买超卖）
        # 0=触及下轨，0.5=中轨，1=触及上轨
        bb_range = data['bb_upper'] - data['bb_lower']
        bb_range = bb_range.replace(0, np.nan)  # 避免除以零
        data['feature_bb_position'] = (data['close'] - data['bb_lower']) / bb_range
        
        # 特征7：价格动量（短期）
        # 5日收益率
        data['feature_momentum_5'] = data['close'].pct_change(5)
        
        # 特征8：价格动量（中期）
        # 20日收益率
        data['feature_momentum_20'] = data['close'].pct_change(20)
        
        # 特征9：成交量比率（确认信号）
        # 当前成交量相对于20日平均的比率
        volume_ma = data['volume'].rolling(window=20).mean()
        data['feature_volume_ratio'] = data['volume'] / volume_ma
        
        return data
    
    def get_setup_mask(self, data: pd.DataFrame) -> pd.Series:
        """
        返回布尔序列，指示哪些交易日满足设置条件
        
        设置条件（Entry Setup）：
        1. 价格 > KAMA（确认上升趋势）
        2. 价格 > 200日均线（长期趋势向上）
        3. ATR不为NaN（数据有效）
        
        注意：这里只是基础过滤，真正的交易决策由ML模型做出
        模型会学习哪些设置条件下成功率最高
        """
        # 基础趋势确认
        is_above_kama = data['close'] > data['kama']
        is_above_sma200 = data['close'] > data['sma_200']
        has_valid_atr = data['atr'].notna() & (data['atr'] > 0)
        
        # 组合所有条件
        setup_mask = is_above_kama & is_above_sma200 & has_valid_atr
        
        return setup_mask
    
    def calculate_target(self, data: pd.DataFrame, setup_mask: pd.Series) -> pd.Series:
        """
        使用三重障碍法计算训练目标
        
        三重障碍法是机器学习交易策略的标准标记方法：
        对于每个潜在交易点，设置三个退出障碍：
        1. 止盈目标（上障碍）：价格上涨到目标位
        2. 止损（下障碍）：价格下跌到止损位
        3. 时间限制（时间障碍）：达到最大持仓天数
        
        第一个触及的障碍决定标签：
        - 触及止盈 = 1（成功交易）
        - 触及止损 = 0（失败交易）
        - 超时 = 0（失败，因为资金被占用但没有足够收益）
        
        这种方法的优势：
        1. 明确定义"成功"和"失败"
        2. 考虑了风险收益比
        3. 避免了前瞻性偏差
        """
        # 获取参数
        profit_mult = self.params.get('initial_stop_atr_multiplier', 3.0)
        stop_mult = self.params.get('initial_stop_atr_multiplier', 3.0)
        eval_bars = self.params.get('stop_out_window', 15)
        
        # 初始化目标列
        target = pd.Series(np.nan, index=data.index)
        
        # 只为满足设置条件的点计算目标
        for i in data.loc[setup_mask].index:
            idx_loc = data.index.get_loc(i)
            
            # 确保有足够的未来数据
            if idx_loc >= len(data) - eval_bars:
                continue
            
            # 当前价格和ATR
            entry_price = data['close'].loc[i]
            atr_at_entry = data['atr'].loc[i]
            
            # 验证数据有效性
            if pd.isna(entry_price) or pd.isna(atr_at_entry) or atr_at_entry <= 0:
                continue
            
            # 计算三个障碍
            profit_target = entry_price + (atr_at_entry * profit_mult)
            stop_loss = entry_price - (atr_at_entry * stop_mult)
            
            # 查看未来的价格走势
            future_bars = data.iloc[idx_loc+1:idx_loc+1+eval_bars]
            
            # 检查哪个障碍先被触及
            hit_profit = False
            hit_stop = False
            
            for _, row in future_bars.iterrows():
                # 检查止盈
                if row['high'] >= profit_target:
                    hit_profit = True
                    break
                # 检查止损
                if row['low'] <= stop_loss:
                    hit_stop = True
                    break
            
            # 设置标签
            if hit_profit:
                target.loc[i] = 1  # 成功
            else:
                target.loc[i] = 0  # 失败（止损或超时）
        
        return target


if __name__ == "__main__":
    """
    测试策略定义
    
    运行此模块可以验证策略配置是否正确
    """
    print("=" * 70)
    print("KAMA + ATR ML策略配置")
    print("=" * 70)
    
    strategy = KamaAtrMLStrategy()
    
    print("\n参数定义:")
    for param_name, param_config in strategy.define_parameters().items():
        print(f"  {param_name:30s} {param_config}")
    
    print(f"\n特征列表 (共{len(strategy.get_feature_list())}个):")
    for i, feature in enumerate(strategy.get_feature_list(), 1):
        print(f"  {i:2d}. {feature}")
    
    print(f"\n策略类型: {'ML策略' if strategy.is_ml_strategy else '规则策略'}")
    print(f"模型配置: {strategy.get_model_config()}")
    
    print("\n✅ 策略配置验证通过！")
    print("\n使用方法：")
    print("  # 训练模型（walk-forward analysis）")
    print("  python quant_engine.py train --ticker QQQ --strategy kama_atr_ml")
    print("\n  # 可视化结果")
    print("  python quant_engine.py visualize-model --ticker QQQ --strategy kama_atr_ml")
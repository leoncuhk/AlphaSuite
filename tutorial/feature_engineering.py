"""
特征工程模块
将原始技术指标转换为机器学习可用的标准化特征

关键思想：
- 原始指标值（如MACD=2.0）在不同价格水平下含义不同
- 标准化后的特征（如MACD/价格）在不同时期可比较
- 这使得模型能够从长期数据中学习稳定的模式
"""

import pandas as pd
import numpy as np
import talib


def calculate_kama(close: pd.Series, period: int = 10) -> pd.Series:
    """
    计算Kaufman自适应移动平均线（KAMA）
    
    KAMA是一种自适应移动平均线，在趋势市场中快速反应，在震荡市场中慢速反应。
    这里使用简化版本（EMA）作为演示，实际应用可以使用完整的KAMA算法。
    
    参数:
        close: 收盘价序列
        period: 周期（默认10）
    
    返回:
        KAMA序列
    """
    # 简化版本：使用EMA代替完整的KAMA算法
    # 完整的KAMA算法更复杂，但这里的重点是特征工程概念
    return close.ewm(span=period, adjust=False).mean()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                  period: int = 14) -> pd.Series:
    """
    计算平均真实波动范围（ATR）
    
    ATR衡量市场波动性，是风险管理的关键指标。
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期（默认14）
    
    返回:
        ATR序列
    """
    return talib.ATR(high, low, close, timeperiod=period)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有基础技术指标
    
    参数:
        df: 包含OHLCV数据的DataFrame，必须有列：open, high, low, close, volume
    
    返回:
        添加了技术指标的DataFrame
    """
    df = df.copy()
    
    # 1. KAMA - 自适应移动平均线
    df['kama'] = calculate_kama(df['close'], period=10)
    
    # 2. ATR - 平均真实波动范围
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    
    # 3. SMA - 简单移动平均线（长期趋势）
    df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
    
    # 4. RSI - 相对强弱指标（动量）
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    
    # 5. MACD - 移动平均收敛散度（趋势和动量）
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
        df['close'], 
        fastperiod=12, 
        slowperiod=26, 
        signalperiod=9
    )
    
    # 6. 布林带 - 波动性指标
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
        df['close'], 
        timeperiod=20, 
        nbdevup=2, 
        nbdevdn=2
    )
    
    return df


def create_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建标准化的机器学习特征
    
    这是整个策略的核心！通过标准化，我们让特征在不同价格水平下具有可比性。
    
    特征设计原则：
    1. 价格相关指标：除以当前价格，转换为百分比
    2. 区间指标（如RSI）：归一化到0-1范围
    3. 距离指标：计算相对距离（百分比偏离）
    
    参数:
        df: 包含技术指标的DataFrame
    
    返回:
        添加了标准化特征的DataFrame
    """
    df = df.copy()
    
    # ============ 特征1：ATR百分比（波动性特征） ============
    # 最重要的特征！模型会学习在低波动时期交易
    # 为什么标准化？ATR=5在$50的股票和$500的股票意义完全不同
    df['feature_atr_pct'] = df['atr'] / df['close']
    
    # ============ 特征2：价格与KAMA的距离（趋势特征） ============
    # 衡量价格相对于自适应趋势线的位置
    # 正值=价格在KAMA之上（上升趋势），负值=价格在KAMA之下
    df['feature_price_kama_dist'] = (df['close'] / df['kama']) - 1
    
    # ============ 特征3：价格与200日均线的距离（长期趋势） ============
    # 模型的第二重要特征：避免逆势交易
    df['feature_price_sma200_dist'] = (df['close'] / df['sma_200']) - 1
    
    # ============ 特征4：标准化RSI（动量特征） ============
    # RSI已经在0-100范围内，归一化到0-1使其与其他特征一致
    df['feature_rsi'] = df['rsi'] / 100.0
    
    # ============ 特征5：MACD柱状图标准化（动量加速度） ============
    # 衡量动量的变化率，对时机选择很重要
    df['feature_macdhist'] = df['macd_hist'] / df['close']
    
    # ============ 特征6：布林带位置（波动性+超买超卖） ============
    # 0=触及下轨，0.5=中轨，1=触及上轨
    bb_range = df['bb_upper'] - df['bb_lower']
    # 避免除以零
    bb_range = bb_range.replace(0, np.nan)
    df['feature_bb_position'] = (df['close'] - df['bb_lower']) / bb_range
    
    # ============ 特征7：价格动量（短期） ============
    # 5日收益率
    df['feature_momentum_5'] = df['close'].pct_change(5)
    
    # ============ 特征8：价格动量（中期） ============
    # 20日收益率
    df['feature_momentum_20'] = df['close'].pct_change(20)
    
    # ============ 特征9：成交量比率（确认信号） ============
    # 当前成交量相对于20日平均的比率
    volume_ma = df['volume'].rolling(window=20).mean()
    df['feature_volume_ratio'] = df['volume'] / volume_ma
    
    # 删除NaN值较多的行（计算指标时产生的）
    # 保守起见，需要至少200个交易日的数据来计算SMA200
    df = df.dropna()
    
    return df


def get_feature_columns() -> list:
    """
    返回用于机器学习的特征列名列表
    
    返回:
        特征列名列表
    """
    return [
        'feature_atr_pct',           # 波动性
        'feature_price_kama_dist',   # KAMA趋势
        'feature_price_sma200_dist', # 长期趋势
        'feature_rsi',               # 动量
        'feature_macdhist',          # 动量加速度
        'feature_bb_position',       # 布林带位置
        'feature_momentum_5',        # 短期动量
        'feature_momentum_20',       # 中期动量
        'feature_volume_ratio'       # 成交量确认
    ]


def prepare_data_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    准备用于机器学习的完整数据集
    
    这是一个便捷函数，按正确顺序执行所有特征工程步骤。
    
    参数:
        df: 原始OHLCV数据
    
    返回:
        包含所有技术指标和标准化特征的DataFrame
    """
    # 步骤1：计算技术指标
    df = calculate_technical_indicators(df)
    
    # 步骤2：创建标准化特征
    df = create_normalized_features(df)
    
    return df


if __name__ == "__main__":
    # 测试代码：验证特征工程是否正常工作
    import yfinance as yf
    
    print("=" * 60)
    print("特征工程模块测试")
    print("=" * 60)
    
    # 下载测试数据
    print("\n下载QQQ测试数据...")
    ticker = yf.Ticker("QQQ")
    df = ticker.history(period="2y")
    df.columns = df.columns.str.lower()
    
    print(f"原始数据形状: {df.shape}")
    print(f"日期范围: {df.index[0]} 到 {df.index[-1]}")
    
    # 计算特征
    print("\n计算技术指标和特征...")
    df_features = prepare_data_for_ml(df)
    
    print(f"处理后数据形状: {df_features.shape}")
    print(f"\n可用特征列:")
    for col in get_feature_columns():
        if col in df_features.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ✗ {col} (缺失)")
    
    # 显示特征统计
    print("\n特征统计摘要:")
    print(df_features[get_feature_columns()].describe())
    
    # 检查是否有无穷大或NaN
    print("\n数据质量检查:")
    feature_cols = get_feature_columns()
    inf_count = np.isinf(df_features[feature_cols]).sum().sum()
    nan_count = df_features[feature_cols].isna().sum().sum()
    print(f"  无穷大值: {inf_count}")
    print(f"  NaN值: {nan_count}")
    
    if inf_count == 0 and nan_count == 0:
        print("\n✅ 特征工程模块测试通过！")
    else:
        print("\n⚠️  检测到数据质量问题，需要进一步处理。")
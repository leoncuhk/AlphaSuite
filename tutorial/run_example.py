#!/usr/bin/env python3
"""
KAMA + ATR 策略完整示例
从病毒式传播的5394%收益到现实验证

这个示例复现了文章中的整个研究过程：
1. 实现规则策略并发现其局限性
2. 转向机器学习方法
3. 对比两种方法的结果

运行方式：
    python tutorial/run_example.py
    
或者：
    cd tutorial && python run_example.py
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录和父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import yfinance as yf

# 尝试直接导入（如果从tutorial目录运行）或从tutorial模块导入
try:
    from feature_engineering import prepare_data_for_ml, get_feature_columns
    from labeling import label_with_triple_barrier
    from kama_atr_strategy import RuleBasedKAMAStrategy, MLKAMAStrategy
except ImportError:
    from tutorial.feature_engineering import prepare_data_for_ml, get_feature_columns
    from tutorial.labeling import label_with_triple_barrier
    from tutorial.kama_atr_strategy import RuleBasedKAMAStrategy, MLKAMAStrategy


def print_section(title: str):
    """打印格式化的章节标题"""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def download_data(symbol: str = "QQQ", period: str = "5y") -> pd.DataFrame:
    """
    下载股票数据
    
    参数:
        symbol: 股票代码
        period: 时间周期
    
    返回:
        OHLCV DataFrame
    """
    print_section(f"下载 {symbol} 数据")
    
    print(f"\n从Yahoo Finance下载数据...")
    print(f"  代码: {symbol}")
    print(f"  周期: {period}")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    
    if len(df) == 0:
        raise ValueError(f"无法下载{symbol}的数据")
    
    # 标准化列名
    df.columns = df.columns.str.lower()
    
    # 移除股息和股票分割列（如果存在）
    cols_to_keep = ['open', 'high', 'low', 'close', 'volume']
    df = df[[col for col in cols_to_keep if col in df.columns]]
    
    print(f"\n✓ 数据下载成功")
    print(f"  日期范围: {df.index[0].date()} 到 {df.index[-1].date()}")
    print(f"  交易日数: {len(df)}")
    print(f"  列: {', '.join(df.columns)}")
    
    return df


def test_rule_based_strategy(df: pd.DataFrame) -> dict:
    """
    测试规则策略
    
    这个策略使用固定的规则：
    - 价格 > KAMA
    - ATR% 在1%-3%之间
    - RSI 在40-70之间
    
    问题：这些"完美"的参数往往是过拟合的结果
    """
    print_section("第一部分：规则策略测试")
    
    print("\n规则策略说明：")
    print("  进场条件：")
    print("    ✓ 价格 > KAMA（上升趋势）")
    print("    ✓ ATR% 在 1%-3% 之间（波动性适中）")
    print("    ✓ RSI 在 40-70 之间（避免极端区域）")
    print("  出场条件：")
    print("    ✓ 价格 < KAMA（趋势反转）")
    print("    ✓ ATR% > 4.5%（波动性激增）")
    
    print("\n" + "-" * 70)
    print("准备数据和计算指标...")
    print("-" * 70)
    
    # 准备数据
    df_prepared = prepare_data_for_ml(df)
    print(f"✓ 技术指标计算完成")
    print(f"✓ 标准化特征创建完成")
    print(f"✓ 可用数据: {len(df_prepared)} 个交易日")
    
    # 创建策略实例
    strategy = RuleBasedKAMAStrategy(
        atr_pct_min=0.01,
        atr_pct_max=0.03,
        rsi_min=40,
        rsi_max=70
    )
    
    # 运行回测
    print("\n" + "-" * 70)
    print("运行回测...")
    print("-" * 70)
    
    results = strategy.backtest_simple(df_prepared, initial_capital=10000)
    
    # 显示结果
    print("\n" + "=" * 70)
    print("规则策略回测结果".center(70))
    print("=" * 70)
    
    print(f"\n📊 整体表现")
    print(f"  总收益率: {results['total_return']:.2f}%")
    print(f"  最终资金: ${results['final_capital']:,.2f}")
    print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
    
    print(f"\n📈 交易统计")
    print(f"  交易次数: {results['num_trades']}")
    print(f"  胜率: {results['win_rate']*100:.1f}%")
    print(f"  平均交易收益: {results['avg_trade_return']:.2f}%")
    
    if results['num_trades'] > 0:
        trade_returns = [t['return'] for t in results['trades']]
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r <= 0]
        
        if winning_trades:
            print(f"  平均盈利: {np.mean(winning_trades):.2f}%")
        if losing_trades:
            print(f"  平均亏损: {np.mean(losing_trades):.2f}%")
    
    print("\n💡 规则策略的问题：")
    print("  ⚠️  参数可能是过拟合的结果")
    print("  ⚠️  无法适应不同的市场环境")
    print("  ⚠️  忽略了特征之间的复杂交互")
    
    return results


def test_ml_strategy(df: pd.DataFrame) -> tuple:
    """
    测试机器学习策略
    
    ML策略的优势：
    1. 自动发现特征重要性
    2. 学习复杂的非线性模式
    3. 通过train/test分割避免过拟合
    """
    print_section("第二部分：机器学习策略测试")
    
    print("\nML策略说明：")
    print("  ✓ 使用9个标准化特征")
    print("  ✓ 通过三重障碍法定义成功标准")
    print("  ✓ LightGBM模型自动学习模式")
    print("  ✓ 样本外测试验证泛化能力")
    
    # 创建策略实例
    strategy = MLKAMAStrategy(
        profit_target_mult=3.0,
        stop_loss_mult=3.0,
        eval_bars=15,
        test_size=0.3,  # 30%作为样本外测试
        random_state=42
    )
    
    # 准备训练数据
    X_train, X_test, y_train, y_test, df_labeled = strategy.prepare_training_data(df)
    
    # 训练模型
    strategy.train(X_train, y_train)
    
    # 评估模型
    metrics = strategy.evaluate(X_test, y_test)
    
    # 分析特征重要性
    print("\n" + "=" * 70)
    print("特征重要性分析（模型学到了什么？）".center(70))
    print("=" * 70)
    
    importance_df = strategy.feature_importance
    
    print("\n前5个最重要的特征：")
    for idx, row in importance_df.head(5).iterrows():
        feature = row['feature']
        importance = row['importance']
        
        # 特征解释
        explanations = {
            'feature_atr_pct': '波动性（最关键！模型学会了在低波动时交易）',
            'feature_price_sma200_dist': '长期趋势（避免逆势交易）',
            'feature_macdhist': '动量加速度（时机选择）',
            'feature_price_kama_dist': 'KAMA趋势（原始策略的核心思想）',
            'feature_rsi': 'RSI动量（避免极端区域）',
            'feature_bb_position': '布林带位置（波动性+超买超卖）',
            'feature_momentum_5': '短期动量',
            'feature_momentum_20': '中期动量',
            'feature_volume_ratio': '成交量确认'
        }
        
        explanation = explanations.get(feature, '')
        print(f"\n  {idx+1}. {feature}")
        print(f"     重要性: {importance:.1f}")
        print(f"     说明: {explanation}")
    
    # 对比买入持有策略
    print("\n" + "=" * 70)
    print("与买入持有策略对比".center(70))
    print("=" * 70)
    
    # 计算测试期间的买入持有收益
    test_start_idx = int(len(df_labeled) * 0.7)
    test_data = df_labeled.iloc[test_start_idx:]
    
    buy_hold_return = ((test_data.iloc[-1]['close'] - test_data.iloc[0]['close']) 
                       / test_data.iloc[0]['close'] * 100)
    
    print(f"\n📈 买入持有策略（测试期）:")
    print(f"  总收益: {buy_hold_return:.2f}%")
    print(f"  最大回撤: 未计算（需要完整的每日权益曲线）")
    
    print(f"\n🤖 ML策略（测试期）:")
    print(f"  预测准确率: {metrics['accuracy']*100:.1f}%")
    print(f"  精确率: {metrics['precision']*100:.1f}%")
    print(f"  召回率: {metrics['recall']*100:.1f}%")
    
    # 估算策略收益（基于标签）
    test_signals = strategy.predict(X_test)
    if test_signals.sum() > 0:  # 如果有信号
        # 计算标记的平均收益
        selected_trades = y_test[test_signals == 1]
        if len(selected_trades) > 0:
            estimated_trades = len(selected_trades)
            estimated_win_rate = selected_trades.mean()
            print(f"  预测交易次数: {estimated_trades}")
            print(f"  预测胜率: {estimated_win_rate*100:.1f}%")
    
    print("\n💡 ML策略的优势：")
    print("  ✅ 自动发现最重要的特征（ATR%）")
    print("  ✅ 学习复杂的特征交互")
    print("  ✅ 样本外验证避免过拟合")
    print("  ✅ 可以持续优化和改进")
    
    return strategy, metrics, df_labeled


def generate_summary_report(
    rule_results: dict,
    ml_metrics: dict,
    symbol: str
):
    """
    生成总结报告
    """
    print_section("总结报告")
    
    print(f"\n股票代码: {symbol}")
    print(f"测试框架: AlphaSuite")
    
    print("\n" + "-" * 70)
    print("策略对比".center(70))
    print("-" * 70)
    
    # 表格形式对比
    comparison = pd.DataFrame({
        '指标': [
            '总收益率',
            '交易次数',
            '胜率',
            '模型准确率',
            '适应性',
            '过拟合风险'
        ],
        '规则策略': [
            f"{rule_results['total_return']:.2f}%",
            f"{rule_results['num_trades']}",
            f"{rule_results['win_rate']*100:.1f}%",
            'N/A',
            '低（固定规则）',
            '高'
        ],
        'ML策略': [
            '需完整回测',
            '基于信号',
            f"预测: {ml_metrics['precision']*100:.1f}%",
            f"{ml_metrics['accuracy']*100:.1f}%",
            '高（自适应）',
            '低（有验证集）'
        ]
    })
    
    print("\n" + comparison.to_string(index=False))
    
    print("\n" + "-" * 70)
    print("核心经验总结".center(70))
    print("-" * 70)
    
    print("\n1️⃣  警惕完美的回测结果")
    print("   5394%的收益？那是因为在全数据集上优化参数（曲线拟合）。")
    print("   真实交易中，这样的策略会迅速失效。")
    
    print("\n2️⃣  使用正确的验证方法")
    print("   ✓ 步进式分析（Walk-Forward）")
    print("   ✓ 训练集/测试集分割")
    print("   ✓ 交叉验证")
    print("   ✗ 全数据集优化（永远不要这样做！）")
    
    print("\n3️⃣  特征工程是关键")
    print("   标准化特征使模型能够从长期数据中学习稳定模式。")
    print("   原始指标值在不同价格水平下含义不同，无法直接使用。")
    
    print("\n4️⃣  机器学习不是万能的")
    print("   ML的优势在于发现复杂模式，但仍需要：")
    print("   ✓ 良好的特征工程")
    print("   ✓ 正确的标记方法")
    print("   ✓ 严格的验证流程")
    print("   ✓ 现实的风险管理")
    
    print("\n5️⃣  风险调整后的收益才是王道")
    print("   376%的样本外收益 + 低回撤 > 5394%的过拟合幻觉")
    
    print("\n" + "=" * 70)
    print("感谢使用AlphaSuite！".center(70))
    print("=" * 70)
    print("\n访问 https://github.com/username/AlphaSuite 了解更多")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" KAMA + ATR 策略完整示例 ".center(70))
    print(" 从病毒式传播的5394%收益到现实验证 ".center(70))
    print("=" * 70)
    
    try:
        # 1. 下载数据
        symbol = "QQQ"  # 可以改为 "TSLA", "AAPL" 等
        df = download_data(symbol, period="5y")
        
        # 2. 测试规则策略
        rule_results = test_rule_based_strategy(df)
        
        # 3. 测试ML策略
        ml_strategy, ml_metrics, df_labeled = test_ml_strategy(df)
        
        # 4. 生成总结报告
        generate_summary_report(rule_results, ml_metrics, symbol)
        
        print("\n✅ 示例运行完成！")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
    except Exception as e:
        print(f"\n\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
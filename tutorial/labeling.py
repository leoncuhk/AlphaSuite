"""
三重障碍法（Triple Barrier Method）标记模块

这个方法是机器学习交易策略的核心：它定义了什么是"成功的交易"。

三个障碍：
1. 止盈目标（上障碍）：价格上涨到目标位
2. 止损（下障碍）：价格下跌到止损位
3. 时间限制（时间障碍）：如果前两个障碍都未触及，到期平仓

第一个触及的障碍决定了交易的标签：
- 触及止盈 = 1（成功）
- 触及止损 = 0（失败）
- 超时 = 0（失败，因为资金被占用但没有足够收益）
"""

import pandas as pd
import numpy as np
from typing import Tuple


def label_with_triple_barrier(
    df: pd.DataFrame,
    profit_target_mult: float = 3.0,
    stop_loss_mult: float = 3.0,
    eval_bars: int = 15,
    atr_column: str = 'atr'
) -> pd.DataFrame:
    """
    使用三重障碍法为每个潜在交易点创建标签
    
    工作原理：
    1. 对于每个交易日，设定进场价格
    2. 基于ATR计算止盈和止损位
    3. 向前查看最多eval_bars个交易日
    4. 判断哪个障碍先被触及
    
    参数:
        df: 包含价格和ATR的DataFrame
        profit_target_mult: 止盈目标（ATR的倍数），默认3倍
        stop_loss_mult: 止损（ATR的倍数），默认3倍
        eval_bars: 最大持仓天数，默认15天
        atr_column: ATR列名
    
    返回:
        添加了以下列的DataFrame：
        - 'label': 二元标签（1=成功，0=失败）
        - 'barrier_hit': 触及的障碍类型（'profit', 'stop', 'timeout'）
        - 'hold_days': 实际持仓天数
        - 'return_pct': 实际收益率（百分比）
    """
    df = df.copy()
    
    # 初始化结果列
    labels = []
    barriers_hit = []
    hold_days_list = []
    returns_pct = []
    
    print(f"开始标记数据：{len(df)}个交易日")
    print(f"参数设置：止盈={profit_target_mult}x ATR, 止损={stop_loss_mult}x ATR, 最大持仓={eval_bars}天")
    
    # 遍历每个可能的进场点（除了最后eval_bars个，因为没有足够的未来数据）
    for i in range(len(df) - eval_bars):
        # 当前进场点的信息
        entry_price = df.iloc[i]['close']
        atr_at_entry = df.iloc[i][atr_column]
        
        # 计算三个障碍的位置
        # 障碍1：止盈目标（上障碍）
        profit_target = entry_price + (atr_at_entry * profit_target_mult)
        
        # 障碍2：止损（下障碍）
        stop_loss = entry_price - (atr_at_entry * stop_loss_mult)
        
        # 障碍3：时间限制由eval_bars定义
        
        # 查看未来的价格走势
        future_bars = df.iloc[i+1:i+1+eval_bars]
        
        # 跟踪结果
        hit_profit = False
        hit_stop = False
        actual_hold_days = eval_bars  # 默认持有到超时
        exit_price = future_bars.iloc[-1]['close']  # 默认退出价格
        
        # 遍历未来的每个交易日，检查是否触及障碍
        for j, (idx, row) in enumerate(future_bars.iterrows(), start=1):
            # 检查是否触及止盈（当日最高价 >= 止盈目标）
            if row['high'] >= profit_target:
                hit_profit = True
                actual_hold_days = j
                exit_price = profit_target  # 假设在止盈位退出
                break
            
            # 检查是否触及止损（当日最低价 <= 止损位）
            if row['low'] <= stop_loss:
                hit_stop = True
                actual_hold_days = j
                exit_price = stop_loss  # 假设在止损位退出
                break
        
        # 根据触及的障碍创建标签
        if hit_profit:
            labels.append(1)  # 成功交易
            barriers_hit.append('profit')
        elif hit_stop:
            labels.append(0)  # 失败交易（止损）
            barriers_hit.append('stop')
        else:
            labels.append(0)  # 失败交易（超时，没有足够收益）
            barriers_hit.append('timeout')
        
        # 记录持仓天数和实际收益率
        hold_days_list.append(actual_hold_days)
        return_pct = ((exit_price - entry_price) / entry_price) * 100
        returns_pct.append(return_pct)
    
    # 为最后的eval_bars个交易日填充NaN（没有足够的未来数据来标记）
    labels.extend([np.nan] * eval_bars)
    barriers_hit.extend([None] * eval_bars)
    hold_days_list.extend([np.nan] * eval_bars)
    returns_pct.extend([np.nan] * eval_bars)
    
    # 添加到DataFrame
    df['label'] = labels
    df['barrier_hit'] = barriers_hit
    df['hold_days'] = hold_days_list
    df['return_pct'] = returns_pct
    
    # 打印标记统计
    print("\n标记结果统计：")
    valid_labels = df['label'].dropna()
    if len(valid_labels) > 0:
        print(f"  总标记数: {len(valid_labels)}")
        print(f"  成功交易: {valid_labels.sum()} ({valid_labels.mean()*100:.1f}%)")
        print(f"  失败交易: {len(valid_labels) - valid_labels.sum()} ({(1-valid_labels.mean())*100:.1f}%)")
        
        # 按障碍类型统计
        barrier_counts = df['barrier_hit'].value_counts()
        print(f"\n障碍触及统计：")
        for barrier_type, count in barrier_counts.items():
            if barrier_type:  # 排除None
                print(f"  {barrier_type}: {count} ({count/len(valid_labels)*100:.1f}%)")
        
        # 平均持仓天数
        avg_hold_days = df['hold_days'].mean()
        print(f"\n平均持仓天数: {avg_hold_days:.1f}天")
        
        # 平均收益率（按标签分组）
        avg_return_winners = df[df['label'] == 1]['return_pct'].mean()
        avg_return_losers = df[df['label'] == 0]['return_pct'].mean()
        print(f"\n平均收益率：")
        print(f"  成功交易: {avg_return_winners:.2f}%")
        print(f"  失败交易: {avg_return_losers:.2f}%")
    
    return df


def optimize_barrier_parameters(
    df: pd.DataFrame,
    profit_multiples: list = [2.0, 3.0, 4.0],
    stop_multiples: list = [2.0, 3.0, 4.0],
    eval_bars_list: list = [10, 15, 20]
) -> Tuple[float, float, int, pd.DataFrame]:
    """
    优化三重障碍法的参数
    
    通过测试不同的参数组合，找到产生最佳成功率和风险收益比的参数。
    
    注意：这应该只在训练集上进行！不要在测试集上优化参数。
    
    参数:
        df: 包含价格和ATR的DataFrame
        profit_multiples: 要测试的止盈倍数列表
        stop_multiples: 要测试的止损倍数列表
        eval_bars_list: 要测试的持仓天数列表
    
    返回:
        (最优止盈倍数, 最优止损倍数, 最优持仓天数, 结果DataFrame)
    """
    print("=" * 60)
    print("优化三重障碍法参数")
    print("=" * 60)
    
    results = []
    
    # 测试所有参数组合
    total_combinations = len(profit_multiples) * len(stop_multiples) * len(eval_bars_list)
    current = 0
    
    for profit_mult in profit_multiples:
        for stop_mult in stop_multiples:
            for eval_bars in eval_bars_list:
                current += 1
                print(f"\n测试组合 {current}/{total_combinations}: "
                      f"止盈={profit_mult}x, 止损={stop_mult}x, 天数={eval_bars}")
                
                # 使用当前参数标记数据
                df_labeled = label_with_triple_barrier(
                    df, 
                    profit_target_mult=profit_mult,
                    stop_loss_mult=stop_mult,
                    eval_bars=eval_bars
                )
                
                # 计算关键指标
                valid_labels = df_labeled['label'].dropna()
                if len(valid_labels) > 0:
                    success_rate = valid_labels.mean()
                    avg_winner = df_labeled[df_labeled['label'] == 1]['return_pct'].mean()
                    avg_loser = df_labeled[df_labeled['label'] == 0]['return_pct'].mean()
                    
                    # 风险收益比：平均盈利 / 平均亏损的绝对值
                    risk_reward_ratio = avg_winner / abs(avg_loser) if avg_loser != 0 else 0
                    
                    # 期望值：(成功率 * 平均盈利) + (失败率 * 平均亏损)
                    expected_return = (success_rate * avg_winner) + ((1 - success_rate) * avg_loser)
                    
                    results.append({
                        'profit_mult': profit_mult,
                        'stop_mult': stop_mult,
                        'eval_bars': eval_bars,
                        'success_rate': success_rate,
                        'avg_winner': avg_winner,
                        'avg_loser': avg_loser,
                        'risk_reward': risk_reward_ratio,
                        'expected_return': expected_return,
                        'total_trades': len(valid_labels)
                    })
    
    # 转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('expected_return', ascending=False)
    
    print("\n" + "=" * 60)
    print("优化结果（按期望收益排序）：")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # 返回最优参数
    best = results_df.iloc[0]
    print("\n" + "=" * 60)
    print("最优参数：")
    print(f"  止盈倍数: {best['profit_mult']}x ATR")
    print(f"  止损倍数: {best['stop_mult']}x ATR")
    print(f"  持仓天数: {best['eval_bars']}天")
    print(f"  成功率: {best['success_rate']*100:.1f}%")
    print(f"  期望收益: {best['expected_return']:.2f}%")
    print("=" * 60)
    
    return best['profit_mult'], best['stop_mult'], int(best['eval_bars']), results_df


if __name__ == "__main__":
    # 测试代码
    import yfinance as yf
    from feature_engineering import prepare_data_for_ml
    
    print("=" * 60)
    print("三重障碍法标记模块测试")
    print("=" * 60)
    
    # 下载测试数据
    print("\n下载QQQ测试数据...")
    ticker = yf.Ticker("QQQ")
    df = ticker.history(period="2y")
    df.columns = df.columns.str.lower()
    
    # 准备数据（计算ATR等指标）
    print("\n准备数据...")
    df = prepare_data_for_ml(df)
    
    # 测试1：使用默认参数标记
    print("\n" + "=" * 60)
    print("测试1：使用默认参数标记")
    print("=" * 60)
    df_labeled = label_with_triple_barrier(df)
    
    # 测试2：参数优化（使用小范围快速测试）
    print("\n" + "=" * 60)
    print("测试2：参数优化")
    print("=" * 60)
    best_profit, best_stop, best_days, results = optimize_barrier_parameters(
        df,
        profit_multiples=[2.0, 3.0],
        stop_multiples=[2.0, 3.0],
        eval_bars_list=[10, 15]
    )
    
    print("\n✅ 三重障碍法标记模块测试完成！")
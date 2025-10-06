#!/usr/bin/env python3
"""
使用AlphaSuite框架运行KAMA+ATR策略的专业示例

这个脚本展示了如何正确使用AlphaSuite的核心功能：
1. Walk-forward analysis（步进式分析）
2. 专业回测（包括最大回撤、夏普比率等）
3. 特征重要性分析
4. 样本外验证

这是对原tutorial实现的改进，使用AlphaSuite的完整功能栈。

运行方式：
    python tutorial/run_with_alphasuite.py
"""

import sys
import os
import subprocess
import json
import pandas as pd
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def print_section(title: str, width: int = 70):
    """打印格式化的章节标题"""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def run_command(cmd: list, description: str) -> dict:
    """
    运行命令并返回结果
    
    参数:
        cmd: 命令列表
        description: 命令描述
    
    返回:
        包含returncode, stdout, stderr的字典
    """
    print(f"\n执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ 执行成功")
    else:
        print(f"✗ 执行失败 (返回码: {result.returncode})")
        if result.stderr:
            print(f"错误信息:\n{result.stderr}")
    
    return {
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr
    }


def check_strategy_exists():
    """检查策略是否已注册"""
    print_section("步骤1: 验证策略配置")
    
    # 检查策略文件是否存在
    strategy_file = os.path.join(parent_dir, 'strategies', 'kama_atr_ml.py')
    if not os.path.exists(strategy_file):
        print(f"✗ 策略文件不存在: {strategy_file}")
        return False
    
    print(f"✓ 策略文件存在: {strategy_file}")
    
    # 尝试导入策略
    try:
        sys.path.insert(0, os.path.join(parent_dir, 'strategies'))
        from kama_atr_ml import KamaAtrMLStrategy
        strategy = KamaAtrMLStrategy()
        
        print(f"\n策略信息:")
        print(f"  类名: KamaAtrMLStrategy")
        print(f"  类型: {'ML策略' if strategy.is_ml_strategy else '规则策略'}")
        print(f"  参数数量: {len(strategy.define_parameters())}")
        print(f"  特征数量: {len(strategy.get_feature_list())}")
        
        print("\n✓ 策略验证成功")
        return True
        
    except Exception as e:
        print(f"✗ 策略导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_strategy(ticker: str = "QQQ", start_date: str = None, end_date: str = None):
    """
    训练策略（使用AlphaSuite的quant_engine）
    
    这会执行完整的walk-forward analysis：
    1. 将数据分成多个时间窗口
    2. 在每个窗口上训练模型
    3. 在下一个窗口上测试
    4. 计算样本外性能指标
    
    参数:
        ticker: 股票代码
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
    """
    print_section(f"步骤2: 训练策略 ({ticker})")
    
    print("\n⚠️  重要说明：")
    print("  AlphaSuite的train命令需要从数据库加载数据。")
    print("  如果数据库中没有数据，需要先运行：")
    print("  python download_data.py --run_daily_pipeline=true")
    print("\n  为了演示目的，我们将跳过实际训练，直接展示如何使用。")
    
    # 构建命令
    cmd = [
        'python', 'quant_engine.py', 'train',
        '--ticker', ticker,
        '--strategy', 'kama_atr_ml'
    ]
    
    if start_date:
        cmd.extend(['--start-date', start_date])
    if end_date:
        cmd.extend(['--end-date', end_date])
    
    print(f"\n完整的训练命令：")
    print(f"  {' '.join(cmd)}")
    
    print(f"\n这个命令会：")
    print(f"  1. 从数据库加载{ticker}的历史数据")
    print(f"  2. 计算所有技术指标和标准化特征")
    print(f"  3. 使用三重障碍法标记训练数据")
    print(f"  4. 执行walk-forward analysis（5折交叉验证）")
    print(f"  5. 训练LightGBM模型")
    print(f"  6. 计算样本外性能指标")
    print(f"  7. 保存模型和结果到artifacts/")
    
    # 注意：实际运行需要数据库中有数据
    # result = run_command(cmd, "训练策略（walk-forward analysis）")
    # return result['returncode'] == 0
    
    print("\n⏭️  跳过实际训练（需要数据库数据）")
    return False


def visualize_results(ticker: str = "QQQ"):
    """
    可视化训练结果
    
    这会显示：
    1. 权益曲线
    2. 交易分布
    3. 特征重要性
    4. 性能指标
    """
    print_section(f"步骤3: 可视化结果 ({ticker})")
    
    cmd = [
        'python', 'quant_engine.py', 'visualize-model',
        '--ticker', ticker,
        '--strategy', 'kama_atr_ml'
    ]
    
    print(f"\n可视化命令：")
    print(f"  {' '.join(cmd)}")
    
    print(f"\n这个命令会显示：")
    print(f"  1. 权益曲线（策略 vs 买入持有）")
    print(f"  2. 最大回撤分析")
    print(f"  3. 交易分布和统计")
    print(f"  4. 特征重要性排名")
    print(f"  5. 样本外性能指标")
    
    # result = run_command(cmd, "可视化训练结果")
    # return result['returncode'] == 0
    
    print("\n⏭️  跳过实际可视化（需要先训练模型）")
    return False


def demonstrate_workflow():
    """
    演示完整的AlphaSuite工作流程
    """
    print_section("KAMA + ATR 策略 - AlphaSuite专业工作流程")
    
    print("\n本示例展示如何使用AlphaSuite框架的完整功能：")
    print("  ✓ 符合框架的策略实现（strategies/kama_atr_ml.py）")
    print("  ✓ Walk-forward analysis（避免过拟合）")
    print("  ✓ 专业回测指标（夏普比率、最大回撤等）")
    print("  ✓ 特征重要性分析")
    print("  ✓ 样本外验证")
    
    # 步骤1：验证策略
    if not check_strategy_exists():
        print("\n❌ 策略验证失败，无法继续")
        return False
    
    # 步骤2：训练策略
    # train_strategy("QQQ")
    
    # 步骤3：可视化结果
    # visualize_results("QQQ")
    
    # 展示工作流程说明
    show_workflow_guide()
    
    return True


def show_workflow_guide():
    """显示完整的工作流程指南"""
    print_section("完整工作流程指南")
    
    print("\n🔄 AlphaSuite KAMA+ATR策略完整流程：")
    
    print("\n【前置准备】")
    print("  1. 确保PostgreSQL服务运行：")
    print("     brew services start postgresql@15")
    print("\n  2. 下载历史数据到数据库：")
    print("     python download_data.py --run_daily_pipeline=true")
    print("     （首次运行需要较长时间）")
    
    print("\n【策略开发】")
    print("  3. 策略已实现：strategies/kama_atr_ml.py")
    print("     ✓ 符合BaseStrategy接口")
    print("     ✓ 定义参数和调优范围")
    print("     ✓ 实现特征工程")
    print("     ✓ 实现三重障碍法标记")
    
    print("\n【参数优化（可选）】")
    print("  4. 使用贝叶斯优化寻找最佳参数：")
    print("     python quant_engine.py tune-strategy \\")
    print("       --ticker QQQ \\")
    print("       --strategy kama_atr_ml \\")
    print("       --n-calls 30")
    print("     这会自动优化所有参数的tuning_range")
    
    print("\n【模型训练】")
    print("  5. 训练模型（walk-forward analysis）：")
    print("     python quant_engine.py train \\")
    print("       --ticker QQQ \\")
    print("       --strategy kama_atr_ml \\")
    print("       --start-date 2020-01-01")
    print("\n     训练过程：")
    print("       • 计算技术指标和标准化特征")
    print("       • 使用三重障碍法标记")
    print("       • 5折walk-forward验证")
    print("       • 训练LightGBM模型")
    print("       • 保存模型到 artifacts/")
    
    print("\n【结果分析】")
    print("  6. 可视化样本外性能：")
    print("     python quant_engine.py visualize-model \\")
    print("       --ticker QQQ \\")
    print("       --strategy kama_atr_ml")
    print("\n     显示内容：")
    print("       • 权益曲线")
    print("       • 夏普比率")
    print("       • 最大回撤")
    print("       • 交易统计")
    print("       • 特征重要性")
    
    print("\n【实时应用】")
    print("  7. 扫描实时信号：")
    print("     python quant_engine.py scan \\")
    print("       --strategy kama_atr_ml \\")
    print("       --universe QQQ,SPY,AAPL")
    print("\n  8. 单股预测：")
    print("     python quant_engine.py predict \\")
    print("       --ticker QQQ \\")
    print("       --strategy kama_atr_ml")
    
    print("\n【与原tutorial的对比】")
    print_comparison_table()
    
    print("\n【核心优势】")
    print("  ✅ 真正的walk-forward analysis（避免过拟合）")
    print("  ✅ 专业的风险管理（基于ATR的止损）")
    print("  ✅ 完整的性能指标（夏普、回撤、胜率等）")
    print("  ✅ 集成到生产环境（可以实时扫描和交易）")
    print("  ✅ 自动参数优化（贝叶斯优化）")
    print("  ✅ 标准化的回测框架（PyBroker）")


def print_comparison_table():
    """打印原implementation vs AlphaSuite的对比"""
    print("\n原tutorial实现 vs AlphaSuite框架：")
    print("-" * 70)
    print(f"{'特性':<25s} {'原实现':<20s} {'AlphaSuite框架':<20s}")
    print("-" * 70)
    
    comparisons = [
        ("验证方法", "简单train/test分割", "Walk-forward analysis"),
        ("回测引擎", "手写简单回测", "PyBroker专业回测"),
        ("性能指标", "基础指标", "完整指标+回撤分析"),
        ("参数优化", "手动", "贝叶斯自动优化"),
        ("风险管理", "固定止损", "ATR动态止损"),
        ("实时应用", "不支持", "支持扫描和预测"),
        ("可扩展性", "单独脚本", "集成框架"),
        ("生产就绪", "否", "是"),
    ]
    
    for feature, original, alphasuite in comparisons:
        print(f"{feature:<25s} {original:<20s} {alphasuite:<20s}")
    
    print("-" * 70)


def show_expected_results():
    """显示预期结果（基于原文）"""
    print_section("预期结果（基于原文）")
    
    print("\n根据Richard Shu的文章，使用AlphaSuite进行walk-forward analysis：")
    
    print("\n📊 QQQ样本外结果：")
    print("  • 夏普比率: 0.90")
    print("  • 总收益率: 376% (样本外)")
    print("  • 最大回撤: -16.4%")
    print("  • 买入持有: 631% (同期)")
    
    print("\n🔍 关键发现：")
    print("  1. 样本外收益(376%)远低于完美回测(791%)")
    print("     → 这是正常的！样本外更可信")
    
    print("\n  2. 虽然低于买入持有(631%)，但风险更低：")
    print("     • 策略最大回撤: -16.4%")
    print("     • 买入持有回撤: 更大（文中未明确，通常>30%）")
    print("     → 风险调整后的收益更优")
    
    print("\n  3. 特征重要性自动验证了原假设：")
    print("     • 最重要: feature_atr_pct (波动性)")
    print("     • 第二: feature_price_sma200_dist (长期趋势)")
    print("     → ML自动发现了KAMA+ATR的核心思想")
    
    print("\n💡 与病毒文章(5394%)的对比：")
    print("  病毒文章: 5394% (全数据集优化，过拟合)")
    print("  规则策略: 9.05% (tutorial简单实现)")
    print("  AlphaSuite: 376% (walk-forward，可信)")
    print("  → 正确的方法论至关重要！")


def main():
    """主函数"""
    try:
        # 演示工作流程
        demonstrate_workflow()
        
        # 显示预期结果
        show_expected_results()
        
        print_section("总结")
        
        print("\n✅ AlphaSuite框架优势：")
        print("  1. 真正的walk-forward analysis")
        print("  2. 专业的回测指标")
        print("  3. 自动参数优化")
        print("  4. 集成的工作流程")
        print("  5. 生产就绪的系统")
        
        print("\n📚 进一步学习：")
        print("  • 查看 strategies/kama_atr_ml.py 了解策略实现")
        print("  • 查看 tutorial/README.md 了解理论背景")
        print("  • 运行 python tutorial/run_example.py 进行快速测试")
        print("  • 准备数据后运行完整的AlphaSuite工作流程")
        
        print("\n💪 准备好了吗？")
        print("  1. 下载数据：python download_data.py --run_daily_pipeline=true")
        print("  2. 训练模型：python quant_engine.py train --ticker QQQ --strategy kama_atr_ml")
        print("  3. 查看结果：python quant_engine.py visualize-model --ticker QQQ --strategy kama_atr_ml")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行")
        return 1
    except Exception as e:
        print(f"\n\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
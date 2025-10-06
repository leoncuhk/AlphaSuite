# KAMA + ATR 策略教程

从病毒式传播的 5394% 收益到现实验证

## 📖 教程说明

这个教程复现了一个经典的量化交易案例研究，展示了：

1. **过拟合的危险**：为什么"完美"的回测结果不可信
2. **正确的验证方法**：如何使用训练/测试分割避免过拟合
3. **特征工程的重要性**：如何创建稳健的机器学习特征
4. **三重障碍法**：如何科学地定义交易成功标准
5. **规则 vs ML**：传统规则策略与机器学习方法的对比

## 📁 文件结构

```
tutorial/
├── __init__.py                 # 模块初始化
├── feature_engineering.py      # 特征工程（9个标准化特征）
├── labeling.py                 # 三重障碍法标记
├── kama_atr_strategy.py        # 策略实现（规则 + ML）
├── run_example.py              # 主运行脚本
└── README.md                   # 本文件
```

## 🚀 快速开始

### 运行完整示例

```bash
# 从项目根目录运行
python tutorial/run_example.py

# 或从 tutorial 目录运行
cd tutorial
python run_example.py
```

### 测试单个模块

```bash
# 测试特征工程
python tutorial/feature_engineering.py

# 测试三重障碍法
python tutorial/labeling.py

# 测试策略
python tutorial/kama_atr_strategy.py
```

## 📊 示例输出

程序会输出：

### 第一部分：规则策略测试

- 使用固定规则（价格 > KAMA，ATR% 在 1%-3%）
- 简单回测结果
- 展示规则策略的局限性

**示例结果**：
- 总收益率：9.05%
- 交易次数：94
- 胜率：35.1%
- 夏普比率：0.19

### 第二部分：机器学习策略

- 自动特征工程（9个标准化特征）
- 三重障碍法标记（定义成功标准）
- LightGBM 模型训练
- 样本外测试验证

**关键发现**：
- 最重要特征：`feature_price_sma200_dist`（长期趋势）
- 第二重要：`feature_atr_pct`（波动性）
- 模型自动验证了原始策略的核心思想

### 第三部分：总结报告

- 策略对比表格
- 核心经验总结
- 5个关键教训

## 🎯 核心概念

### 1. 特征标准化

```python
# ❌ 错误：使用原始值
feature = df['macd_hist']  # $2.00 在不同价格水平下含义不同

# ✅ 正确：标准化为百分比
feature = df['macd_hist'] / df['close']  # 跨时期可比
```

### 2. 三重障碍法

为每个交易设置三个退出条件：
1. **止盈**：价格上涨到目标位（如 3x ATR）
2. **止损**：价格下跌到止损位（如 3x ATR）
3. **超时**：达到最大持仓天数（如 15 天）

第一个触及的障碍决定标签（1=成功，0=失败）

### 3. 避免过拟合

```python
# ❌ 错误：在全数据集上优化
# 结果：5394% 收益（不真实）

# ✅ 正确：训练/测试分割
# 训练集：70% 历史数据
# 测试集：30% 最近数据（模拟真实交易）
# 结果：更低但更可信的收益
```

## 🔍 详细代码说明

### feature_engineering.py

**核心函数**：
- `calculate_technical_indicators()` - 计算 KAMA、ATR、RSI、MACD 等
- `create_normalized_features()` - 创建 9 个标准化特征
- `prepare_data_for_ml()` - 一站式数据准备

**9 个特征**：
1. `feature_atr_pct` - ATR 百分比（波动性）
2. `feature_price_kama_dist` - 价格与 KAMA 的距离
3. `feature_price_sma200_dist` - 价格与 200 日均线的距离
4. `feature_rsi` - 标准化 RSI（0-1）
5. `feature_macdhist` - 标准化 MACD 柱状图
6. `feature_bb_position` - 布林带位置
7. `feature_momentum_5` - 5 日动量
8. `feature_momentum_20` - 20 日动量
9. `feature_volume_ratio` - 成交量比率

### labeling.py

**核心函数**：
- `label_with_triple_barrier()` - 使用三重障碍法标记
- `optimize_barrier_parameters()` - 优化障碍参数

**标记逻辑**：
```python
for 每个交易日:
    设置止盈 = 入场价 + (3 × ATR)
    设置止损 = 入场价 - (3 × ATR)
    
    向前查看最多 15 天:
        if 触及止盈:
            标签 = 1（成功）
        elif 触及止损:
            标签 = 0（失败）
        elif 超时:
            标签 = 0（失败）
```

### kama_atr_strategy.py

**两个策略类**：

1. `RuleBasedKAMAStrategy` - 规则策略
   - 固定的进场/出场规则
   - 简单回测功能
   - 展示过拟合问题

2. `MLKAMAStrategy` - ML 策略
   - 自动特征学习
   - LightGBM 模型
   - 训练/测试分割
   - 特征重要性分析

## 📈 扩展练习

1. **尝试不同的股票**：
   ```python
   # 在 run_example.py 中修改
   symbol = "AAPL"  # 或 "TSLA", "SPY" 等
   ```

2. **调整三重障碍参数**：
   ```python
   ml_strategy = MLKAMAStrategy(
       profit_target_mult=4.0,  # 增加止盈目标
       stop_loss_mult=2.0,      # 降低止损
       eval_bars=20             # 增加最大持仓天数
   )
   ```

3. **添加新特征**：
   在 `feature_engineering.py` 中添加自己的特征

4. **优化模型参数**：
   在 `kama_atr_strategy.py` 中调整 LightGBM 参数

## 💡 关键经验

1. **警惕完美的回测结果**
   - 5394% 的收益？那是曲线拟合
   - 真实交易中会迅速失效

2. **使用正确的验证方法**
   - ✅ 训练/测试分割
   - ✅ 步进式分析
   - ❌ 全数据集优化

3. **特征工程是关键**
   - 标准化使特征跨时期可比
   - 原始值在不同价格水平下无意义

4. **机器学习不是万能的**
   - 需要良好的特征工程
   - 需要正确的标记方法
   - 需要严格的验证流程

5. **风险调整后的收益才重要**
   - 376% + 低回撤 > 5394% 过拟合幻觉

## 🔗 参考资源

- [原文：We Backtested a Viral Trading Strategy](https://medium.com/@richardshu)
- [AlphaSuite GitHub](https://github.com/username/AlphaSuite)
- [TA-Lib 文档](https://ta-lib.org/)
- [LightGBM 文档](https://lightgbm.readthedocs.io/)

## 📝 许可

本教程是 AlphaSuite 项目的一部分，遵循 MIT 许可证。
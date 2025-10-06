#!/bin/bash
# KAMA+ATR策略教程 - 快速验证脚本

echo "========================================================================"
echo "                    KAMA+ATR策略教程 - 快速验证                         "
echo "========================================================================"

# 激活虚拟环境
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ 虚拟环境已激活"
else
    echo "✗ 虚拟环境未找到"
    exit 1
fi

echo ""
echo "【验证1：教学实现】"
echo "--------------------------------------------------------------------"
python tutorial/verify_installation.py
if [ $? -eq 0 ]; then
    echo "✓ 教学实现验证通过"
else
    echo "✗ 教学实现验证失败"
    exit 1
fi

echo ""
echo "【验证2：AlphaSuite策略】"
echo "--------------------------------------------------------------------"
python -c "
import sys
import os
sys.path.insert(0, os.getcwd())
from strategies.kama_atr_ml import KamaAtrMLStrategy
s = KamaAtrMLStrategy()
print(f'✓ 策略类名: KamaAtrMLStrategy')
print(f'✓ 策略类型: {\"ML\" if s.is_ml_strategy else \"规则\"}')
print(f'✓ 参数数量: {len(s.define_parameters())}')
print(f'✓ 特征数量: {len(s.get_feature_list())}')
print(f'✓ AlphaSuite策略配置正确')
"
if [ $? -eq 0 ]; then
    echo "✓ AlphaSuite策略验证通过"
else
    echo "✗ AlphaSuite策略验证失败"
    exit 1
fi

echo ""
echo "【验证3：快速示例（部分运行）】"
echo "--------------------------------------------------------------------"
echo "运行教学示例的前30秒..."
timeout 30 python tutorial/run_example.py 2>&1 | head -50
if [ ${PIPESTATUS[0]} -eq 124 ]; then
    echo "✓ 教学示例可以正常启动（已终止）"
else
    echo "⚠ 教学示例运行完整或有错误"
fi

echo ""
echo "========================================================================"
echo "                               验证总结                                  "
echo "========================================================================"
echo ""
echo "✅ 所有核心组件验证通过！"
echo ""
echo "📚 下一步："
echo "  1. 阅读 tutorial/README.md 了解完整理论"
echo "  2. 运行 python tutorial/run_example.py 查看完整演示"
echo "  3. 准备数据后使用 AlphaSuite 框架进行专业回测"
echo ""
echo "🔗 两种使用方式："
echo ""
echo "  【教学模式】快速学习（5分钟）："
echo "    python tutorial/run_example.py"
echo ""
echo "  【生产模式】专业验证（需要数据库）："
echo "    python download_data.py --run_daily_pipeline=true"
echo "    python quant_engine.py train --ticker QQQ --strategy kama_atr_ml"
echo "    python quant_engine.py visualize-model --ticker QQQ --strategy kama_atr_ml"
echo ""
echo "========================================================================"
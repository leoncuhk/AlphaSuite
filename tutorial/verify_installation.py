#!/usr/bin/env python3
"""
验证 tutorial 模块是否正确安装和配置

运行方式：
    python tutorial/verify_installation.py
"""

import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def verify_imports():
    """验证所有模块能否正确导入"""
    print("=" * 60)
    print("验证模块导入")
    print("=" * 60)
    
    modules = [
        ('feature_engineering', [
            'prepare_data_for_ml', 
            'get_feature_columns',
            'calculate_kama',
            'calculate_atr'
        ]),
        ('labeling', [
            'label_with_triple_barrier',
            'optimize_barrier_parameters'
        ]),
        ('kama_atr_strategy', [
            'RuleBasedKAMAStrategy',
            'MLKAMAStrategy'
        ])
    ]
    
    all_success = True
    
    for module_name, items in modules:
        try:
            module = __import__(module_name)
            print(f"\n✓ {module_name}")
            
            for item in items:
                if hasattr(module, item):
                    print(f"  ✓ {item}")
                else:
                    print(f"  ✗ {item} (未找到)")
                    all_success = False
                    
        except ImportError as e:
            print(f"\n✗ {module_name} 导入失败: {e}")
            all_success = False
    
    return all_success


def verify_dependencies():
    """验证依赖包"""
    print("\n" + "=" * 60)
    print("验证依赖包")
    print("=" * 60)
    
    dependencies = [
        'pandas',
        'numpy',
        'talib',
        'yfinance',
        'lightgbm',
        'sklearn'
    ]
    
    all_success = True
    
    for dep in dependencies:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {dep:15s} {version}")
        except ImportError:
            print(f"✗ {dep:15s} 未安装")
            all_success = False
    
    return all_success


def verify_files():
    """验证文件存在"""
    print("\n" + "=" * 60)
    print("验证文件")
    print("=" * 60)
    
    required_files = [
        '__init__.py',
        'feature_engineering.py',
        'labeling.py',
        'kama_atr_strategy.py',
        'run_example.py',
        'README.md'
    ]
    
    all_success = True
    
    for filename in required_files:
        filepath = os.path.join(current_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {filename:30s} ({size:,} bytes)")
        else:
            print(f"✗ {filename:30s} (不存在)")
            all_success = False
    
    return all_success


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("KAMA + ATR 策略教程 - 安装验证")
    print("=" * 60)
    
    results = []
    
    # 验证文件
    results.append(("文件检查", verify_files()))
    
    # 验证依赖
    results.append(("依赖包检查", verify_dependencies()))
    
    # 验证导入
    results.append(("模块导入检查", verify_imports()))
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name:20s} {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✅ 所有检查通过！")
        print("\n可以运行示例：")
        print("  python tutorial/run_example.py")
        return 0
    else:
        print("\n⚠️  部分检查失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
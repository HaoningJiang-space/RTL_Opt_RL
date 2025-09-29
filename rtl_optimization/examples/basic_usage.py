#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTL优化系统基本使用示例
Basic usage example for RTL optimization system

该示例展示如何使用RTL优化系统进行基本的RTL网表优化训练和评估。
This example demonstrates how to use RTL optimization system for basic
RTL netlist optimization training and evaluation.
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径 / Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入RTL优化系统 / Import RTL optimization system
from rtl_optimization import (
    get_config,
    create_optimization_pipeline,
    quick_start,
    get_system_info
)


def setup_logging():
    """设置日志系统 / Setup logging system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rtl_optimization.log')
        ]
    )


def collect_rtl_files(benchmark_dir: str) -> list:
    """收集RTL文件 / Collect RTL files

    Args:
        benchmark_dir: 基准测试目录 / Benchmark directory

    Returns:
        list: RTL文件路径列表 / List of RTL file paths
    """
    rtl_files = []

    # 支持的文件扩展名 / Supported file extensions
    extensions = ['.v', '.verilog', '.aig']

    for ext in extensions:
        pattern = f"**/*{ext}"
        files = list(Path(benchmark_dir).glob(pattern))
        rtl_files.extend([str(f) for f in files])

    logging.info(f"找到 {len(rtl_files)} 个RTL文件 / Found {len(rtl_files)} RTL files")
    return rtl_files


def basic_training_example():
    """基本训练示例 / Basic training example"""
    print("=== RTL优化系统基本训练示例 / Basic Training Example ===")

    # 1. 系统信息检查 / System information check
    print("\n1. 系统信息检查 / System Information Check:")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")

    if not system_info.get("components_available", False):
        print("❌ 系统组件不可用，请检查依赖安装 / System components not available")
        return

    # 2. 收集RTL文件 / Collect RTL files
    print("\n2. 收集RTL文件 / Collecting RTL files:")
    benchmark_dir = "../short_benchmark"  # 相对于examples目录

    if not os.path.exists(benchmark_dir):
        print(f"❌ 基准目录不存在: {benchmark_dir}")
        print("请确保RTL文件在正确的目录中 / Please ensure RTL files are in correct directory")
        return

    rtl_files = collect_rtl_files(benchmark_dir)

    if not rtl_files:
        print("❌ 未找到RTL文件 / No RTL files found")
        return

    print(f"✓ 收集到 {len(rtl_files)} 个RTL文件")

    # 3. 加载配置 / Load configuration
    print("\n3. 加载配置 / Loading configuration:")
    config = get_config()
    print(f"✓ 配置加载完成 / Configuration loaded")
    print(f"  状态维度: {config.rl.state_dim}")
    print(f"  动作维度: {config.rl.action_dim}")
    print(f"  最大步数: {config.rl.max_steps_per_episode}")

    # 4. 创建优化流水线 / Create optimization pipeline
    print("\n4. 创建优化流水线 / Creating optimization pipeline:")
    try:
        pipeline = create_optimization_pipeline(rtl_files[:5])  # 使用前5个文件进行快速测试
        print("✓ 优化流水线创建成功 / Optimization pipeline created successfully")

        # 打印环境信息 / Print environment info
        env_stats = pipeline["environment"].get_environment_statistics()
        print(f"  环境统计: {env_stats}")

    except Exception as e:
        print(f"❌ 优化流水线创建失败 / Pipeline creation failed: {e}")
        return

    # 5. 简短训练演示 / Short training demonstration
    print("\n5. 简短训练演示 / Short training demonstration:")
    try:
        # 训练几个回合作为演示 / Train a few episodes for demonstration
        training_result = pipeline["agent"].train(num_episodes=5)

        print(f"✓ 训练完成 / Training completed")
        print(f"  训练回合数: {len(training_result)}")

        if training_result:
            avg_reward = sum(r["reward"] for r in training_result) / len(training_result)
            print(f"  平均奖励: {avg_reward:.3f}")
            print(f"  最佳奖励: {max(r['reward'] for r in training_result):.3f}")

    except Exception as e:
        print(f"❌ 训练失败 / Training failed: {e}")
        return

    # 6. 评估演示 / Evaluation demonstration
    print("\n6. 评估演示 / Evaluation demonstration:")
    try:
        eval_result = pipeline["agent"].evaluate(num_episodes=2)
        print(f"✓ 评估完成 / Evaluation completed")
        print(f"  平均奖励: {eval_result['mean_reward']:.3f}")
        print(f"  标准差: {eval_result['std_reward']:.3f}")

    except Exception as e:
        print(f"❌ 评估失败 / Evaluation failed: {e}")

    print("\n=== 示例完成 / Example completed ===")


def quick_start_example():
    """快速开始示例 / Quick start example"""
    print("\n=== 快速开始示例 / Quick Start Example ===")

    # 收集RTL文件 / Collect RTL files
    benchmark_dir = "../short_benchmark"

    if not os.path.exists(benchmark_dir):
        print(f"❌ 基准目录不存在: {benchmark_dir}")
        return

    rtl_files = collect_rtl_files(benchmark_dir)

    if not rtl_files:
        print("❌ 未找到RTL文件 / No RTL files found")
        return

    try:
        # 使用便捷函数快速开始 / Use convenience function for quick start
        result = quick_start(
            rtl_files=rtl_files[:3],  # 使用前3个文件
            num_episodes=10,          # 训练10个回合
            config_path=None          # 使用默认配置
        )

        print("✓ 快速训练完成 / Quick training completed")
        print(f"  训练历史长度: {len(result['training_history'])}")
        print(f"  最终统计: {result['final_stats']['episodes']} 回合")

    except Exception as e:
        print(f"❌ 快速开始失败 / Quick start failed: {e}")


def individual_component_example():
    """单独组件使用示例 / Individual component usage example"""
    print("\n=== 单独组件使用示例 / Individual Component Example ===")

    try:
        # 1. ABC接口示例 / ABC interface example
        from rtl_optimization.tools import ABCInterface

        print("\n1. ABC接口测试 / ABC Interface Test:")
        with ABCInterface() as abc:
            print("✓ ABC接口初始化成功 / ABC interface initialized")

            # 这里可以测试具体的ABC功能 / Can test specific ABC functions here
            # 需要实际的RTL文件 / Need actual RTL files

        # 2. 图构建示例 / Graph building example
        from rtl_optimization.graph import RTLNetlistGraph

        print("\n2. 图构建测试 / Graph Building Test:")
        graph_builder = RTLNetlistGraph()
        print("✓ 图构建器初始化成功 / Graph builder initialized")
        print(f"  节点类型: {list(graph_builder.node_type_mapping.keys())}")

        # 3. GNN模型示例 / GNN model example
        from rtl_optimization.graph import RTLOptimizationGNN

        print("\n3. GNN模型测试 / GNN Model Test:")
        gnn_model = RTLOptimizationGNN()
        model_info = gnn_model.get_model_info()
        print(f"✓ GNN模型创建成功 / GNN model created")
        print(f"  参数数量: {model_info['total_parameters']}")

    except Exception as e:
        print(f"❌ 组件测试失败 / Component test failed: {e}")


def main():
    """主函数 / Main function"""
    setup_logging()

    print("RTL优化系统使用示例 / RTL Optimization System Usage Examples")
    print("=" * 60)

    # 基本训练示例 / Basic training example
    basic_training_example()

    # 快速开始示例 / Quick start example
    quick_start_example()

    # 单独组件示例 / Individual component example
    individual_component_example()

    print("\n所有示例执行完成 / All examples completed")


if __name__ == "__main__":
    main()
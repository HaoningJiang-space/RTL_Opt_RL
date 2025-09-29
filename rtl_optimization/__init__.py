#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTL优化系统主模块
RTL optimization system main module

该包提供完整的RTL网表优化解决方案，基于图神经网络和强化学习技术。
This package provides complete RTL netlist optimization solution based on
graph neural networks and reinforcement learning techniques.
"""

__version__ = "1.0.0"
__author__ = "RTL Optimization Team"
__description__ = "RTL网表优化系统 / RTL Netlist Optimization System"

# 导入核心组件 / Import core components
try:
    from .utils.config import Config, get_config
    from .tools.abc_interface import ABCInterface
    from .tools.evaluator import HybridEvaluationSystem
    from .graph.rtl_graph import RTLNetlistGraph
    from .graph.gnn_model import RTLOptimizationGNN, create_rtl_gnn_model
    from .rl.environment import RTLOptimizationEnvironment, create_rtl_environment
    from .rl.ppo_agent import RTLOptimizationPPO, create_ppo_agent

    # 标记可用组件 / Mark available components
    _COMPONENTS_AVAILABLE = True

except ImportError as e:
    # 如果某些依赖不可用，标记为不可用 / Mark as unavailable if some dependencies are missing
    _COMPONENTS_AVAILABLE = False
    import warnings
    warnings.warn(f"某些组件不可用，可能缺少依赖: {e}")

# 导出主要接口 / Export main interfaces
__all__ = [
    # 配置 / Configuration
    "Config",
    "get_config",

    # 工具 / Tools
    "ABCInterface",
    "HybridEvaluationSystem",

    # 图表示和模型 / Graph representation and models
    "RTLNetlistGraph",
    "RTLOptimizationGNN",
    "create_rtl_gnn_model",

    # 强化学习 / Reinforcement learning
    "RTLOptimizationEnvironment",
    "create_rtl_environment",
    "RTLOptimizationPPO",
    "create_ppo_agent",

    # 版本信息 / Version info
    "__version__",
    "__author__",
    "__description__"
]


def get_system_info():
    """获取系统信息 / Get system information"""
    import sys
    import platform

    info = {
        "version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "components_available": _COMPONENTS_AVAILABLE
    }

    if _COMPONENTS_AVAILABLE:
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
        except ImportError:
            info["torch_version"] = "Not installed"
            info["cuda_available"] = False

        try:
            import torch_geometric
            info["torch_geometric_version"] = torch_geometric.__version__
        except ImportError:
            info["torch_geometric_version"] = "Not installed"

    return info


def create_optimization_pipeline(rtl_dataset, config_path=None):
    """创建完整的RTL优化流水线 / Create complete RTL optimization pipeline

    这是一个便捷函数，用于快速设置完整的RTL优化环境。
    This is a convenience function for quickly setting up complete RTL optimization environment.

    Args:
        rtl_dataset: RTL数据集文件列表 / List of RTL dataset files
        config_path: 配置文件路径 / Configuration file path

    Returns:
        dict: 包含环境、代理和配置的字典 / Dictionary containing environment, agent and config
    """
    if not _COMPONENTS_AVAILABLE:
        raise RuntimeError("系统组件不可用，请检查依赖安装 / System components not available, please check dependencies")

    # 加载配置 / Load configuration
    config = get_config(config_path)

    # 创建GNN模型 / Create GNN model
    gnn_model = create_rtl_gnn_model(config.gnn)

    # 创建环境 / Create environment
    env = create_rtl_environment(rtl_dataset, config.rl, gnn_model)

    # 创建PPO代理 / Create PPO agent
    agent = create_ppo_agent(env, config.rl)

    return {
        "environment": env,
        "agent": agent,
        "gnn_model": gnn_model,
        "config": config
    }


# 主要的便捷函数 / Main convenience functions
def quick_start(rtl_files, num_episodes=1000, config_path=None):
    """快速开始RTL优化训练 / Quick start RTL optimization training

    Args:
        rtl_files: RTL文件列表 / List of RTL files
        num_episodes: 训练回合数 / Number of training episodes
        config_path: 配置文件路径 / Configuration file path

    Returns:
        训练历史 / Training history
    """
    if not _COMPONENTS_AVAILABLE:
        raise RuntimeError("系统组件不可用 / System components not available")

    # 创建优化流水线 / Create optimization pipeline
    pipeline = create_optimization_pipeline(rtl_files, config_path)

    # 开始训练 / Start training
    training_history = pipeline["agent"].train(num_episodes)

    return {
        "training_history": training_history,
        "final_stats": pipeline["agent"].get_training_statistics(),
        "environment_stats": pipeline["environment"].get_environment_statistics()
    }


if __name__ == "__main__":
    # 打印系统信息 / Print system information
    import json
    system_info = get_system_info()
    print("RTL优化系统信息 / RTL Optimization System Information:")
    print(json.dumps(system_info, indent=2, ensure_ascii=False))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTL优化系统配置文件
Configuration file for RTL optimization system

该文件包含所有系统配置参数，包括模型超参数、训练参数、文件路径等
This file contains all system configuration parameters including model hyperparameters,
training parameters, file paths, etc.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GraphConfig:
    """图表示相关配置 / Graph representation configuration"""

    # 节点特征维度 / Node feature dimensions
    node_type_dim: int = 4          # 节点类型维度 (gate/reg/port/wire)
    gate_type_embedding_dim: int = 8 # 门类型嵌入维度
    numerical_feature_dim: int = 8   # 数值特征维度
    timing_feature_dim: int = 4      # 时序特征维度
    total_node_feature_dim: int = 24 # 总节点特征维度

    # 边类型 / Edge types
    edge_types: List[str] = None

    def __post_init__(self):
        if self.edge_types is None:
            self.edge_types = ["connection", "timing", "fanout"]


@dataclass
class GNNConfig:
    """GNN模型配置 / GNN model configuration"""

    # 模型架构 / Model architecture
    hidden_dim: int = 128            # 隐藏层维度
    num_conv_layers: int = 2         # 图卷积层数
    dropout_rate: float = 0.1        # Dropout率
    activation: str = "relu"         # 激活函数

    # 输出头 / Output heads
    ppa_output_dim: int = 3          # PPA预测输出维度 (delay/area/power)
    value_output_dim: int = 1        # 值函数输出维度


@dataclass
class RLConfig:
    """强化学习配置 / Reinforcement learning configuration"""

    # 环境参数 / Environment parameters
    max_steps_per_episode: int = 20  # 每轮最大步数
    state_dim: int = 132             # 状态维度 (128 graph + 3 PPA + 1 step)
    action_dim: int = 12             # 动作维度 (11 ABC commands + 1 no-op)

    # PPO超参数 / PPO hyperparameters
    learning_rate_policy: float = 3e-4    # 策略网络学习率
    learning_rate_value: float = 1e-3     # 值函数学习率
    clip_ratio: float = 0.2               # PPO截断比例
    entropy_coef: float = 0.01             # 熵系数
    value_loss_coef: float = 0.5           # 值函数损失系数
    max_grad_norm: float = 0.5             # 梯度截断

    # 训练参数 / Training parameters
    batch_size: int = 64                   # 批量大小
    buffer_size: int = 2048                # 经验缓冲区大小
    ppo_epochs: int = 10                   # PPO更新轮数
    num_envs: int = 4                      # 并行环境数


@dataclass
class ABCConfig:
    """ABC工具配置 / ABC tool configuration"""

    # ABC命令映射 / ABC command mapping
    abc_commands: Dict[str, str] = None
    combo_sequences: Dict[str, List[str]] = None

    # 工具路径 / Tool paths
    abc_binary_path: str = "abc"           # ABC可执行文件路径
    temp_dir: str = "/tmp/rtl_opt"         # 临时文件目录

    # 超时设置 / Timeout settings
    optimization_timeout: int = 30         # 优化超时时间(秒)
    equivalence_timeout: int = 60          # 等效性检查超时时间(秒)

    def __post_init__(self):
        if self.abc_commands is None:
            self.abc_commands = {
                "rewrite": "rewrite -l",       # AIG重写，保持逻辑等效
                "refactor": "refactor -l",     # 重构优化
                "balance": "balance -l",       # 平衡AIG深度
                "resub": "resub -l",           # 替换优化
                "compress2": "compress2",      # 综合压缩优化
                "choice": "choice",            # 选择计算优化
                "fraig": "fraig",              # 功能性归约
                "dch": "dch",                  # 深度选择计算
                "if": "if -K 6",               # FPGA技术映射
                "mfs": "mfs",                  # 最大扇入简化
                "lutpack": "lutpack"           # LUT打包优化
            }

        if self.combo_sequences is None:
            self.combo_sequences = {
                "light_opt": ["rewrite", "refactor"],
                "heavy_opt": ["rewrite", "refactor", "balance", "resub"],
                "fpga_opt": ["rewrite", "balance", "if", "lutpack"],
                "area_opt": ["compress2", "choice", "mfs"]
            }


@dataclass
class EvaluationConfig:
    """评估系统配置 / Evaluation system configuration"""

    # 混合评估策略 / Hybrid evaluation strategy
    full_eval_interval: int = 10                    # 完整评估间隔
    prediction_confidence_threshold: float = 0.8    # 预测置信度阈值

    # PPA权重 / PPA weights
    timing_weight: float = 0.5                      # 时序权重
    area_weight: float = 0.3                        # 面积权重
    power_weight: float = 0.2                       # 功耗权重

    # 奖励函数参数 / Reward function parameters
    abc_improvement_weight: float = 0.1             # ABC改善权重
    equivalence_bonus: float = 0.05                 # 等效性奖励
    failure_penalty: float = -0.2                   # 失败惩罚
    no_op_penalty: float = -0.05                    # 无操作惩罚


@dataclass
class DataConfig:
    """数据配置 / Data configuration"""

    # 数据路径 / Data paths
    rtl_rewriter_bench_path: str = "./short_benchmark"  # RTL-Rewriter基准路径
    circuitnet_path: str = "./circuitnet_data"          # CircuitNet数据路径
    cache_dir: str = "./cache"                           # 缓存目录

    # 数据分割 / Data split
    train_ratio: float = 0.7                            # 训练集比例
    val_ratio: float = 0.15                             # 验证集比例
    test_ratio: float = 0.15                            # 测试集比例

    # 预处理 / Preprocessing
    max_nodes: int = 10000                               # 最大节点数
    min_nodes: int = 10                                  # 最小节点数


@dataclass
class ExperimentConfig:
    """实验配置 / Experiment configuration"""

    # 实验参数 / Experiment parameters
    num_training_episodes: int = 10000           # 训练轮数
    eval_frequency: int = 100                    # 评估频率
    save_frequency: int = 500                    # 保存频率

    # 日志和保存 / Logging and saving
    log_dir: str = "./logs"                      # 日志目录
    checkpoint_dir: str = "./checkpoints"        # 检查点目录
    result_dir: str = "./results"                # 结果目录

    # 基线方法 / Baseline methods
    baseline_methods: List[str] = None

    def __post_init__(self):
        if self.baseline_methods is None:
            self.baseline_methods = [
                "yosys_default",
                "abc_default",
                "random_search",
                "ironman_baseline"
            ]


class Config:
    """主配置类 / Main configuration class"""

    def __init__(self):
        # 初始化各模块配置 / Initialize module configurations
        self.graph = GraphConfig()
        self.gnn = GNNConfig()
        self.rl = RLConfig()
        self.abc = ABCConfig()
        self.evaluation = EvaluationConfig()
        self.data = DataConfig()
        self.experiment = ExperimentConfig()

        # 创建必要目录 / Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """创建必要的目录 / Create necessary directories"""
        directories = [
            self.abc.temp_dir,
            self.data.cache_dir,
            self.experiment.log_dir,
            self.experiment.checkpoint_dir,
            self.experiment.result_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def save(self, filepath: str):
        """保存配置到文件 / Save configuration to file"""
        import json
        import dataclasses

        config_dict = {}
        for key, value in self.__dict__.items():
            if dataclasses.is_dataclass(value):
                config_dict[key] = dataclasses.asdict(value)
            else:
                config_dict[key] = value

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str):
        """从文件加载配置 / Load configuration from file"""
        import json

        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


# 默认配置实例 / Default configuration instance
default_config = Config()


def get_config(config_path: Optional[str] = None) -> Config:
    """获取配置实例 / Get configuration instance

    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
                    Configuration file path, use default if None

    Returns:
        Config: 配置实例 / Configuration instance
    """
    if config_path and os.path.exists(config_path):
        return Config.load(config_path)
    else:
        return default_config


if __name__ == "__main__":
    # 测试配置系统 / Test configuration system
    config = get_config()
    print("RTL优化系统配置已加载 / RTL optimization system configuration loaded")
    print(f"节点特征维度 / Node feature dimension: {config.graph.total_node_feature_dim}")
    print(f"GNN隐藏层维度 / GNN hidden dimension: {config.gnn.hidden_dim}")
    print(f"RL状态维度 / RL state dimension: {config.rl.state_dim}")
    print(f"ABC命令数量 / Number of ABC commands: {len(config.abc.abc_commands)}")
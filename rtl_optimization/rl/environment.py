#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTL优化强化学习环境
RTL optimization reinforcement learning environment

该模块实现专门针对RTL优化的强化学习环境，定义状态空间、动作空间、
奖励函数等核心组件，支持与ABC工具的集成。
This module implements reinforcement learning environment specifically for RTL optimization,
defining state space, action space, reward function and other core components,
supporting integration with ABC tool.
"""

import os
import random
import tempfile
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import numpy as np

import torch

try:
    import gym
    from gym import spaces
except ImportError:
    # 提供简化的gym接口 / Provide simplified gym interface
    class spaces:
        class Discrete:
            def __init__(self, n): self.n = n
        class Box:
            def __init__(self, low, high, shape, dtype):
                self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    class gym:
        class Env:
            def __init__(self): pass
            def reset(self): raise NotImplementedError
            def step(self, action): raise NotImplementedError

from ..utils.config import RLConfig, EvaluationConfig
from ..tools.abc_interface import ABCInterface
from ..tools.evaluator import HybridEvaluationSystem
from ..graph.rtl_graph import RTLNetlistGraph
from ..graph.gnn_model import RTLOptimizationGNN


class RTLOptimizationEnvironment(gym.Env):
    """RTL优化强化学习环境 / RTL optimization reinforcement learning environment

    该环境将RTL优化过程建模为马尔可夫决策过程，智能体通过选择ABC优化命令
    来改善电路的PPA指标。
    This environment models RTL optimization process as Markov Decision Process,
    where agent improves circuit PPA metrics by selecting ABC optimization commands.
    """

    def __init__(self, rtl_dataset: List[str],
                 config: Optional[RLConfig] = None,
                 eval_config: Optional[EvaluationConfig] = None,
                 gnn_model: Optional[RTLOptimizationGNN] = None):
        """初始化RL环境 / Initialize RL environment

        Args:
            rtl_dataset: RTL设计文件列表 / List of RTL design files
            config: RL配置 / RL configuration
            eval_config: 评估配置 / Evaluation configuration
            gnn_model: 预训练的GNN模型 / Pretrained GNN model
        """
        super(RTLOptimizationEnvironment, self).__init__()

        self.config = config or RLConfig()
        self.eval_config = eval_config or EvaluationConfig()
        self.logger = logging.getLogger(__name__)

        # 数据集 / Dataset
        self.rtl_dataset = rtl_dataset
        self.current_file_idx = 0

        # 核心组件 / Core components
        self.abc_tool = ABCInterface()
        self.graph_builder = RTLNetlistGraph()
        self.evaluator = HybridEvaluationSystem(eval_config, gnn_model)
        self.gnn_model = gnn_model

        # 动作空间：基于ABC的优化命令 / Action space: ABC-based optimization commands
        self.actions = {
            0: "rewrite",      # AIG重写，保持逻辑等效
            1: "refactor",     # 重构优化
            2: "balance",      # 平衡AIG深度
            3: "resub",        # 替换优化
            4: "compress2",    # 综合压缩优化
            5: "choice",       # 选择计算优化
            6: "fraig",        # 功能性归约
            7: "dch",          # 深度选择计算
            8: "if",           # FPGA技术映射
            9: "mfs",          # 最大扇入简化
            10: "lutpack",     # LUT打包优化
            11: "no_operation" # 跳过当前步骤
        }

        self.action_space = spaces.Discrete(len(self.actions))

        # 状态空间：图嵌入 + PPA当前值 + 步数信息
        # State space: graph embedding + current PPA + step information
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.state_dim,),
            dtype=np.float32
        )

        # 环境状态 / Environment state
        self.current_file = None
        self.current_graph = None
        self.current_ppa = None
        self.baseline_ppa = None
        self.step_count = 0
        self.episode_history = []
        self.optimization_sequence = []

        # 奖励函数权重 / Reward function weights
        self.reward_weights = {
            "timing": self.eval_config.timing_weight,
            "area": self.eval_config.area_weight,
            "power": self.eval_config.power_weight,
            "abc_improvement": self.eval_config.abc_improvement_weight,
            "equivalence_bonus": self.eval_config.equivalence_bonus,
            "failure_penalty": self.eval_config.failure_penalty,
            "no_op_penalty": self.eval_config.no_op_penalty
        }

        # 统计信息 / Statistics
        self.episode_stats = {
            "episodes": 0,
            "total_steps": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "average_reward": 0.0,
            "best_improvement": 0.0
        }

        self.logger.info(f"RTL优化RL环境初始化完成 / RTL optimization RL environment initialized: "
                        f"数据集大小={len(rtl_dataset)}, 动作空间={len(self.actions)}")

    def reset(self, file_idx: Optional[int] = None) -> np.ndarray:
        """重置环境到初始状态 / Reset environment to initial state

        Args:
            file_idx: 指定文件索引，如果为None则随机选择 / Specific file index, random if None

        Returns:
            np.ndarray: 初始观测状态 / Initial observation state
        """
        try:
            # 选择RTL设计文件 / Select RTL design file
            if file_idx is not None:
                self.current_file_idx = file_idx % len(self.rtl_dataset)
            else:
                self.current_file_idx = random.randint(0, len(self.rtl_dataset) - 1)

            self.current_file = self.rtl_dataset[self.current_file_idx]

            if not os.path.exists(self.current_file):
                raise FileNotFoundError(f"RTL文件不存在 / RTL file not found: {self.current_file}")

            # 构建初始图表示 / Build initial graph representation
            self.current_graph = self.graph_builder.build_from_file(self.current_file)

            # 获取基线PPA指标 / Get baseline PPA metrics
            self.baseline_ppa = self.evaluator.full_evaluate(self.current_file)
            self.current_ppa = self.baseline_ppa.copy()

            # 重置状态 / Reset state
            self.step_count = 0
            self.episode_history = []
            self.optimization_sequence = []

            # 更新统计信息 / Update statistics
            self.episode_stats["episodes"] += 1

            observation = self._get_observation()

            self.logger.debug(f"环境重置完成 / Environment reset completed: "
                            f"文件={os.path.basename(self.current_file)}, "
                            f"基线PPA={self.baseline_ppa}")

            return observation

        except Exception as e:
            self.logger.error(f"环境重置失败 / Environment reset failed: {e}")
            # 使用默认状态 / Use default state
            return np.zeros(self.config.state_dim, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作并返回结果 / Execute action and return result

        Args:
            action: 动作索引 / Action index

        Returns:
            Tuple[observation, reward, done, info]: 标准RL返回格式
        """
        self.step_count += 1
        self.episode_stats["total_steps"] += 1

        # 解析动作 / Parse action
        action_name = self.actions[action]

        info = {
            "action": action_name,
            "step": self.step_count,
            "file": os.path.basename(self.current_file) if self.current_file else "unknown"
        }

        try:
            # 执行优化动作 / Execute optimization action
            if action_name != "no_operation":
                optimization_result = self._apply_optimization_action(action_name)

                if optimization_result["success"]:
                    # 优化成功 / Optimization successful
                    new_ppa = optimization_result["new_ppa"]
                    abc_stats = optimization_result.get("abc_stats", {})

                    # 计算奖励 / Calculate reward
                    reward = self._calculate_reward(
                        self.current_ppa, new_ppa, action_name, abc_stats
                    )

                    # 更新状态 / Update state
                    self.current_ppa = new_ppa
                    self.current_graph = optimization_result.get("new_graph", self.current_graph)

                    self.episode_stats["successful_optimizations"] += 1

                else:
                    # 优化失败 / Optimization failed
                    reward = self.reward_weights["failure_penalty"]
                    self.episode_stats["failed_optimizations"] += 1

                info.update(optimization_result)

            else:
                # 跳过操作 / Skip operation
                reward = self.reward_weights["no_op_penalty"]
                info["skipped"] = True

            # 记录历史 / Record history
            step_info = {
                "action": action_name,
                "reward": reward,
                "ppa": self.current_ppa.copy(),
                "step": self.step_count
            }
            self.episode_history.append(step_info)
            self.optimization_sequence.append(action_name)

            # 判断回合结束 / Determine episode termination
            done = self._is_episode_done()

            # 获取新观测 / Get new observation
            observation = self._get_observation()

            # 更新统计信息 / Update statistics
            self._update_episode_stats(reward, done)

            info["episode_done"] = done
            info["total_reward"] = sum([h["reward"] for h in self.episode_history])

            return observation, reward, done, info

        except Exception as e:
            self.logger.error(f"步骤执行失败 / Step execution failed: {e}")

            # 返回错误状态 / Return error state
            reward = self.reward_weights["failure_penalty"]
            done = True
            observation = self._get_observation()

            info["error"] = str(e)

            return observation, reward, done, info

    def _apply_optimization_action(self, action_name: str) -> Dict[str, Any]:
        """应用优化动作 / Apply optimization action"""
        try:
            # 应用ABC优化 / Apply ABC optimization
            abc_result = self.abc_tool.apply_optimization(
                self.current_file, action_name, "auto"
            )

            if not abc_result["success"]:
                return {
                    "success": False,
                    "error": abc_result.get("error", "Unknown ABC error"),
                    "action": action_name
                }

            # 构建新的图表示 / Build new graph representation
            optimized_file = abc_result["optimized_file"]
            new_graph = self.graph_builder.build_from_file(optimized_file)

            # 评估新的PPA / Evaluate new PPA
            new_ppa = self.evaluator.evaluate(optimized_file, self.step_count)

            return {
                "success": True,
                "new_ppa": new_ppa,
                "new_graph": new_graph,
                "abc_stats": abc_result.get("statistics", {}),
                "optimized_file": optimized_file,
                "is_equivalent": abc_result.get("is_equivalent", True),
                "action": action_name
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": action_name
            }

    def _calculate_reward(self, old_ppa: Dict[str, Any], new_ppa: Dict[str, Any],
                         action_name: str, abc_stats: Dict[str, Any]) -> float:
        """计算奖励函数 / Calculate reward function"""

        # 1. PPA改善奖励（主要奖励）/ PPA improvement reward (main reward)
        ppa_reward = 0.0

        for metric, weight in [("delay", "timing"), ("area", "area"), ("power", "power")]:
            if metric in old_ppa and metric in new_ppa:
                old_val = old_ppa[metric]
                new_val = new_ppa[metric]

                if old_val > 0:
                    improvement = (old_val - new_val) / old_val
                    ppa_reward += self.reward_weights[weight] * improvement

        # 2. ABC级别的改善奖励（中间反馈）/ ABC level improvement reward (intermediate feedback)
        abc_reward = 0.0
        if abc_stats and "improvement" in abc_stats:
            improvement = abc_stats["improvement"]
            nodes_reduction = improvement.get("nodes_reduction", 0.0)
            depth_reduction = improvement.get("depth_reduction", 0.0)

            abc_reward = self.reward_weights["abc_improvement"] * (
                nodes_reduction + depth_reduction
            )

        # 3. 动作特异性奖励 / Action-specific reward
        action_reward = self._get_action_specific_reward(action_name, abc_stats)

        # 4. 逻辑等效性奖励 / Logic equivalence reward
        equivalence_reward = self.reward_weights["equivalence_bonus"]

        # 5. 约束违反惩罚 / Constraint violation penalty
        constraint_penalty = 0.0
        if new_ppa.get("timing_violation", False):
            constraint_penalty = -1.0

        # 总奖励 / Total reward
        total_reward = (ppa_reward + abc_reward + action_reward +
                       equivalence_reward + constraint_penalty)

        return total_reward

    def _get_action_specific_reward(self, action_name: str, abc_stats: Dict[str, Any]) -> float:
        """获取动作特异性奖励 / Get action-specific reward"""
        # 不同ABC动作的基础奖励 / Base rewards for different ABC actions
        action_rewards = {
            "rewrite": 0.02,    "refactor": 0.03,   "balance": 0.02,
            "resub": 0.03,      "compress2": 0.04,  "choice": 0.03,
            "fraig": 0.03,      "dch": 0.02,        "if": 0.02,
            "mfs": 0.04,        "lutpack": 0.02
        }

        base_reward = action_rewards.get(action_name, 0.01)

        # 如果ABC统计显示显著改善，增加奖励 / Increase reward if ABC stats show significant improvement
        if abc_stats and "improvement" in abc_stats:
            improvement = abc_stats["improvement"]
            if (improvement.get("nodes_reduction", 0) > 0.1 or
                improvement.get("depth_reduction", 0) > 0.1):
                base_reward *= 1.5

        return base_reward

    def _get_observation(self) -> np.ndarray:
        """获取当前观测状态 / Get current observation state"""
        try:
            if self.current_graph is None or self.gnn_model is None:
                # 返回默认观测 / Return default observation
                return np.zeros(self.config.state_dim, dtype=np.float32)

            # 获取图嵌入 / Get graph embedding
            with torch.no_grad():
                outputs = self.gnn_model(self.current_graph)
                graph_embedding = outputs["graph_embedding"].flatten()

            # 确保图嵌入维度正确 / Ensure correct graph embedding dimension
            if graph_embedding.size(0) != 128:
                # 调整到128维 / Adjust to 128 dimensions
                if graph_embedding.size(0) > 128:
                    graph_embedding = graph_embedding[:128]
                else:
                    padding = torch.zeros(128 - graph_embedding.size(0))
                    graph_embedding = torch.cat([graph_embedding, padding])

            # PPA当前值（归一化）/ Current PPA values (normalized)
            ppa_values = np.array([
                self.current_ppa.get("delay", 0.5),
                self.current_ppa.get("area", 0.5),
                self.current_ppa.get("power", 0.5)
            ], dtype=np.float32)

            # 步数归一化 / Normalized step count
            step_info = np.array([
                self.step_count / self.config.max_steps_per_episode
            ], dtype=np.float32)

            # 组合状态 / Combine state
            observation = np.concatenate([
                graph_embedding.numpy(),
                ppa_values,
                step_info
            ])

            # 确保观测维度正确 / Ensure correct observation dimension
            if len(observation) != self.config.state_dim:
                observation = np.resize(observation, self.config.state_dim)

            return observation.astype(np.float32)

        except Exception as e:
            self.logger.warning(f"观测获取失败 / Observation retrieval failed: {e}")
            return np.zeros(self.config.state_dim, dtype=np.float32)

    def _is_episode_done(self) -> bool:
        """判断回合是否结束 / Determine if episode is done"""
        # 达到最大步数 / Reached maximum steps
        if self.step_count >= self.config.max_steps_per_episode:
            return True

        # 早期终止条件 / Early termination conditions
        if len(self.episode_history) >= 3:
            # 连续几步都没有改善 / No improvement for consecutive steps
            recent_rewards = [h["reward"] for h in self.episode_history[-3:]]
            if all(r <= 0 for r in recent_rewards):
                return True

        # 达到显著改善 / Reached significant improvement
        if self.baseline_ppa and self.current_ppa:
            total_improvement = 0.0
            for metric, weight in [("delay", "timing"), ("area", "area"), ("power", "power")]:
                if metric in self.baseline_ppa and metric in self.current_ppa:
                    old_val = self.baseline_ppa[metric]
                    new_val = self.current_ppa[metric]
                    if old_val > 0:
                        improvement = (old_val - new_val) / old_val
                        total_improvement += self.reward_weights[weight.replace("timing", "timing")] * improvement

            # 如果总改善超过阈值，提前结束 / Early termination if total improvement exceeds threshold
            if total_improvement > 0.5:
                return True

        return False

    def _update_episode_stats(self, reward: float, done: bool) -> None:
        """更新回合统计信息 / Update episode statistics"""
        if done:
            # 计算总奖励 / Calculate total reward
            total_reward = sum([h["reward"] for h in self.episode_history])

            # 更新平均奖励 / Update average reward
            episodes = self.episode_stats["episodes"]
            old_avg = self.episode_stats["average_reward"]
            self.episode_stats["average_reward"] = (
                (old_avg * (episodes - 1) + total_reward) / episodes
            )

            # 计算总改善 / Calculate total improvement
            if self.baseline_ppa and self.current_ppa:
                total_improvement = 0.0
                for metric in ["delay", "area", "power"]:
                    if metric in self.baseline_ppa and metric in self.current_ppa:
                        old_val = self.baseline_ppa[metric]
                        new_val = self.current_ppa[metric]
                        if old_val > 0:
                            improvement = (old_val - new_val) / old_val
                            total_improvement += improvement

                if total_improvement > self.episode_stats["best_improvement"]:
                    self.episode_stats["best_improvement"] = total_improvement

    def get_episode_summary(self) -> Dict[str, Any]:
        """获取回合总结 / Get episode summary"""
        if not self.episode_history:
            return {"error": "No episode data available"}

        total_reward = sum([h["reward"] for h in self.episode_history])

        # PPA改善计算 / PPA improvement calculation
        ppa_improvement = {}
        if self.baseline_ppa and self.current_ppa:
            for metric in ["delay", "area", "power"]:
                if metric in self.baseline_ppa and metric in self.current_ppa:
                    old_val = self.baseline_ppa[metric]
                    new_val = self.current_ppa[metric]
                    if old_val > 0:
                        improvement = (old_val - new_val) / old_val
                        ppa_improvement[metric] = improvement

        return {
            "file": os.path.basename(self.current_file) if self.current_file else "unknown",
            "steps": self.step_count,
            "total_reward": total_reward,
            "average_reward": total_reward / len(self.episode_history),
            "optimization_sequence": self.optimization_sequence,
            "ppa_improvement": ppa_improvement,
            "baseline_ppa": self.baseline_ppa,
            "final_ppa": self.current_ppa,
            "successful_opts": sum(1 for h in self.episode_history if h["reward"] > 0),
            "failed_opts": sum(1 for h in self.episode_history if h["reward"] < 0)
        }

    def get_environment_statistics(self) -> Dict[str, Any]:
        """获取环境统计信息 / Get environment statistics"""
        stats = self.episode_stats.copy()
        stats.update(self.evaluator.get_evaluation_statistics())

        if stats["total_steps"] > 0:
            stats["success_rate"] = stats["successful_optimizations"] / stats["total_steps"]

        return stats

    def render(self, mode: str = "human") -> Optional[str]:
        """渲染环境状态 / Render environment state"""
        if mode == "human":
            print(f"\n=== RTL优化环境状态 / RTL Optimization Environment State ===")
            print(f"当前文件 / Current file: {os.path.basename(self.current_file) if self.current_file else 'None'}")
            print(f"步数 / Step: {self.step_count}/{self.config.max_steps_per_episode}")

            if self.current_ppa:
                print(f"当前PPA / Current PPA: delay={self.current_ppa.get('delay', 'N/A'):.3f}, "
                      f"area={self.current_ppa.get('area', 'N/A'):.3f}, "
                      f"power={self.current_ppa.get('power', 'N/A'):.3f}")

            if self.optimization_sequence:
                print(f"优化序列 / Optimization sequence: {' -> '.join(self.optimization_sequence[-5:])}")

            print("=" * 60)

        return None

    def close(self) -> None:
        """关闭环境 / Close environment"""
        self.evaluator.cleanup()
        self.abc_tool.cleanup()


# 工具函数 / Utility functions

def create_rtl_environment(rtl_dataset: List[str],
                          config: Optional[RLConfig] = None,
                          gnn_model: Optional[RTLOptimizationGNN] = None) -> RTLOptimizationEnvironment:
    """创建RTL优化环境 / Create RTL optimization environment"""
    return RTLOptimizationEnvironment(rtl_dataset, config, gnn_model=gnn_model)


# 测试代码 / Test code
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # 创建测试数据集 / Create test dataset
    test_dataset = []  # 这里需要实际的RTL文件路径 / Need actual RTL file paths here

    if test_dataset:
        # 测试环境 / Test environment
        env = create_rtl_environment(test_dataset)

        print("RTL优化RL环境测试 / RTL optimization RL environment test:")
        print(f"动作空间 / Action space: {env.action_space}")
        print(f"状态空间 / State space: {env.observation_space}")

        # 简单的随机策略测试 / Simple random policy test
        obs = env.reset()
        done = False
        step = 0

        while not done and step < 5:
            action = env.action_space.sample() if hasattr(env.action_space, 'sample') else 0
            obs, reward, done, info = env.step(action)
            print(f"步骤 {step}: 动作={action}, 奖励={reward:.3f}, 完成={done}")
            step += 1

        summary = env.get_episode_summary()
        print(f"\n回合总结 / Episode summary: {summary}")

        env.close()
    else:
        print("需要提供RTL数据集路径进行测试 / Need to provide RTL dataset paths for testing")

    print("RL环境测试完成 / RL environment test completed")
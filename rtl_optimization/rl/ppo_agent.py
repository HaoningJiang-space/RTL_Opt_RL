#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO算法实现 - 专门针对RTL优化
PPO algorithm implementation - specialized for RTL optimization

该模块实现专门针对RTL优化任务的PPO算法，包括策略网络、值函数网络、
经验缓冲区和训练逻辑。
This module implements PPO algorithm specifically for RTL optimization tasks,
including policy network, value function network, experience buffer and training logic.
"""

import os
# import random  # 暂时未使用
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from ..utils.config import RLConfig
from ..rl.environment import RTLOptimizationEnvironment


class PolicyNetwork(nn.Module):
    """策略网络 / Policy network

    将状态映射到动作概率分布的神经网络。
    Neural network that maps states to action probability distributions.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """初始化策略网络 / Initialize policy network

        Args:
            state_dim: 状态维度 / State dimension
            action_dim: 动作维度 / Action dimension
            hidden_dim: 隐藏层维度 / Hidden layer dimension
        """
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # 权重初始化 / Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重 / Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播 / Forward propagation

        Args:
            state: 状态张量 / State tensor

        Returns:
            torch.Tensor: 动作logits
        """
        return self.network(state)

    def get_action_distribution(self, state: torch.Tensor) -> Categorical:
        """获取动作分布 / Get action distribution

        Args:
            state: 状态张量 / State tensor

        Returns:
            Categorical: 动作分布
        """
        logits = self.forward(state)
        return Categorical(logits=logits)

    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作和对数概率 / Get action and log probability

        Args:
            state: 状态张量 / State tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 动作和对数概率
        """
        dist = self.get_action_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class ValueNetwork(nn.Module):
    """值函数网络 / Value function network

    估计给定状态的价值函数。
    Estimates value function for given states.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        """初始化值函数网络 / Initialize value function network

        Args:
            state_dim: 状态维度 / State dimension
            hidden_dim: 隐藏层维度 / Hidden layer dimension
        """
        super(ValueNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 权重初始化 / Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重 / Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播 / Forward propagation

        Args:
            state: 状态张量 / State tensor

        Returns:
            torch.Tensor: 状态价值
        """
        return self.network(state)


class ExperienceBuffer:
    """经验缓冲区 / Experience buffer

    存储和管理PPO训练所需的经验数据。
    Stores and manages experience data required for PPO training.
    """

    def __init__(self, buffer_size: int):
        """初始化经验缓冲区 / Initialize experience buffer

        Args:
            buffer_size: 缓冲区大小 / Buffer size
        """
        self.buffer_size = buffer_size
        self.clear()

    def clear(self):
        """清空缓冲区 / Clear buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.next_states = []

    def add(self, state: np.ndarray, action: int, reward: float,
            log_prob: float, value: float, done: bool, next_state: np.ndarray):
        """添加经验 / Add experience

        Args:
            state: 状态 / State
            action: 动作 / Action
            reward: 奖励 / Reward
            log_prob: 对数概率 / Log probability
            value: 状态价值 / State value
            done: 是否结束 / Whether done
            next_state: 下一状态 / Next state
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.next_states.append(next_state)

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """获取批量数据 / Get batch data

        Returns:
            Dict[str, torch.Tensor]: 批量经验数据
        """
        # 计算优势和回报 / Calculate advantages and returns
        advantages, returns = self._compute_advantages_and_returns()

        return {
            "states": torch.FloatTensor(np.array(self.states)),
            "actions": torch.LongTensor(self.actions),
            "old_log_probs": torch.FloatTensor(self.log_probs),
            "advantages": torch.FloatTensor(advantages),
            "returns": torch.FloatTensor(returns),
            "values": torch.FloatTensor(self.values)
        }

    def _compute_advantages_and_returns(self, gamma: float = 0.99,
                                       lambda_gae: float = 0.95) -> Tuple[List[float], List[float]]:
        """计算优势和回报 / Compute advantages and returns

        使用GAE (Generalized Advantage Estimation) 计算优势函数。
        Uses GAE (Generalized Advantage Estimation) to compute advantage function.

        Args:
            gamma: 折扣因子 / Discount factor
            lambda_gae: GAE参数 / GAE parameter

        Returns:
            Tuple[List[float], List[float]]: 优势和回报
        """
        advantages = []
        returns = []

        # 计算TD误差 / Calculate TD errors
        td_errors = []
        for i in range(len(self.rewards)):
            if i == len(self.rewards) - 1 or self.dones[i]:
                next_value = 0.0
            else:
                next_value = self.values[i + 1]

            td_error = self.rewards[i] + gamma * next_value - self.values[i]
            td_errors.append(td_error)

        # 计算GAE优势 / Calculate GAE advantages
        gae = 0.0
        for i in reversed(range(len(td_errors))):
            if self.dones[i]:
                gae = 0.0

            gae = td_errors[i] + gamma * lambda_gae * gae
            advantages.insert(0, gae)

        # 计算回报 / Calculate returns
        for i in range(len(advantages)):
            returns.append(advantages[i] + self.values[i])

        return advantages, returns

    def size(self) -> int:
        """获取缓冲区大小 / Get buffer size"""
        return len(self.states)

    def is_full(self) -> bool:
        """检查缓冲区是否已满 / Check if buffer is full"""
        return len(self.states) >= self.buffer_size


class RTLOptimizationPPO:
    """RTL优化专用PPO算法 / PPO algorithm specialized for RTL optimization

    实现PPO算法的完整训练流程，专门针对RTL优化任务进行优化。
    Implements complete PPO training pipeline, specifically optimized for RTL optimization tasks.
    """

    def __init__(self, env: RTLOptimizationEnvironment, config: Optional[RLConfig] = None):
        """初始化PPO算法 / Initialize PPO algorithm

        Args:
            env: RTL优化环境 / RTL optimization environment
            config: RL配置 / RL configuration
        """
        self.env = env
        self.config = config or RLConfig()
        self.logger = logging.getLogger(__name__)

        # 设备选择 / Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建网络 / Create networks
        self.policy_net = PolicyNetwork(
            self.config.state_dim,
            self.config.action_dim,
            hidden_dim=256
        ).to(self.device)

        self.value_net = ValueNetwork(
            self.config.state_dim,
            hidden_dim=256
        ).to(self.device)

        # 优化器 / Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate_policy
        )

        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=self.config.learning_rate_value
        )

        # 经验缓冲区 / Experience buffer
        self.buffer = ExperienceBuffer(self.config.buffer_size)

        # 训练统计 / Training statistics
        self.training_stats = {
            "episodes": 0,
            "total_steps": 0,
            "policy_losses": deque(maxlen=100),
            "value_losses": deque(maxlen=100),
            "entropies": deque(maxlen=100),
            "episode_rewards": deque(maxlen=100),
            "episode_lengths": deque(maxlen=100),
            "best_reward": float('-inf'),
            "convergence_threshold": 1e-4,
            "no_improvement_count": 0
        }

        self.logger.info(f"RTL优化PPO算法初始化完成 / RTL optimization PPO algorithm initialized: "
                        f"设备={self.device}, 状态维度={self.config.state_dim}, "
                        f"动作维度={self.config.action_dim}")

    def select_action(self, state: np.ndarray, training: bool = True) -> Dict[str, Any]:
        """选择动作 / Select action

        Args:
            state: 当前状态 / Current state
            training: 是否处于训练模式 / Whether in training mode

        Returns:
            Dict[str, Any]: 包含动作、对数概率、价值等信息的字典
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 获取动作分布 / Get action distribution
            action_dist = self.policy_net.get_action_distribution(state_tensor)

            if training:
                # 训练时采样 / Sample during training
                action = action_dist.sample()
            else:
                # 评估时选择最可能的动作 / Choose most likely action during evaluation
                action = torch.argmax(action_dist.probs)

            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()

            # 获取状态价值 / Get state value
            value = self.value_net(state_tensor)

        return {
            "action": action.item(),
            "log_prob": log_prob.item(),
            "value": value.item(),
            "entropy": entropy.item()
        }

    def update_policy(self) -> Dict[str, float]:
        """更新策略 / Update policy

        Returns:
            Dict[str, float]: 训练损失统计
        """
        if not self.buffer.is_full():
            return {"warning": "Buffer not full, skipping update"}

        # 获取批量数据 / Get batch data
        batch = self.buffer.get_batch()
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)

        # 标准化优势 / Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        # PPO多轮更新 / PPO multiple epochs update
        for epoch in range(self.config.ppo_epochs):
            # 创建数据加载器 / Create data loader
            dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            num_batches = 0

            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in dataloader:
                # 策略网络更新 / Policy network update
                policy_loss, entropy = self._update_policy_step(
                    batch_states, batch_actions, batch_old_log_probs, batch_advantages
                )

                # 值函数网络更新 / Value network update
                value_loss = self._update_value_step(batch_states, batch_returns)

                epoch_policy_loss += policy_loss
                epoch_value_loss += value_loss
                epoch_entropy += entropy
                num_batches += 1

            total_policy_loss += epoch_policy_loss / num_batches
            total_value_loss += epoch_value_loss / num_batches
            total_entropy += epoch_entropy / num_batches

        # 平均损失 / Average losses
        avg_policy_loss = total_policy_loss / self.config.ppo_epochs
        avg_value_loss = total_value_loss / self.config.ppo_epochs
        avg_entropy = total_entropy / self.config.ppo_epochs

        # 更新统计信息 / Update statistics
        self.training_stats["policy_losses"].append(avg_policy_loss)
        self.training_stats["value_losses"].append(avg_value_loss)
        self.training_stats["entropies"].append(avg_entropy)

        # 清空缓冲区 / Clear buffer
        self.buffer.clear()

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy
        }

    def _update_policy_step(self, states: torch.Tensor, actions: torch.Tensor,
                           old_log_probs: torch.Tensor, advantages: torch.Tensor) -> Tuple[float, float]:
        """策略网络单步更新 / Single step policy network update"""

        # 计算新的动作概率 / Calculate new action probabilities
        action_dist = self.policy_net.get_action_distribution(states)
        new_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()

        # 计算概率比率 / Calculate probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # PPO截断目标 / PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 熵正则化 / Entropy regularization
        total_loss = policy_loss - self.config.entropy_coef * entropy

        # 梯度更新 / Gradient update
        self.policy_optimizer.zero_grad()
        total_loss.backward()

        # 梯度截断 / Gradient clipping
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)

        self.policy_optimizer.step()

        return policy_loss.item(), entropy.item()

    def _update_value_step(self, states: torch.Tensor, returns: torch.Tensor) -> float:
        """值函数网络单步更新 / Single step value network update"""

        # 计算价值损失 / Calculate value loss
        values = self.value_net(states).squeeze()
        value_loss = F.mse_loss(values, returns)

        # 梯度更新 / Gradient update
        self.value_optimizer.zero_grad()
        value_loss.backward()

        # 梯度截断 / Gradient clipping
        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)

        self.value_optimizer.step()

        return value_loss.item()

    def train_episode(self) -> Dict[str, Any]:
        """训练一个回合 / Train one episode

        Returns:
            Dict[str, Any]: 回合训练结果
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # 选择动作 / Select action
            action_info = self.select_action(state, training=True)
            action = action_info["action"]

            # 执行动作 / Execute action
            next_state, reward, done, info = self.env.step(action)

            # 存储经验 / Store experience
            self.buffer.add(
                state, action, reward,
                action_info["log_prob"], action_info["value"],
                done, next_state
            )

            state = next_state
            episode_reward += reward
            episode_length += 1
            self.training_stats["total_steps"] += 1

            # 如果缓冲区满了，进行策略更新 / If buffer is full, perform policy update
            if self.buffer.is_full():
                update_info = self.update_policy()
                self.logger.debug(f"策略更新 / Policy update: {update_info}")

        # 更新回合统计 / Update episode statistics
        self.training_stats["episodes"] += 1
        self.training_stats["episode_rewards"].append(episode_reward)
        self.training_stats["episode_lengths"].append(episode_length)

        # 检查是否达到最佳性能 / Check if best performance is reached
        if episode_reward > self.training_stats["best_reward"]:
            self.training_stats["best_reward"] = episode_reward
            self.training_stats["no_improvement_count"] = 0
        else:
            self.training_stats["no_improvement_count"] += 1

        return {
            "episode": self.training_stats["episodes"],
            "reward": episode_reward,
            "length": episode_length,
            "best_reward": self.training_stats["best_reward"],
            "total_steps": self.training_stats["total_steps"]
        }

    def train(self, num_episodes: int, save_frequency: int = 100,
              checkpoint_dir: str = "./checkpoints") -> List[Dict[str, Any]]:
        """训练PPO算法 / Train PPO algorithm

        Args:
            num_episodes: 训练回合数 / Number of training episodes
            save_frequency: 保存频率 / Save frequency
            checkpoint_dir: 检查点目录 / Checkpoint directory

        Returns:
            List[Dict[str, Any]]: 训练历史记录
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        training_history = []

        self.logger.info(f"开始PPO训练 / Starting PPO training: {num_episodes} episodes")

        for episode in range(num_episodes):
            # 训练一个回合 / Train one episode
            episode_result = self.train_episode()
            training_history.append(episode_result)

            # 定期日志 / Periodic logging
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean([r["reward"] for r in training_history[-10:]])
                self.logger.info(f"回合 {episode + 1}/{num_episodes}: "
                               f"平均奖励={avg_reward:.3f}, "
                               f"最佳奖励={self.training_stats['best_reward']:.3f}")

            # 定期保存 / Periodic saving
            if (episode + 1) % save_frequency == 0:
                self.save_checkpoint(os.path.join(checkpoint_dir, f"ppo_checkpoint_{episode + 1}.pth"))

            # 早期停止检查 / Early stopping check
            if self._should_early_stop():
                self.logger.info(f"早期停止训练 / Early stopping at episode {episode + 1}")
                break

        self.logger.info("PPO训练完成 / PPO training completed")
        return training_history

    def _should_early_stop(self) -> bool:
        """检查是否应该早期停止 / Check if should early stop"""
        # 如果连续多个回合没有改善，停止训练 / Stop if no improvement for consecutive episodes
        if self.training_stats["no_improvement_count"] > 50:
            return True

        # 如果策略损失收敛，停止训练 / Stop if policy loss converges
        if len(self.training_stats["policy_losses"]) > 20:
            recent_losses = list(self.training_stats["policy_losses"])[-20:]
            if max(recent_losses) - min(recent_losses) < self.training_stats["convergence_threshold"]:
                return True

        return False

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """评估策略性能 / Evaluate policy performance

        Args:
            num_episodes: 评估回合数 / Number of evaluation episodes

        Returns:
            Dict[str, Any]: 评估结果
        """
        self.logger.info(f"开始策略评估 / Starting policy evaluation: {num_episodes} episodes")

        eval_rewards = []
        eval_lengths = []
        eval_summaries = []

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                # 选择动作（评估模式）/ Select action (evaluation mode)
                action_info = self.select_action(state, training=False)
                action = action_info["action"]

                # 执行动作 / Execute action
                state, reward, done, _ = self.env.step(action)

                episode_reward += reward
                episode_length += 1

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)

            # 获取回合总结 / Get episode summary
            summary = self.env.get_episode_summary()
            eval_summaries.append(summary)

        # 计算评估统计 / Calculate evaluation statistics
        eval_stats = {
            "num_episodes": num_episodes,
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "min_reward": np.min(eval_rewards),
            "max_reward": np.max(eval_rewards),
            "mean_length": np.mean(eval_lengths),
            "std_length": np.std(eval_lengths),
            "episode_summaries": eval_summaries
        }

        self.logger.info(f"策略评估完成 / Policy evaluation completed: "
                        f"平均奖励={eval_stats['mean_reward']:.3f} ± {eval_stats['std_reward']:.3f}")

        return eval_stats

    def save_checkpoint(self, filepath: str) -> None:
        """保存检查点 / Save checkpoint"""
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "training_stats": self.training_stats,
            "config": self.config
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"检查点已保存 / Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """加载检查点 / Load checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        self.training_stats = checkpoint["training_stats"]

        self.logger.info(f"检查点已加载 / Checkpoint loaded: {filepath}")

    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息 / Get training statistics"""
        stats = self.training_stats.copy()

        # 转换deque为list / Convert deque to list
        for key in ["policy_losses", "value_losses", "entropies", "episode_rewards", "episode_lengths"]:
            stats[key] = list(stats[key])

        # 计算额外统计 / Calculate additional statistics
        if stats["episode_rewards"]:
            stats["recent_avg_reward"] = np.mean(stats["episode_rewards"][-10:])

        return stats


# 工具函数 / Utility functions

def create_ppo_agent(env: RTLOptimizationEnvironment,
                    config: Optional[RLConfig] = None) -> RTLOptimizationPPO:
    """创建PPO代理 / Create PPO agent"""
    return RTLOptimizationPPO(env, config)


# 测试代码 / Test code
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # 这里需要实际的环境来测试 / Need actual environment for testing
    print("PPO算法模块测试 / PPO algorithm module test")

    # 测试网络创建 / Test network creation
    policy_net = PolicyNetwork(132, 12)
    value_net = ValueNetwork(132)

    print(f"策略网络参数数量 / Policy network parameters: {sum(p.numel() for p in policy_net.parameters())}")
    print(f"值函数网络参数数量 / Value network parameters: {sum(p.numel() for p in value_net.parameters())}")

    # 测试前向传播 / Test forward propagation
    test_state = torch.randn(1, 132)
    policy_output = policy_net(test_state)
    value_output = value_net(test_state)

    print(f"策略网络输出形状 / Policy network output shape: {policy_output.shape}")
    print(f"值函数网络输出形状 / Value network output shape: {value_output.shape}")

    print("PPO算法模块测试完成 / PPO algorithm module test completed")
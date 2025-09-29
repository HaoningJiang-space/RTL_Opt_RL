"""
RTLMultiAgentTrainer - 基于VeRL的RTL多智能体训练器
集成ReMA框架，支持RTL优化的多智能体强化学习训练
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

# ReMA/VeRL imports - 需要根据实际框架调整
try:
    from verl.trainer.ppo_trainer import PPOTrainer
    from verl.trainer.config import TrainerConfig
    from verl.single_controller.ray import RayWorkerGroup
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    logging.warning("VeRL框架不可用，将使用简化训练逻辑")

from .multi_agent_environment import MultiAgentRTLEnvironment, RewardCalculator
from ..agents import MetaOptimizerAgent, CodeRewriterAgent
from ..utils import VerilogVerifier, DataProcessor, OptimizationAnalyzer


class RTLMultiAgentTrainer:
    """基于VeRL的RTL多智能体训练器"""

    def __init__(
        self,
        config: Dict[str, Any],
        agents: Dict[str, Any],
        optimization_data: List[Dict[str, Any]],
        verifier: VerilogVerifier
    ):
        self.config = config
        self.agents = agents
        self.optimization_data = optimization_data
        self.verifier = verifier

        # 设置日志
        self.logger = logging.getLogger(self.__class__.__name__)

        # 初始化组件
        self.environment = MultiAgentRTLEnvironment(
            agents=agents,
            optimization_data=optimization_data,
            verifier=verifier,
            max_steps=config.get("max_steps_per_episode", 20)
        )

        self.analyzer = OptimizationAnalyzer()

        # 训练状态
        self.current_epoch = 0
        self.episode_histories = []
        self.best_performance = 0.0

        # 初始化VeRL训练器（如果可用）
        if VERL_AVAILABLE:
            self._setup_verl_trainer()
        else:
            self._setup_simple_trainer()

        # 实验追踪
        if config.get("use_wandb", False):
            self._setup_wandb()

    def _setup_verl_trainer(self):
        """设置VeRL训练器"""
        try:
            # 构建VeRL配置
            verl_config = self._build_verl_config()

            # 初始化PPO训练器
            self.ppo_trainer = PPOTrainer(verl_config)

            # 设置多智能体参数
            self._setup_multi_agent_parameters()

            self.logger.info("VeRL训练器初始化成功")

        except Exception as e:
            self.logger.error(f"VeRL训练器初始化失败: {e}")
            self._setup_simple_trainer()

    def _setup_simple_trainer(self):
        """设置简化训练器"""
        self.ppo_trainer = None

        # 直接设置优化器
        self.optimizers = {}
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'parameters'):
                self.optimizers[agent_name] = torch.optim.AdamW(
                    agent.parameters(),
                    lr=self.config.get("learning_rate", 1e-4),
                    weight_decay=self.config.get("weight_decay", 1e-5)
                )

        self.logger.info("简化训练器初始化完成")

    def _build_verl_config(self) -> Dict[str, Any]:
        """构建VeRL配置"""
        return {
            "trainer": {
                "project_name": self.config.get("project_name", "rtl_multi_agent"),
                "experiment_name": self.config.get("experiment_name", f"rtl_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "total_epochs": self.config.get("num_epochs", 10),
                "save_freq": self.config.get("save_freq", 5),
                "test_freq": self.config.get("test_freq", 2)
            },
            "data": {
                "train_batch_size": self.config.get("batch_size", 8),
                "max_prompt_length": self.config.get("max_prompt_length", 4096),
                "max_response_length": self.config.get("max_response_length", 2048)
            },
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": self.config.get("ppo_mini_batch_size", 4),
                    "clip_mode": "turn",  # ReMA特性：turn-level clipping
                    "agg_mode": "trajectory"  # ReMA特性：trajectory aggregation
                },
                "rollout": {
                    "max_num_turns": self.config.get("max_turns", 20),
                    "n": self.config.get("rollout_n", 4),
                    "stop_when_truncated": True
                }
            },
            "algorithm": {
                "adv_estimator": "grpo"  # ReMA推荐的优势估计器
            }
        }

    def _setup_multi_agent_parameters(self):
        """设置多智能体参数"""

        # 为每个智能体设置独立的优化器
        self.agent_optimizers = {}
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'parameters'):
                self.agent_optimizers[agent_name] = torch.optim.AdamW(
                    agent.parameters(),
                    lr=self.config.get("learning_rates", {}).get(agent_name, 1e-4)
                )

        # 设置奖励函数
        self.reward_functions = {
            "meta_optimizer": self._meta_planning_reward,
            "code_rewriter": self._code_quality_reward
        }

    def _setup_wandb(self):
        """设置W&B实验追踪"""
        try:
            wandb.init(
                project=self.config.get("wandb_project", "rtl-multi-agent"),
                name=self.config.get("experiment_name"),
                config=self.config
            )
            self.logger.info("W&B初始化成功")
        except Exception as e:
            self.logger.warning(f"W&B初始化失败: {e}")

    def train(self) -> Dict[str, Any]:
        """主训练循环"""

        self.logger.info("开始RTL多智能体训练")

        training_results = {
            "training_history": [],
            "best_performance": 0.0,
            "final_analysis": {}
        }

        for epoch in range(self.config.get("num_epochs", 10)):
            self.current_epoch = epoch

            # 训练一个epoch
            epoch_results = self.train_epoch()
            training_results["training_history"].append(epoch_results)

            # 更新最佳性能
            current_performance = epoch_results.get("average_success_rate", 0.0)
            if current_performance > training_results["best_performance"]:
                training_results["best_performance"] = current_performance
                self._save_best_model()

            # 记录和可视化
            self._log_epoch_results(epoch, epoch_results)

            # 定期评估和保存
            if (epoch + 1) % self.config.get("eval_freq", 2) == 0:
                eval_results = self.evaluate()
                self._log_evaluation_results(epoch, eval_results)

            if (epoch + 1) % self.config.get("save_freq", 5) == 0:
                self._save_checkpoint(epoch)

        # 最终分析
        training_results["final_analysis"] = self.analyzer.analyze_episode_results(
            self.episode_histories
        )

        self.logger.info("训练完成")
        return training_results

    def train_epoch(self) -> Dict[str, Any]:
        """训练一个epoch"""

        epoch_results = {
            "epoch": self.current_epoch,
            "episode_results": [],
            "average_reward": 0.0,
            "average_success_rate": 0.0,
            "agent_losses": {}
        }

        episodes_per_epoch = self.config.get("episodes_per_epoch", 20)

        for episode_idx in range(episodes_per_epoch):
            # 运行一个episode
            episode_result = self.run_episode()
            epoch_results["episode_results"].append(episode_result)

            # 如果使用VeRL，执行PPO更新
            if self.ppo_trainer:
                self._update_with_verl(episode_result)
            else:
                # 使用简化的策略梯度更新
                agent_losses = self._update_agents_simple(episode_result)
                epoch_results["agent_losses"] = agent_losses

        # 计算epoch统计
        epoch_results["average_reward"] = np.mean([
            ep.get("total_reward", 0.0) for ep in epoch_results["episode_results"]
        ])

        epoch_results["average_success_rate"] = np.mean([
            1.0 if ep.get("success", False) else 0.0 for ep in epoch_results["episode_results"]
        ])

        return epoch_results

    def run_episode(self) -> Dict[str, Any]:
        """运行一个训练episode"""

        # 重置环境
        observations = self.environment.reset()
        done = False
        episode_history = []
        total_reward = 0.0

        while not done:
            # 生成智能体动作
            actions = self._generate_agent_actions(observations)

            # 执行环境步骤
            next_observations, rewards, done, info = self.environment.step(actions)

            # 记录步骤
            step_record = {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "next_observations": next_observations,
                "info": info
            }
            episode_history.append(step_record)

            total_reward += sum(rewards.values())
            observations = next_observations

        # 获取episode总结
        episode_summary = self.environment.get_episode_summary()
        episode_result = {
            **episode_summary,
            "episode_history": episode_history,
            "total_reward": total_reward
        }

        # 添加到历史记录
        self.episode_histories.append(episode_result)

        return episode_result

    def _generate_agent_actions(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """生成智能体动作"""
        actions = {}

        # MetaOptimizer生成动作
        if "meta_optimizer" in self.agents:
            meta_state = {
                "rtl_code": observations["meta_optimizer"]["rtl_code"],
                "goal": self._decode_goal(observations["meta_optimizer"]["optimization_goal"]),
                "step": observations["meta_optimizer"]["current_step"]
            }
            meta_action = self.agents["meta_optimizer"].generate_action(meta_state)
            actions["meta_optimizer"] = meta_action

        # CodeRewriter生成动作
        if "code_rewriter" in self.agents:
            rewriter_state = {
                "rtl_code": observations["code_rewriter"]["rtl_code"],
                "meta_instructions": json.loads(observations["code_rewriter"]["meta_instructions"])
                    if observations["code_rewriter"]["meta_instructions"] else {},
                "step": observations["code_rewriter"]["current_step"]
            }
            rewriter_action = self.agents["code_rewriter"].generate_action(rewriter_state)
            actions["code_rewriter"] = rewriter_action

        return actions

    def _update_with_verl(self, episode_result: Dict[str, Any]):
        """使用VeRL框架更新智能体"""
        try:
            # 构建VeRL格式的经验数据
            experience_data = self._convert_to_verl_format(episode_result)

            # 执行PPO更新
            self.ppo_trainer.train_step(experience_data)

        except Exception as e:
            self.logger.error(f"VeRL更新失败: {e}")

    def _update_agents_simple(self, episode_result: Dict[str, Any]) -> Dict[str, float]:
        """简化的智能体更新"""
        agent_losses = {}

        try:
            episode_history = episode_result.get("episode_history", [])

            for agent_name, optimizer in self.optimizers.items():
                if agent_name in self.agents:
                    # 计算策略梯度损失
                    loss = self._calculate_policy_loss(agent_name, episode_history)

                    # 反向传播和优化
                    optimizer.zero_grad()
                    if loss is not None:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.agents[agent_name].parameters(),
                            max_norm=1.0
                        )
                        optimizer.step()
                        agent_losses[agent_name] = loss.item()
                    else:
                        agent_losses[agent_name] = 0.0

        except Exception as e:
            self.logger.error(f"智能体更新失败: {e}")

        return agent_losses

    def _calculate_policy_loss(self, agent_name: str, episode_history: List[Dict]) -> Optional[torch.Tensor]:
        """计算策略损失（简化版REINFORCE）"""

        if not episode_history:
            return None

        try:
            returns = []
            rewards = []

            # 收集该智能体的奖励
            for step in episode_history:
                agent_reward = step.get("rewards", {}).get(agent_name, 0.0)
                rewards.append(agent_reward)

            # 计算discounted returns
            gamma = self.config.get("discount_factor", 0.95)
            discounted_return = 0.0

            for reward in reversed(rewards):
                discounted_return = reward + gamma * discounted_return
                returns.insert(0, discounted_return)

            if not returns:
                return None

            # 标准化returns
            returns = torch.tensor(returns, dtype=torch.float32)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # 简化的策略损失（这里需要根据实际的智能体架构调整）
            policy_loss = -returns.mean()  # 负号因为要最大化奖励

            return policy_loss

        except Exception as e:
            self.logger.error(f"计算{agent_name}策略损失失败: {e}")
            return None

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """评估当前模型性能"""

        self.logger.info(f"开始评估，运行{num_episodes}个episodes")

        eval_results = {
            "num_episodes": num_episodes,
            "episode_results": [],
            "average_reward": 0.0,
            "success_rate": 0.0,
            "average_steps": 0.0
        }

        # 设置为评估模式
        for agent in self.agents.values():
            if hasattr(agent, 'eval'):
                agent.eval()

        try:
            for _ in range(num_episodes):
                episode_result = self.run_episode()
                eval_results["episode_results"].append(episode_result)

            # 计算评估统计
            rewards = [ep.get("total_reward", 0.0) for ep in eval_results["episode_results"]]
            successes = [ep.get("success", False) for ep in eval_results["episode_results"]]
            steps = [ep.get("total_steps", 0) for ep in eval_results["episode_results"]]

            eval_results["average_reward"] = np.mean(rewards)
            eval_results["success_rate"] = np.mean([1.0 if s else 0.0 for s in successes])
            eval_results["average_steps"] = np.mean(steps)

        finally:
            # 恢复训练模式
            for agent in self.agents.values():
                if hasattr(agent, 'train'):
                    agent.train()

        return eval_results

    def _decode_goal(self, goal_code: int) -> str:
        """解码优化目标"""
        goal_mapping = {0: "timing", 1: "area", 2: "power", 3: "balanced"}
        return goal_mapping.get(goal_code, "balanced")

    def _convert_to_verl_format(self, episode_result: Dict[str, Any]) -> Dict[str, Any]:
        """转换为VeRL格式的经验数据"""
        # 这里需要根据VeRL的具体API进行实现
        return {
            "episode_data": episode_result,
            "formatted_for_verl": True
        }

    def _meta_planning_reward(self, result: Dict[str, Any]) -> float:
        """MetaOptimizer奖励函数"""
        return result.get("rewards", {}).get("meta_optimizer", 0.0)

    def _code_quality_reward(self, result: Dict[str, Any]) -> float:
        """CodeRewriter奖励函数"""
        return result.get("rewards", {}).get("code_rewriter", 0.0)

    def _log_epoch_results(self, epoch: int, results: Dict[str, Any]):
        """记录epoch结果"""

        log_data = {
            "epoch": epoch,
            "average_reward": results["average_reward"],
            "success_rate": results["average_success_rate"],
            "num_episodes": len(results["episode_results"])
        }

        # 记录智能体损失
        if results.get("agent_losses"):
            for agent_name, loss in results["agent_losses"].items():
                log_data[f"{agent_name}_loss"] = loss

        # W&B记录
        if wandb.run:
            wandb.log(log_data)

        # 控制台记录
        self.logger.info(
            f"Epoch {epoch}: Reward={results['average_reward']:.3f}, "
            f"Success Rate={results['average_success_rate']:.2%}"
        )

    def _log_evaluation_results(self, epoch: int, eval_results: Dict[str, Any]):
        """记录评估结果"""

        log_data = {
            "eval_epoch": epoch,
            "eval_average_reward": eval_results["average_reward"],
            "eval_success_rate": eval_results["success_rate"],
            "eval_average_steps": eval_results["average_steps"]
        }

        if wandb.run:
            wandb.log(log_data)

        self.logger.info(
            f"Evaluation at epoch {epoch}: "
            f"Reward={eval_results['average_reward']:.3f}, "
            f"Success Rate={eval_results['success_rate']:.2%}"
        )

    def _save_best_model(self):
        """保存最佳模型"""
        save_path = Path(self.config.get("save_dir", "./models")) / "best_model"
        save_path.mkdir(parents=True, exist_ok=True)

        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'save_checkpoint'):
                agent.save_checkpoint(str(save_path / f"{agent_name}_best.pth"))

        self.logger.info(f"最佳模型已保存到: {save_path}")

    def _save_checkpoint(self, epoch: int):
        """保存检查点"""
        save_path = Path(self.config.get("save_dir", "./models")) / f"checkpoint_epoch_{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存智能体
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'save_checkpoint'):
                agent.save_checkpoint(str(save_path / f"{agent_name}.pth"))

        # 保存训练状态
        training_state = {
            "epoch": epoch,
            "config": self.config,
            "episode_histories": self.episode_histories[-100:],  # 只保存最近100个episodes
            "best_performance": self.best_performance
        }

        with open(save_path / "training_state.json", 'w', encoding='utf-8') as f:
            json.dump(training_state, f, indent=2, ensure_ascii=False)

        self.logger.info(f"检查点已保存到: {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint_path = Path(checkpoint_path)

        try:
            # 加载智能体
            for agent_name, agent in self.agents.items():
                agent_path = checkpoint_path / f"{agent_name}.pth"
                if agent_path.exists() and hasattr(agent, 'load_checkpoint'):
                    self.agents[agent_name] = agent.load_checkpoint(str(agent_path))

            # 加载训练状态
            state_path = checkpoint_path / "training_state.json"
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    training_state = json.load(f)

                self.current_epoch = training_state.get("epoch", 0)
                self.best_performance = training_state.get("best_performance", 0.0)
                # 可选择是否恢复episode历史

            self.logger.info(f"检查点加载成功: {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")

    def generate_final_report(self) -> str:
        """生成最终训练报告"""
        return self.analyzer.generate_performance_report(
            self.episode_histories,
            self.config.get("save_dir", "./models") + "/training_report.md"
        )
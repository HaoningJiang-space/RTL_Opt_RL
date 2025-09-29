"""
MultiAgentRTLEnvironment - 多智能体RTL优化环境
为ReMA框架提供RTL优化的多智能体强化学习环境
"""

import gym
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import random
import json
import logging
from pathlib import Path

from ..agents import MetaOptimizerAgent, CodeRewriterAgent
from ..utils import VerilogVerifier


class MultiAgentRTLEnvironment(gym.Env):
    """多智能体RTL优化环境"""

    def __init__(
        self,
        agents: Dict[str, Any],
        optimization_data: List[Dict[str, Any]],
        verifier: VerilogVerifier,
        max_steps: int = 20,
        success_threshold: float = 0.8
    ):
        super().__init__()

        self.agents = agents
        self.optimization_data = optimization_data
        self.verifier = verifier
        self.max_steps = max_steps
        self.success_threshold = success_threshold

        # 环境状态
        self.current_episode_data = None
        self.current_step = 0
        self.episode_history = []

        # 设置日志
        self.logger = logging.getLogger(self.__class__.__name__)

        # 定义观测和动作空间
        self.observation_spaces = self._define_observation_spaces()
        self.action_spaces = self._define_action_spaces()

        # 奖励计算器
        self.reward_calculator = RewardCalculator(verifier)

    def _define_observation_spaces(self) -> Dict[str, gym.Space]:
        """定义各智能体的观测空间"""
        return {
            "meta_optimizer": gym.spaces.Dict({
                "rtl_code": gym.spaces.Text(max_length=10000),  # RTL代码文本
                "optimization_goal": gym.spaces.Discrete(4),    # 优化目标：timing, area, power, mixed
                "current_step": gym.spaces.Discrete(self.max_steps),
                "ppa_metrics": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),  # [delay, area, power]
                "complexity_score": gym.spaces.Box(
                    low=0.0, high=100.0, shape=(1,), dtype=np.float32
                )
            }),

            "code_rewriter": gym.spaces.Dict({
                "rtl_code": gym.spaces.Text(max_length=10000),
                "meta_instructions": gym.spaces.Text(max_length=5000),
                "current_step": gym.spaces.Discrete(self.max_steps),
                "constraints": gym.spaces.Text(max_length=2000),
                "previous_results": gym.spaces.Text(max_length=3000)
            })
        }

    def _define_action_spaces(self) -> Dict[str, gym.Space]:
        """定义各智能体的动作空间"""
        return {
            "meta_optimizer": gym.spaces.Dict({
                "strategy_type": gym.spaces.Discrete(5),  # 策略类型
                "priority_areas": gym.spaces.MultiBinary(10),  # 优先区域
                "optimization_sequence": gym.spaces.Text(max_length=3000)
            }),

            "code_rewriter": gym.spaces.Dict({
                "optimized_code": gym.spaces.Text(max_length=15000),
                "applied_techniques": gym.spaces.MultiBinary(8),  # 应用的技术
                "confidence": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            })
        }

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """重置环境"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 随机选择优化案例
        self.current_episode_data = random.choice(self.optimization_data)
        self.current_step = 0
        self.episode_history = []

        # 为各智能体生成初始观测
        observations = self._generate_observations()

        self.logger.info(f"环境重置，选择案例: {self.current_episode_data.get('case_id', 'unknown')}")

        return observations

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """执行一步多智能体交互"""
        self.current_step += 1

        # 执行智能体动作序列
        step_results = self._execute_agent_actions(actions)

        # 计算奖励
        rewards = self.reward_calculator.calculate_step_rewards(
            self.current_episode_data,
            step_results,
            self.current_step
        )

        # 检查终止条件
        done, termination_info = self._check_termination(step_results, rewards)

        # 生成新观测
        next_observations = self._generate_observations(step_results)

        # 记录步骤历史
        self.episode_history.append({
            "step": self.current_step,
            "actions": actions,
            "results": step_results,
            "rewards": rewards,
            "observations": next_observations
        })

        # 构建信息字典
        info = {
            "step_results": step_results,
            "termination_info": termination_info,
            "episode_progress": self.current_step / self.max_steps
        }

        return next_observations, rewards, done, info

    def _execute_agent_actions(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """按序执行智能体动作"""
        results = {}

        # 1. 先执行MetaOptimizer的规划动作
        if "meta_optimizer" in actions:
            meta_result = self._execute_meta_action(actions["meta_optimizer"])
            results["meta_optimizer"] = meta_result

        # 2. 基于Meta的输出执行CodeRewriter动作
        if "code_rewriter" in actions:
            # 将Meta的指令传递给CodeRewriter
            meta_instructions = results.get("meta_optimizer", {}).get("instructions", {})
            rewrite_result = self._execute_rewrite_action(
                actions["code_rewriter"],
                meta_instructions
            )
            results["code_rewriter"] = rewrite_result

            # 3. 验证重写结果
            if rewrite_result.get("optimized_code"):
                verification_result = self.verifier.comprehensive_verify(
                    rewrite_result["optimized_code"],
                    self.current_episode_data["original_code"],
                    self._extract_module_name(self.current_episode_data["original_code"])
                )
                results["verification"] = verification_result

        return results

    def _execute_meta_action(self, meta_action: Dict[str, Any]) -> Dict[str, Any]:
        """执行MetaOptimizer动作"""
        try:
            # 构建状态
            state = {
                "rtl_code": self.current_episode_data["original_code"],
                "goal": self.current_episode_data.get("optimization_goal", "balanced"),
                "step": self.current_step
            }

            # 让MetaOptimizer智能体生成规划
            meta_agent = self.agents.get("meta_optimizer")
            if meta_agent:
                planning_result = meta_agent.generate_action(state)
                return planning_result
            else:
                return {"error": "MetaOptimizer智能体未找到"}

        except Exception as e:
            self.logger.error(f"执行Meta动作失败: {e}")
            return {"error": str(e)}

    def _execute_rewrite_action(
        self,
        rewrite_action: Dict[str, Any],
        meta_instructions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行CodeRewriter动作"""
        try:
            # 构建状态
            state = {
                "rtl_code": self.current_episode_data["original_code"],
                "meta_instructions": meta_instructions,
                "step": self.current_step
            }

            # 让CodeRewriter智能体生成优化代码
            rewriter_agent = self.agents.get("code_rewriter")
            if rewriter_agent:
                rewrite_result = rewriter_agent.generate_action(state)
                return rewrite_result
            else:
                return {"error": "CodeRewriter智能体未找到"}

        except Exception as e:
            self.logger.error(f"执行Rewrite动作失败: {e}")
            return {"error": str(e)}

    def _generate_observations(self, step_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成智能体观测"""
        observations = {}

        # MetaOptimizer观测
        observations["meta_optimizer"] = {
            "rtl_code": self.current_episode_data["original_code"][:8000],  # 截断长代码
            "optimization_goal": self._encode_goal(
                self.current_episode_data.get("optimization_goal", "balanced")
            ),
            "current_step": self.current_step,
            "ppa_metrics": self._extract_ppa_metrics(),
            "complexity_score": self._calculate_complexity_score()
        }

        # CodeRewriter观测
        meta_instructions = ""
        if step_results and "meta_optimizer" in step_results:
            meta_instructions = json.dumps(
                step_results["meta_optimizer"].get("instructions", {}),
                ensure_ascii=False
            )[:3000]

        observations["code_rewriter"] = {
            "rtl_code": self.current_episode_data["original_code"][:8000],
            "meta_instructions": meta_instructions,
            "current_step": self.current_step,
            "constraints": json.dumps(
                self.current_episode_data.get("constraints", {}),
                ensure_ascii=False
            )[:1500],
            "previous_results": self._get_previous_results()[:2000]
        }

        return observations

    def _check_termination(
        self,
        step_results: Dict[str, Any],
        rewards: Dict[str, float]
    ) -> Tuple[bool, Dict[str, Any]]:
        """检查终止条件"""

        termination_info = {
            "reason": None,
            "success": False,
            "final_scores": rewards
        }

        # 1. 达到最大步数
        if self.current_step >= self.max_steps:
            termination_info["reason"] = "max_steps_reached"
            termination_info["success"] = any(
                score >= self.success_threshold for score in rewards.values()
            )
            return True, termination_info

        # 2. 验证失败（语法或综合错误）
        verification = step_results.get("verification", {})
        if verification and not verification.get("detailed_scores", {}).get("syntax", True):
            termination_info["reason"] = "syntax_error"
            termination_info["success"] = False
            return True, termination_info

        # 3. 达到成功阈值
        verification_reward = verification.get("verification_reward", 0.0)
        if verification_reward >= self.success_threshold:
            termination_info["reason"] = "success_achieved"
            termination_info["success"] = True
            return True, termination_info

        # 4. 连续多步无改善
        if self._check_no_improvement():
            termination_info["reason"] = "no_improvement"
            termination_info["success"] = False
            return True, termination_info

        return False, termination_info

    def _encode_goal(self, goal: str) -> int:
        """将优化目标编码为数字"""
        goal_mapping = {
            "timing": 0,
            "area": 1,
            "power": 2,
            "mixed": 3,
            "balanced": 3
        }
        return goal_mapping.get(goal.lower(), 3)

    def _extract_ppa_metrics(self) -> np.ndarray:
        """提取PPA指标"""
        ppa_data = self.current_episode_data.get("ppa_improvement", {})
        return np.array([
            ppa_data.get("delay", 0.0),
            ppa_data.get("area", 0.0),
            ppa_data.get("power", 0.0)
        ], dtype=np.float32)

    def _calculate_complexity_score(self) -> np.ndarray:
        """计算复杂度分数"""
        code = self.current_episode_data["original_code"]

        # 简化的复杂度计算
        lines = len(code.split('\n'))
        always_blocks = code.count('always')
        assign_statements = code.count('assign')

        score = lines * 0.1 + always_blocks * 2.0 + assign_statements * 0.5
        return np.array([min(100.0, score)], dtype=np.float32)

    def _get_previous_results(self) -> str:
        """获取之前步骤的结果摘要"""
        if not self.episode_history:
            return "无历史记录"

        recent_history = self.episode_history[-3:]  # 最近3步
        summary = []

        for hist in recent_history:
            step_summary = {
                "step": hist["step"],
                "rewards": hist["rewards"],
                "success": hist["results"].get("verification", {}).get("detailed_scores", {})
            }
            summary.append(step_summary)

        return json.dumps(summary, ensure_ascii=False)

    def _check_no_improvement(self, patience: int = 3) -> bool:
        """检查是否连续多步无改善"""
        if len(self.episode_history) < patience:
            return False

        recent_rewards = [
            hist["rewards"].get("total", 0.0)
            for hist in self.episode_history[-patience:]
        ]

        # 如果最近几步的奖励都很低，认为无改善
        return all(reward < 0.3 for reward in recent_rewards)

    def _extract_module_name(self, verilog_code: str) -> Optional[str]:
        """从Verilog代码中提取模块名"""
        import re
        match = re.search(r'module\s+(\w+)', verilog_code)
        return match.group(1) if match else None

    def get_episode_summary(self) -> Dict[str, Any]:
        """获取episode总结"""
        if not self.episode_history:
            return {"error": "没有历史记录"}

        total_reward = sum(
            sum(hist["rewards"].values()) for hist in self.episode_history
        )

        final_verification = None
        if self.episode_history:
            final_verification = self.episode_history[-1]["results"].get("verification")

        return {
            "total_steps": len(self.episode_history),
            "total_reward": total_reward,
            "average_reward": total_reward / len(self.episode_history) if self.episode_history else 0,
            "final_verification": final_verification,
            "case_id": self.current_episode_data.get("case_id", "unknown"),
            "success": final_verification.get("verification_reward", 0.0) >= self.success_threshold if final_verification else False
        }

    def render(self, mode: str = "human") -> Optional[str]:
        """渲染环境状态"""
        if mode == "human":
            summary = self.get_episode_summary()
            print(f"Episode Summary: {json.dumps(summary, indent=2, ensure_ascii=False)}")
            return None
        elif mode == "json":
            return json.dumps(self.get_episode_summary(), ensure_ascii=False)

    def close(self):
        """关闭环境"""
        if hasattr(self.verifier, 'cleanup'):
            self.verifier.cleanup()


class RewardCalculator:
    """奖励计算器"""

    def __init__(self, verifier: VerilogVerifier):
        self.verifier = verifier

    def calculate_step_rewards(
        self,
        episode_data: Dict[str, Any],
        step_results: Dict[str, Any],
        current_step: int
    ) -> Dict[str, float]:
        """计算步骤奖励"""

        rewards = {
            "meta_optimizer": 0.0,
            "code_rewriter": 0.0,
            "total": 0.0
        }

        # MetaOptimizer奖励：基于规划质量
        meta_result = step_results.get("meta_optimizer", {})
        if meta_result:
            meta_reward = self._calculate_meta_reward(meta_result, episode_data)
            rewards["meta_optimizer"] = meta_reward

        # CodeRewriter奖励：基于代码质量和验证结果
        rewrite_result = step_results.get("code_rewriter", {})
        verification_result = step_results.get("verification", {})
        if rewrite_result:
            rewrite_reward = self._calculate_rewrite_reward(
                rewrite_result, verification_result, episode_data
            )
            rewards["code_rewriter"] = rewrite_reward

        # 总奖励
        rewards["total"] = (rewards["meta_optimizer"] + rewards["code_rewriter"]) / 2

        # 步骤惩罚（鼓励早期成功）
        step_penalty = current_step * 0.01
        rewards["total"] = max(0.0, rewards["total"] - step_penalty)

        return rewards

    def _calculate_meta_reward(self, meta_result: Dict[str, Any], episode_data: Dict[str, Any]) -> float:
        """计算MetaOptimizer奖励"""
        reward = 0.0

        # 基础奖励：成功生成规划
        if "strategy" in meta_result and "instructions" in meta_result:
            reward += 0.3

        # 置信度奖励
        confidence = meta_result.get("confidence", 0.0)
        reward += confidence * 0.3

        # 策略合理性奖励
        strategy = meta_result.get("strategy", {})
        if strategy.get("optimization_sequence"):
            reward += 0.2

        # 目标匹配奖励
        expected_goal = episode_data.get("optimization_goal", "balanced")
        actual_strategy = strategy.get("strategy_type", "")
        if expected_goal in actual_strategy or expected_goal == "balanced":
            reward += 0.2

        return min(1.0, reward)

    def _calculate_rewrite_reward(
        self,
        rewrite_result: Dict[str, Any],
        verification_result: Dict[str, Any],
        episode_data: Dict[str, Any]
    ) -> float:
        """计算CodeRewriter奖励"""

        # 主要基于验证结果
        if verification_result:
            verification_reward = verification_result.get("verification_reward", 0.0)

            # 添加代码生成质量奖励
            confidence = rewrite_result.get("confidence", 0.0)
            safety_bonus = 0.1 if rewrite_result.get("safety_check", {}).get("passed", False) else 0.0

            total_reward = verification_reward * 0.7 + confidence * 0.2 + safety_bonus
            return min(1.0, total_reward)

        # 如果没有验证结果，基于基础指标
        if rewrite_result.get("optimized_code"):
            return rewrite_result.get("confidence", 0.0) * 0.5

        return 0.0
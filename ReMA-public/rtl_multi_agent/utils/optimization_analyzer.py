"""
OptimizationAnalyzer - 优化分析器
分析优化结果和性能指标
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import logging
from collections import defaultdict
import json


class OptimizationAnalyzer:
    """优化结果分析器"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_episode_results(self, episode_histories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析episode结果"""

        if not episode_histories:
            return {"error": "无历史数据"}

        total_episodes = len(episode_histories)
        successful_episodes = sum(1 for ep in episode_histories if ep.get("success", False))

        # 基础统计
        basic_stats = {
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "success_rate": successful_episodes / total_episodes if total_episodes > 0 else 0.0,
            "average_steps": np.mean([len(ep.get("episode_history", [])) for ep in episode_histories]),
            "average_total_reward": np.mean([ep.get("total_reward", 0.0) for ep in episode_histories])
        }

        # 奖励分析
        reward_stats = self.analyze_rewards(episode_histories)

        # 优化类型分析
        optimization_stats = self.analyze_optimization_types(episode_histories)

        # 验证结果分析
        verification_stats = self.analyze_verification_results(episode_histories)

        # 学习曲线数据
        learning_curve = self.compute_learning_curve(episode_histories)

        return {
            "basic_stats": basic_stats,
            "reward_analysis": reward_stats,
            "optimization_analysis": optimization_stats,
            "verification_analysis": verification_stats,
            "learning_curve": learning_curve
        }

    def analyze_rewards(self, episode_histories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析奖励统计"""

        meta_rewards = []
        rewrite_rewards = []
        total_rewards = []

        for episode in episode_histories:
            for step in episode.get("episode_history", []):
                rewards = step.get("rewards", {})
                meta_rewards.append(rewards.get("meta_optimizer", 0.0))
                rewrite_rewards.append(rewards.get("code_rewriter", 0.0))
                total_rewards.append(rewards.get("total", 0.0))

        return {
            "meta_optimizer": {
                "mean": np.mean(meta_rewards) if meta_rewards else 0.0,
                "std": np.std(meta_rewards) if meta_rewards else 0.0,
                "max": np.max(meta_rewards) if meta_rewards else 0.0,
                "min": np.min(meta_rewards) if meta_rewards else 0.0
            },
            "code_rewriter": {
                "mean": np.mean(rewrite_rewards) if rewrite_rewards else 0.0,
                "std": np.std(rewrite_rewards) if rewrite_rewards else 0.0,
                "max": np.max(rewrite_rewards) if rewrite_rewards else 0.0,
                "min": np.min(rewrite_rewards) if rewrite_rewards else 0.0
            },
            "total": {
                "mean": np.mean(total_rewards) if total_rewards else 0.0,
                "std": np.std(total_rewards) if total_rewards else 0.0,
                "max": np.max(total_rewards) if total_rewards else 0.0,
                "min": np.min(total_rewards) if total_rewards else 0.0
            }
        }

    def analyze_optimization_types(self, episode_histories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析优化类型统计"""

        optimization_counts = defaultdict(int)
        success_by_type = defaultdict(int)
        total_by_type = defaultdict(int)

        for episode in episode_histories:
            opt_goal = episode.get("case_data", {}).get("optimization_goal", "unknown")
            total_by_type[opt_goal] += 1

            if episode.get("success", False):
                success_by_type[opt_goal] += 1

            # 统计使用的优化策略
            for step in episode.get("episode_history", []):
                meta_result = step.get("results", {}).get("meta_optimizer", {})
                strategy = meta_result.get("strategy", {}).get("strategy_type", "unknown")
                optimization_counts[strategy] += 1

        success_rates = {
            opt_type: success_by_type[opt_type] / total_by_type[opt_type] if total_by_type[opt_type] > 0 else 0.0
            for opt_type in total_by_type.keys()
        }

        return {
            "optimization_goal_distribution": dict(total_by_type),
            "success_rates_by_goal": success_rates,
            "strategy_usage": dict(optimization_counts)
        }

    def analyze_verification_results(self, episode_histories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析验证结果统计"""

        syntax_success_count = 0
        synthesis_success_count = 0
        total_verifications = 0
        verification_rewards = []

        for episode in episode_histories:
            for step in episode.get("episode_history", []):
                verification = step.get("results", {}).get("verification", {})
                if verification:
                    total_verifications += 1
                    verification_rewards.append(verification.get("verification_reward", 0.0))

                    detailed_scores = verification.get("detailed_scores", {})
                    if detailed_scores.get("syntax", 0.0) > 0.8:
                        syntax_success_count += 1
                    if detailed_scores.get("synthesis", 0.0) > 0.8:
                        synthesis_success_count += 1

        if total_verifications > 0:
            return {
                "total_verifications": total_verifications,
                "syntax_success_rate": syntax_success_count / total_verifications,
                "synthesis_success_rate": synthesis_success_count / total_verifications,
                "average_verification_reward": np.mean(verification_rewards),
                "verification_reward_std": np.std(verification_rewards)
            }
        else:
            return {
                "total_verifications": 0,
                "syntax_success_rate": 0.0,
                "synthesis_success_rate": 0.0,
                "average_verification_reward": 0.0,
                "verification_reward_std": 0.0
            }

    def compute_learning_curve(self, episode_histories: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """计算学习曲线"""

        episode_rewards = []
        success_rates = []
        window_size = max(1, len(episode_histories) // 10)  # 10个窗口

        for i, episode in enumerate(episode_histories):
            episode_rewards.append(episode.get("total_reward", 0.0))

            # 计算滑动窗口成功率
            start_idx = max(0, i - window_size + 1)
            window_episodes = episode_histories[start_idx:i+1]
            window_success_rate = sum(1 for ep in window_episodes if ep.get("success", False)) / len(window_episodes)
            success_rates.append(window_success_rate)

        return {
            "episode_rewards": episode_rewards,
            "success_rates": success_rates,
            "episode_indices": list(range(len(episode_histories)))
        }

    def compare_optimization_strategies(
        self,
        episode_histories: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """比较不同优化策略的效果"""

        strategy_results = defaultdict(lambda: {
            "episodes": [],
            "rewards": [],
            "success_count": 0,
            "total_count": 0
        })

        for episode in episode_histories:
            # 提取主要使用的策略
            strategies_used = []
            for step in episode.get("episode_history", []):
                meta_result = step.get("results", {}).get("meta_optimizer", {})
                strategy = meta_result.get("strategy", {}).get("strategy_type")
                if strategy:
                    strategies_used.append(strategy)

            if strategies_used:
                # 使用最常出现的策略
                main_strategy = max(set(strategies_used), key=strategies_used.count)
                strategy_results[main_strategy]["episodes"].append(episode)
                strategy_results[main_strategy]["rewards"].append(episode.get("total_reward", 0.0))
                strategy_results[main_strategy]["total_count"] += 1

                if episode.get("success", False):
                    strategy_results[main_strategy]["success_count"] += 1

        # 计算每个策略的统计信息
        comparison_results = {}
        for strategy, data in strategy_results.items():
            if data["total_count"] > 0:
                comparison_results[strategy] = {
                    "total_episodes": data["total_count"],
                    "success_rate": data["success_count"] / data["total_count"],
                    "average_reward": np.mean(data["rewards"]),
                    "reward_std": np.std(data["rewards"]),
                    "max_reward": np.max(data["rewards"]),
                    "min_reward": np.min(data["rewards"])
                }

        return comparison_results

    def generate_performance_report(
        self,
        episode_histories: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """生成性能报告"""

        analysis_results = self.analyze_episode_results(episode_histories)
        strategy_comparison = self.compare_optimization_strategies(episode_histories)

        report = self._format_performance_report(analysis_results, strategy_comparison)

        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"性能报告已保存到: {output_path}")
            except Exception as e:
                self.logger.error(f"保存报告失败: {e}")

        return report

    def _format_performance_report(
        self,
        analysis_results: Dict[str, Any],
        strategy_comparison: Dict[str, Dict[str, Any]]
    ) -> str:
        """格式化性能报告"""

        basic_stats = analysis_results["basic_stats"]
        reward_analysis = analysis_results["reward_analysis"]
        verification_analysis = analysis_results["verification_analysis"]

        report = f"""# RTL多智能体优化系统性能报告

## 基础统计信息
- 总episode数: {basic_stats['total_episodes']}
- 成功episode数: {basic_stats['successful_episodes']}
- 成功率: {basic_stats['success_rate']:.2%}
- 平均步数: {basic_stats['average_steps']:.1f}
- 平均总奖励: {basic_stats['average_total_reward']:.3f}

## 奖励分析

### MetaOptimizer Agent
- 平均奖励: {reward_analysis['meta_optimizer']['mean']:.3f} ± {reward_analysis['meta_optimizer']['std']:.3f}
- 奖励范围: [{reward_analysis['meta_optimizer']['min']:.3f}, {reward_analysis['meta_optimizer']['max']:.3f}]

### CodeRewriter Agent
- 平均奖励: {reward_analysis['code_rewriter']['mean']:.3f} ± {reward_analysis['code_rewriter']['std']:.3f}
- 奖励范围: [{reward_analysis['code_rewriter']['min']:.3f}, {reward_analysis['code_rewriter']['max']:.3f}]

## 验证结果分析
- 总验证次数: {verification_analysis['total_verifications']}
- 语法成功率: {verification_analysis['syntax_success_rate']:.2%}
- 综合成功率: {verification_analysis['synthesis_success_rate']:.2%}
- 平均验证奖励: {verification_analysis['average_verification_reward']:.3f}

## 优化策略对比
"""

        for strategy, stats in strategy_comparison.items():
            report += f"""
### {strategy.upper()}
- Episode数量: {stats['total_episodes']}
- 成功率: {stats['success_rate']:.2%}
- 平均奖励: {stats['average_reward']:.3f} ± {stats['reward_std']:.3f}
- 奖励范围: [{stats['min_reward']:.3f}, {stats['max_reward']:.3f}]
"""

        report += f"""
## 生成时间
{self._get_current_time()}
"""

        return report

    def _get_current_time(self) -> str:
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
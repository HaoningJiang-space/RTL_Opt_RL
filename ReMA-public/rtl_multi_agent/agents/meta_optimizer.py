"""
MetaOptimizer Agent - 元优化智能体
负责分析RTL代码，制定高层次的优化策略和计划
"""

from typing import Dict, Any, List, Optional
import json
import re
from .base_agent import BaseAgent


class MetaOptimizerAgent(BaseAgent):
    """元优化智能体：分析全局特征，制定优化战略"""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, "meta_optimizer", **kwargs)

    def _get_agent_config(self) -> Dict[str, Any]:
        """获取元优化智能体的特定配置"""
        return {
            "optimization_strategies": [
                "timing_optimization",      # 时序优化
                "area_optimization",        # 面积优化
                "power_optimization",       # 功耗优化
                "mixed_optimization"        # 混合优化
            ],
            "analysis_focus": [
                "critical_path",           # 关键路径分析
                "resource_utilization",    # 资源利用分析
                "parallelism",             # 并行性分析
                "pipeline_opportunities"   # 流水线机会分析
            ],
            "priority_levels": ["high", "medium", "low"],
            "max_optimization_steps": 10
        }

    def generate_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成元优化动作"""
        rtl_code = state.get("rtl_code", "")
        optimization_goal = state.get("goal", "balanced")
        current_step = state.get("step", 0)

        # 分析RTL代码
        analysis_result = self.meta_analyze(rtl_code, optimization_goal)

        # 生成优化策略
        optimization_strategy = self.generate_optimization_strategy(
            analysis_result, optimization_goal, current_step
        )

        return {
            "action_type": "meta_planning",
            "analysis": analysis_result,
            "strategy": optimization_strategy,
            "instructions": self.generate_rewriter_instructions(optimization_strategy),
            "confidence": self.calculate_confidence(analysis_result)
        }

    def meta_analyze(self, rtl_code: str, goal: str = "balanced") -> Dict[str, Any]:
        """对RTL代码进行元分析"""

        # 基础结构分析
        structure_analysis = self.analyze_rtl_structure(rtl_code)

        # 使用LLM进行深度分析
        analysis_prompt = self._build_analysis_prompt(rtl_code, goal)
        llm_analysis = self.generate_text(analysis_prompt, max_new_tokens=800)

        # 解析LLM分析结果
        parsed_analysis = self._parse_llm_analysis(llm_analysis)

        return {
            "structure": structure_analysis,
            "llm_analysis": parsed_analysis,
            "optimization_opportunities": self.identify_optimization_opportunities(
                rtl_code, structure_analysis
            ),
            "bottlenecks": self.identify_bottlenecks(rtl_code, structure_analysis),
            "complexity_assessment": self.assess_complexity(structure_analysis)
        }

    def _build_analysis_prompt(self, rtl_code: str, goal: str) -> str:
        """构建分析提示"""
        prompt_template = """你是一位经验丰富的RTL优化专家。请分析以下Verilog代码并提供优化建议。

分析目标: {goal}

Verilog代码:
```verilog
{rtl_code}
```

请从以下方面进行分析：
1. **关键路径分析**: 识别可能的时序瓶颈和关键路径
2. **资源利用**: 分析逻辑资源、存储资源的使用效率
3. **并行性机会**: 识别可以并行化的操作
4. **流水线机会**: 分析是否可以插入流水线寄存器
5. **逻辑优化**: 识别可以简化或重组的逻辑
6. **存储优化**: 分析存储结构是否可以优化

请以JSON格式输出分析结果：
{{
    "critical_paths": ["路径描述"],
    "resource_bottlenecks": ["瓶颈描述"],
    "parallelism_opportunities": ["并行机会"],
    "pipeline_opportunities": ["流水线机会"],
    "logic_simplification": ["逻辑简化建议"],
    "memory_optimization": ["存储优化建议"],
    "overall_assessment": "整体评估",
    "recommended_strategy": "推荐策略"
}}"""

        return self.format_prompt(
            prompt_template,
            rtl_code=rtl_code[:2000],  # 限制代码长度
            goal=goal
        )

    def _parse_llm_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """解析LLM分析结果"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass

        # 如果JSON解析失败，使用启发式解析
        return self._heuristic_parse_analysis(analysis_text)

    def _heuristic_parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """启发式解析分析文本"""
        result = {
            "critical_paths": [],
            "resource_bottlenecks": [],
            "parallelism_opportunities": [],
            "pipeline_opportunities": [],
            "logic_simplification": [],
            "memory_optimization": [],
            "overall_assessment": "",
            "recommended_strategy": "balanced"
        }

        lines = analysis_text.split('\n')

        for line in lines:
            line = line.strip().lower()

            if any(keyword in line for keyword in ['critical', 'timing', 'delay']):
                result["critical_paths"].append(line)
            elif any(keyword in line for keyword in ['parallel', 'concurrent']):
                result["parallelism_opportunities"].append(line)
            elif any(keyword in line for keyword in ['pipeline', 'stage']):
                result["pipeline_opportunities"].append(line)
            elif any(keyword in line for keyword in ['simplify', 'reduce', 'optimize']):
                result["logic_simplification"].append(line)
            elif any(keyword in line for keyword in ['memory', 'ram', 'storage']):
                result["memory_optimization"].append(line)

        return result

    def generate_optimization_strategy(
        self,
        analysis: Dict[str, Any],
        goal: str,
        current_step: int
    ) -> Dict[str, Any]:
        """根据分析结果生成优化策略"""

        strategy = {
            "strategy_type": self._determine_strategy_type(analysis, goal),
            "priority_areas": self._prioritize_optimization_areas(analysis),
            "optimization_sequence": self._plan_optimization_sequence(analysis, goal),
            "expected_improvements": self._estimate_improvements(analysis, goal),
            "constraints": self._identify_constraints(analysis),
            "step": current_step
        }

        return strategy

    def _determine_strategy_type(self, analysis: Dict[str, Any], goal: str) -> str:
        """确定策略类型"""
        if goal == "timing":
            return "timing_optimization"
        elif goal == "area":
            return "area_optimization"
        elif goal == "power":
            return "power_optimization"
        else:
            # 基于分析结果自动选择
            complexity = analysis.get("complexity_assessment", {})
            if complexity.get("timing_critical", False):
                return "timing_optimization"
            elif complexity.get("area_constrained", False):
                return "area_optimization"
            else:
                return "mixed_optimization"

    def _prioritize_optimization_areas(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """优先级排序优化区域"""
        areas = []

        # 基于LLM分析确定优先级
        llm_analysis = analysis.get("llm_analysis", {})

        if llm_analysis.get("critical_paths"):
            areas.append({
                "area": "timing_optimization",
                "priority": "high",
                "description": "关键路径优化",
                "targets": llm_analysis["critical_paths"]
            })

        if llm_analysis.get("pipeline_opportunities"):
            areas.append({
                "area": "pipeline_insertion",
                "priority": "medium",
                "description": "流水线插入机会",
                "targets": llm_analysis["pipeline_opportunities"]
            })

        if llm_analysis.get("logic_simplification"):
            areas.append({
                "area": "logic_optimization",
                "priority": "medium",
                "description": "逻辑简化",
                "targets": llm_analysis["logic_simplification"]
            })

        # 按优先级排序
        priority_order = {"high": 0, "medium": 1, "low": 2}
        areas.sort(key=lambda x: priority_order.get(x["priority"], 2))

        return areas

    def _plan_optimization_sequence(
        self,
        analysis: Dict[str, Any],
        goal: str
    ) -> List[Dict[str, Any]]:
        """规划优化序列"""
        sequence = []
        priority_areas = self._prioritize_optimization_areas(analysis)

        step = 1
        for area in priority_areas[:self.agent_config["max_optimization_steps"]]:
            sequence.append({
                "step": step,
                "operation": area["area"],
                "target": area["targets"][:2] if area["targets"] else ["general"],
                "priority": area["priority"],
                "description": area["description"]
            })
            step += 1

        return sequence

    def _estimate_improvements(self, analysis: Dict[str, Any], goal: str) -> Dict[str, float]:
        """估算预期改善"""
        # 基于复杂度和优化机会估算
        complexity = analysis.get("complexity_assessment", {})
        opportunities = analysis.get("optimization_opportunities", [])

        base_improvement = min(0.3, len(opportunities) * 0.05)

        if goal == "timing":
            return {
                "delay": base_improvement * 1.5,
                "area": -base_improvement * 0.5,
                "power": base_improvement * 0.8
            }
        elif goal == "area":
            return {
                "delay": -base_improvement * 0.3,
                "area": base_improvement * 1.2,
                "power": base_improvement * 0.6
            }
        else:  # balanced
            return {
                "delay": base_improvement,
                "area": base_improvement * 0.8,
                "power": base_improvement * 0.9
            }

    def _identify_constraints(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """识别优化约束"""
        return {
            "preserve_functionality": True,
            "timing_constraints": analysis.get("timing_requirements", {}),
            "area_budget": analysis.get("area_constraints", {}),
            "power_budget": analysis.get("power_constraints", {}),
            "interface_compatibility": True
        }

    def generate_rewriter_instructions(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """为CodeRewriter生成具体指令"""
        return {
            "strategy_type": strategy["strategy_type"],
            "optimization_sequence": strategy["optimization_sequence"],
            "priority_areas": strategy["priority_areas"],
            "constraints": strategy["constraints"],
            "expected_improvements": strategy["expected_improvements"],
            "step_instructions": self._generate_step_instructions(strategy)
        }

    def _generate_step_instructions(self, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成步骤化指令"""
        instructions = []

        for seq_item in strategy["optimization_sequence"]:
            instruction = {
                "step": seq_item["step"],
                "operation": seq_item["operation"],
                "target": seq_item["target"],
                "priority": seq_item["priority"],
                "specific_actions": self._get_specific_actions(seq_item["operation"])
            }
            instructions.append(instruction)

        return instructions

    def _get_specific_actions(self, operation: str) -> List[str]:
        """获取特定操作的具体行动"""
        action_map = {
            "timing_optimization": [
                "识别关键路径",
                "插入流水线寄存器",
                "减少逻辑深度",
                "优化组合逻辑"
            ],
            "area_optimization": [
                "资源共享",
                "逻辑简化",
                "消除冗余逻辑",
                "优化存储结构"
            ],
            "pipeline_insertion": [
                "分析数据流",
                "确定流水线边界",
                "插入寄存器",
                "调整控制逻辑"
            ],
            "logic_optimization": [
                "布尔简化",
                "常数传播",
                "死代码消除",
                "表达式重构"
            ]
        }

        return action_map.get(operation, ["通用优化"])

    def identify_optimization_opportunities(
        self,
        rtl_code: str,
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """识别优化机会"""
        opportunities = []

        # 基于结构分析识别机会
        if structure["always_blocks"] > 3:
            opportunities.append({
                "type": "pipeline_opportunity",
                "description": "多个always块可能存在流水线机会",
                "confidence": 0.7
            })

        if structure["complexity_score"] > 5:
            opportunities.append({
                "type": "logic_simplification",
                "description": "复杂逻辑可能需要简化",
                "confidence": 0.8
            })

        return opportunities

    def identify_bottlenecks(
        self,
        rtl_code: str,
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        bottlenecks = []

        # 简单的启发式瓶颈识别
        if len(structure["input_ports"]) > 10:
            bottlenecks.append({
                "type": "io_bottleneck",
                "description": "输入端口过多可能影响时序",
                "severity": "medium"
            })

        return bottlenecks

    def assess_complexity(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """评估复杂度"""
        score = structure["complexity_score"]

        return {
            "overall_score": score,
            "level": "high" if score > 10 else "medium" if score > 5 else "low",
            "timing_critical": score > 8,
            "area_constrained": len(structure["internal_signals"]) > 20,
            "power_sensitive": structure["always_blocks"] > 5
        }

    def calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """计算分析置信度"""
        base_confidence = 0.7

        # 基于分析完整性调整置信度
        completeness_factors = [
            bool(analysis.get("structure")),
            bool(analysis.get("optimization_opportunities")),
            bool(analysis.get("bottlenecks")),
            bool(analysis.get("llm_analysis", {}).get("overall_assessment"))
        ]

        completeness_score = sum(completeness_factors) / len(completeness_factors)
        return min(0.95, base_confidence + completeness_score * 0.2)
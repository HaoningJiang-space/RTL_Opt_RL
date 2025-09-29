"""
CodeRewriter Agent - 代码重写智能体
负责根据MetaOptimizer的指令执行具体的RTL代码优化和重写
"""

from typing import Dict, Any, List, Optional
import re
from .base_agent import BaseAgent


class CodeRewriterAgent(BaseAgent):
    """代码重写智能体：执行具体的代码变换和优化"""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, "code_rewriter", **kwargs)

    def _get_agent_config(self) -> Dict[str, Any]:
        """获取代码重写智能体的特定配置"""
        return {
            "rewrite_strategies": [
                "timing_optimization",
                "area_optimization",
                "power_optimization",
                "pipeline_insertion",
                "logic_simplification",
                "resource_sharing"
            ],
            "optimization_techniques": [
                "register_balancing",     # 寄存器平衡
                "logic_restructuring",   # 逻辑重构
                "constant_propagation",  # 常数传播
                "dead_code_elimination", # 死代码消除
                "common_subexpression",  # 公共子表达式消除
                "parallel_expansion"     # 并行展开
            ],
            "max_iterations": 5,
            "safety_checks": True
        }

    def generate_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成代码重写动作"""
        rtl_code = state.get("rtl_code", "")
        meta_instructions = state.get("meta_instructions", {})
        current_step = state.get("step", 0)

        # 执行代码重写
        rewrite_result = self.execute_rewrite(
            rtl_code,
            meta_instructions,
            current_step
        )

        return {
            "action_type": "code_rewrite",
            "original_code": rtl_code,
            "optimized_code": rewrite_result["optimized_code"],
            "applied_optimizations": rewrite_result["applied_optimizations"],
            "rewrite_log": rewrite_result["rewrite_log"],
            "confidence": rewrite_result["confidence"],
            "safety_check": rewrite_result["safety_check"]
        }

    def execute_rewrite(
        self,
        rtl_code: str,
        meta_instructions: Dict[str, Any],
        step: int
    ) -> Dict[str, Any]:
        """根据元指令执行代码重写"""

        strategy_type = meta_instructions.get("strategy_type", "mixed_optimization")
        optimization_sequence = meta_instructions.get("optimization_sequence", [])
        constraints = meta_instructions.get("constraints", {})

        # 选择当前步骤的优化操作
        current_operation = self._get_current_operation(optimization_sequence, step)

        # 执行具体的重写策略
        if strategy_type == "timing_optimization":
            result = self.optimize_timing(rtl_code, current_operation, constraints)
        elif strategy_type == "area_optimization":
            result = self.optimize_area(rtl_code, current_operation, constraints)
        elif strategy_type == "power_optimization":
            result = self.optimize_power(rtl_code, current_operation, constraints)
        elif strategy_type == "pipeline_insertion":
            result = self.insert_pipeline(rtl_code, current_operation, constraints)
        else:
            result = self.mixed_optimization(rtl_code, current_operation, constraints)

        # 安全性检查
        safety_result = self._perform_safety_check(rtl_code, result["optimized_code"])
        result["safety_check"] = safety_result

        return result

    def _get_current_operation(self, optimization_sequence: List[Dict], step: int) -> Dict[str, Any]:
        """获取当前步骤的优化操作"""
        if not optimization_sequence:
            return {"operation": "general_optimization", "target": ["general"]}

        # 找到对应步骤的操作
        for op in optimization_sequence:
            if op.get("step", 1) == step + 1:  # step从0开始，序列从1开始
                return op

        # 如果没找到，返回第一个操作
        return optimization_sequence[0] if optimization_sequence else {
            "operation": "general_optimization",
            "target": ["general"]
        }

    def optimize_timing(
        self,
        rtl_code: str,
        operation: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """时序优化实现"""

        optimization_prompt = self._build_timing_optimization_prompt(
            rtl_code, operation, constraints
        )

        optimized_code = self.generate_text(
            optimization_prompt,
            max_new_tokens=1500,
            temperature=0.3
        )

        # 提取Verilog代码
        clean_code = self.extract_verilog_code(optimized_code)

        return {
            "optimized_code": clean_code if clean_code else rtl_code,
            "applied_optimizations": ["timing_optimization"],
            "rewrite_log": [f"应用时序优化: {operation.get('operation', 'general')}"],
            "confidence": self._calculate_rewrite_confidence(clean_code, rtl_code)
        }

    def optimize_area(
        self,
        rtl_code: str,
        operation: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """面积优化实现"""

        optimization_prompt = self._build_area_optimization_prompt(
            rtl_code, operation, constraints
        )

        optimized_code = self.generate_text(
            optimization_prompt,
            max_new_tokens=1500,
            temperature=0.3
        )

        clean_code = self.extract_verilog_code(optimized_code)

        return {
            "optimized_code": clean_code if clean_code else rtl_code,
            "applied_optimizations": ["area_optimization"],
            "rewrite_log": [f"应用面积优化: {operation.get('operation', 'general')}"],
            "confidence": self._calculate_rewrite_confidence(clean_code, rtl_code)
        }

    def optimize_power(
        self,
        rtl_code: str,
        operation: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """功耗优化实现"""

        optimization_prompt = self._build_power_optimization_prompt(
            rtl_code, operation, constraints
        )

        optimized_code = self.generate_text(
            optimization_prompt,
            max_new_tokens=1500,
            temperature=0.3
        )

        clean_code = self.extract_verilog_code(optimized_code)

        return {
            "optimized_code": clean_code if clean_code else rtl_code,
            "applied_optimizations": ["power_optimization"],
            "rewrite_log": [f"应用功耗优化: {operation.get('operation', 'general')}"],
            "confidence": self._calculate_rewrite_confidence(clean_code, rtl_code)
        }

    def insert_pipeline(
        self,
        rtl_code: str,
        operation: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """流水线插入优化"""

        optimization_prompt = self._build_pipeline_prompt(
            rtl_code, operation, constraints
        )

        optimized_code = self.generate_text(
            optimization_prompt,
            max_new_tokens=1800,
            temperature=0.2
        )

        clean_code = self.extract_verilog_code(optimized_code)

        return {
            "optimized_code": clean_code if clean_code else rtl_code,
            "applied_optimizations": ["pipeline_insertion"],
            "rewrite_log": [f"应用流水线插入: {operation.get('operation', 'general')}"],
            "confidence": self._calculate_rewrite_confidence(clean_code, rtl_code)
        }

    def mixed_optimization(
        self,
        rtl_code: str,
        operation: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """混合优化策略"""

        optimization_prompt = self._build_mixed_optimization_prompt(
            rtl_code, operation, constraints
        )

        optimized_code = self.generate_text(
            optimization_prompt,
            max_new_tokens=1500,
            temperature=0.4
        )

        clean_code = self.extract_verilog_code(optimized_code)

        return {
            "optimized_code": clean_code if clean_code else rtl_code,
            "applied_optimizations": ["mixed_optimization"],
            "rewrite_log": [f"应用混合优化: {operation.get('operation', 'general')}"],
            "confidence": self._calculate_rewrite_confidence(clean_code, rtl_code)
        }

    def _build_timing_optimization_prompt(
        self,
        rtl_code: str,
        operation: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """构建时序优化提示"""

        prompt_template = """作为RTL时序优化专家，请优化以下Verilog代码的时序性能。

当前优化操作: {operation}
目标区域: {targets}

原始Verilog代码:
```verilog
{rtl_code}
```

时序优化要求:
1. **关键路径优化**: 识别和缩短关键时序路径
2. **流水线机会**: 在适当位置插入流水线寄存器
3. **并行化**: 将串行操作转换为并行操作
4. **逻辑深度减少**: 减少组合逻辑的层数
5. **寄存器平衡**: 平衡各流水级的逻辑深度

优化约束:
- 保持功能等效性
- 维护接口兼容性
- 时序约束: {timing_constraints}

请生成优化后的Verilog代码，并在注释中说明优化要点:

```verilog"""

        return self.format_prompt(
            prompt_template,
            operation=operation.get("operation", "timing_optimization"),
            targets=", ".join(operation.get("target", ["general"])),
            rtl_code=rtl_code[:1500],
            timing_constraints=constraints.get("timing_constraints", "无特殊约束")
        )

    def _build_area_optimization_prompt(
        self,
        rtl_code: str,
        operation: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """构建面积优化提示"""

        prompt_template = """作为RTL面积优化专家，请优化以下Verilog代码的面积使用。

当前优化操作: {operation}
目标区域: {targets}

原始Verilog代码:
```verilog
{rtl_code}
```

面积优化技术:
1. **资源共享**: 共享乘法器、加法器等资源
2. **逻辑简化**: 简化布尔表达式和逻辑结构
3. **存储优化**: 优化存储器的使用和组织
4. **多路复用**: 使用MUX代替多个独立单元
5. **常数优化**: 优化常数乘法和特殊值处理

优化约束:
- 保持功能正确性
- 面积预算: {area_constraints}
- 性能影响最小化

请生成优化后的Verilog代码:

```verilog"""

        return self.format_prompt(
            prompt_template,
            operation=operation.get("operation", "area_optimization"),
            targets=", ".join(operation.get("target", ["general"])),
            rtl_code=rtl_code[:1500],
            area_constraints=constraints.get("area_budget", "无特殊限制")
        )

    def _build_power_optimization_prompt(
        self,
        rtl_code: str,
        operation: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """构建功耗优化提示"""

        prompt_template = """作为RTL功耗优化专家，请优化以下Verilog代码的功耗特性。

当前优化操作: {operation}
目标区域: {targets}

原始Verilog代码:
```verilog
{rtl_code}
```

功耗优化策略:
1. **时钟门控**: 在不需要时禁用时钟
2. **动态功耗**: 减少开关活动
3. **静态功耗**: 减少泄漏电流
4. **电源管理**: 添加电源门控逻辑
5. **低功耗设计**: 使用低功耗设计模式

优化重点:
- 减少不必要的开关活动
- 使用使能信号控制计算
- 优化数据路径宽度
- 功耗预算: {power_constraints}

请生成功耗优化后的Verilog代码:

```verilog"""

        return self.format_prompt(
            prompt_template,
            operation=operation.get("operation", "power_optimization"),
            targets=", ".join(operation.get("target", ["general"])),
            rtl_code=rtl_code[:1500],
            power_constraints=constraints.get("power_budget", "无特殊限制")
        )

    def _build_pipeline_prompt(
        self,
        rtl_code: str,
        operation: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """构建流水线优化提示"""

        prompt_template = """作为RTL流水线设计专家，请为以下Verilog代码添加适当的流水线结构。

当前优化操作: {operation}
目标区域: {targets}

原始Verilog代码:
```verilog
{rtl_code}
```

流水线设计原则:
1. **流水级划分**: 在适当位置插入流水线寄存器
2. **数据流分析**: 确保数据依赖关系正确
3. **控制信号**: 添加相应的控制和使能信号
4. **冒险处理**: 处理数据冒险和控制冒险
5. **性能平衡**: 平衡各流水级的延迟

流水线要求:
- 保持数据完整性
- 添加适当的流水线控制
- 处理数据依赖
- 维护接口时序

请生成流水线化后的Verilog代码:

```verilog"""

        return self.format_prompt(
            prompt_template,
            operation=operation.get("operation", "pipeline_insertion"),
            targets=", ".join(operation.get("target", ["general"])),
            rtl_code=rtl_code[:1500]
        )

    def _build_mixed_optimization_prompt(
        self,
        rtl_code: str,
        operation: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """构建混合优化提示"""

        prompt_template = """作为RTL综合优化专家，请对以下Verilog代码进行综合优化。

当前优化操作: {operation}
目标区域: {targets}

原始Verilog代码:
```verilog
{rtl_code}
```

综合优化策略（平衡时序、面积、功耗）:
1. **时序优化**: 适度的流水线和并行化
2. **面积优化**: 合理的资源共享和逻辑简化
3. **功耗优化**: 减少不必要的开关活动
4. **可读性**: 保持代码的可读性和维护性
5. **综合友好**: 利于EDA工具综合的代码结构

优化目标:
- 20% 时序改善
- 15% 面积减少
- 10% 功耗降低
- 保持功能正确性

请生成综合优化后的Verilog代码:

```verilog"""

        return self.format_prompt(
            prompt_template,
            operation=operation.get("operation", "mixed_optimization"),
            targets=", ".join(operation.get("target", ["general"])),
            rtl_code=rtl_code[:1500]
        )

    def _perform_safety_check(self, original_code: str, optimized_code: str) -> Dict[str, Any]:
        """执行安全性检查"""

        safety_result = {
            "passed": True,
            "issues": [],
            "warnings": []
        }

        if not optimized_code or optimized_code.strip() == "":
            safety_result["passed"] = False
            safety_result["issues"].append("优化后代码为空")
            return safety_result

        # 检查基本语法结构
        if "module" not in optimized_code:
            safety_result["passed"] = False
            safety_result["issues"].append("缺少module声明")

        if "endmodule" not in optimized_code:
            safety_result["passed"] = False
            safety_result["issues"].append("缺少endmodule声明")

        # 检查端口一致性
        original_ports = self._extract_ports(original_code)
        optimized_ports = self._extract_ports(optimized_code)

        if original_ports and optimized_ports:
            if set(original_ports) != set(optimized_ports):
                safety_result["warnings"].append("端口可能发生变化，需要验证")

        # 检查代码长度变化
        length_ratio = len(optimized_code) / max(len(original_code), 1)
        if length_ratio > 3.0:
            safety_result["warnings"].append("代码长度显著增加，可能存在问题")
        elif length_ratio < 0.2:
            safety_result["warnings"].append("代码长度显著减少，可能丢失功能")

        return safety_result

    def _extract_ports(self, rtl_code: str) -> List[str]:
        """提取端口名称"""
        ports = []

        # 简单的端口提取正则表达式
        port_patterns = [
            r'input\s+(?:\w+\s+)?(\w+)',
            r'output\s+(?:\w+\s+)?(\w+)',
            r'inout\s+(?:\w+\s+)?(\w+)'
        ]

        for pattern in port_patterns:
            matches = re.findall(pattern, rtl_code)
            ports.extend(matches)

        return ports

    def _calculate_rewrite_confidence(self, optimized_code: str, original_code: str) -> float:
        """计算重写置信度"""

        if not optimized_code or optimized_code.strip() == "":
            return 0.0

        # 基础置信度
        base_confidence = 0.7

        # 基于代码质量的置信度调整
        quality_factors = []

        # 检查是否包含module结构
        if "module" in optimized_code and "endmodule" in optimized_code:
            quality_factors.append(0.2)

        # 检查是否有合理的代码长度
        length_ratio = len(optimized_code) / max(len(original_code), 1)
        if 0.5 <= length_ratio <= 2.0:
            quality_factors.append(0.1)

        # 检查是否包含Verilog关键字
        verilog_keywords = ['always', 'assign', 'reg', 'wire', 'input', 'output']
        keyword_count = sum(1 for keyword in verilog_keywords if keyword in optimized_code)
        if keyword_count >= 2:
            quality_factors.append(0.1)

        # 计算最终置信度
        confidence = base_confidence + sum(quality_factors)
        return min(0.95, confidence)

    def validate_optimization_result(
        self,
        original_code: str,
        optimized_code: str,
        optimization_type: str
    ) -> Dict[str, Any]:
        """验证优化结果"""

        validation_result = {
            "syntax_valid": True,
            "structure_preserved": True,
            "optimization_applied": True,
            "confidence_score": 0.0,
            "recommendations": []
        }

        # 语法验证
        if not self._basic_syntax_check(optimized_code):
            validation_result["syntax_valid"] = False
            validation_result["recommendations"].append("检查Verilog语法错误")

        # 结构保持验证
        if not self._structure_preservation_check(original_code, optimized_code):
            validation_result["structure_preserved"] = False
            validation_result["recommendations"].append("验证模块结构是否保持")

        # 优化应用验证
        if not self._optimization_applied_check(original_code, optimized_code, optimization_type):
            validation_result["optimization_applied"] = False
            validation_result["recommendations"].append(f"确认{optimization_type}优化是否正确应用")

        # 计算总体置信度
        validation_result["confidence_score"] = self._calculate_rewrite_confidence(
            optimized_code, original_code
        )

        return validation_result

    def _basic_syntax_check(self, code: str) -> bool:
        """基础语法检查"""
        required_elements = ['module', 'endmodule']
        return all(element in code for element in required_elements)

    def _structure_preservation_check(self, original: str, optimized: str) -> bool:
        """结构保持检查"""
        # 简化的结构检查，实际应该更复杂
        original_ports = self._extract_ports(original)
        optimized_ports = self._extract_ports(optimized)

        return len(original_ports) == len(optimized_ports)

    def _optimization_applied_check(self, original: str, optimized: str, opt_type: str) -> bool:
        """优化应用检查"""
        # 简化检查，实际应该根据优化类型做更具体的验证
        return len(optimized) != len(original)  # 至少代码有变化
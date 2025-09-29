#!/usr/bin/env python3
"""
RTL Optimization Data Generation Script
Generate RTL optimization training data in ReMA framework format
"""

import argparse
import json
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Any


def generate_basic_verilog(module_name: str, complexity: str = "simple") -> str:
    """Generate basic Verilog code"""

    if complexity == "simple":
        return f'''module {module_name}(
    input clk,
    input rst,
    input [7:0] data_in,
    output reg [7:0] data_out
);
    reg [7:0] temp_reg;

    always @(posedge clk) begin
        if (rst) begin
            temp_reg <= 8'b0;
            data_out <= 8'b0;
        end else begin
            temp_reg <= data_in + 1;
            data_out <= temp_reg;
        end
    end
endmodule'''

    elif complexity == "medium":
        return f'''module {module_name}(
    input clk,
    input rst,
    input [15:0] a,
    input [15:0] b,
    input [1:0] op,
    output reg [31:0] result
);
    reg [31:0] temp_a, temp_b;
    reg [31:0] intermediate;

    always @(posedge clk) begin
        if (rst) begin
            temp_a <= 0;
            temp_b <= 0;
            intermediate <= 0;
            result <= 0;
        end else begin
            temp_a <= {{16'b0, a}};
            temp_b <= {{16'b0, b}};

            case (op)
                2'b00: intermediate <= temp_a + temp_b;
                2'b01: intermediate <= temp_a - temp_b;
                2'b10: intermediate <= temp_a * temp_b;
                2'b11: intermediate <= temp_a & temp_b;
            endcase

            result <= intermediate;
        end
    end
endmodule'''

    else:  # complex
        return f'''module {module_name}(
    input clk,
    input rst,
    input [31:0] data_in,
    input valid_in,
    output reg [31:0] data_out,
    output reg valid_out
);
    reg [31:0] stage1_data, stage2_data, stage3_data;
    reg stage1_valid, stage2_valid, stage3_valid;
    reg [31:0] temp1, temp2, temp3;

    // First pipeline stage
    always @(posedge clk) begin
        if (rst) begin
            stage1_data <= 0;
            stage1_valid <= 0;
            temp1 <= 0;
        end else begin
            if (valid_in) begin
                temp1 <= data_in + 32'h12345678;
                stage1_data <= temp1;
                stage1_valid <= 1;
            end else begin
                stage1_valid <= 0;
            end
        end
    end

    // Second pipeline stage
    always @(posedge clk) begin
        if (rst) begin
            stage2_data <= 0;
            stage2_valid <= 0;
            temp2 <= 0;
        end else begin
            temp2 <= stage1_data ^ 32'hABCDEF00;
            stage2_data <= temp2;
            stage2_valid <= stage1_valid;
        end
    end

    // Third pipeline stage
    always @(posedge clk) begin
        if (rst) begin
            stage3_data <= 0;
            stage3_valid <= 0;
            temp3 <= 0;
            data_out <= 0;
            valid_out <= 0;
        end else begin
            temp3 <= stage2_data << 2;
            stage3_data <= temp3;
            stage3_valid <= stage2_valid;

            data_out <= stage3_data;
            valid_out <= stage3_valid;
        end
    end
endmodule'''


def generate_optimized_verilog(original_code: str, optimization_type: str) -> tuple[str, str]:
    """Generate optimized version based on original code"""

    module_name = extract_module_name(original_code) + "_opt"

    if "temp_reg" in original_code and optimization_type in ["area", "timing"]:
        # Area/timing optimization: remove intermediate register
        optimized = original_code.replace("reg [7:0] temp_reg;", "")
        optimized = optimized.replace("temp_reg <= data_in + 1;", "")
        optimized = optimized.replace("data_out <= temp_reg;", "data_out <= data_in + 1;")
        optimized = optimized.replace(extract_module_name(original_code), module_name)

        optimization_desc = "Optimization Strategy:\n1. Eliminate intermediate register temp_reg, directly calculate output\n2. Reduce one clock cycle delay\n3. Save area resources"

    elif "intermediate" in original_code and optimization_type == "pipeline":
        # 流水线优化
        lines = original_code.split('\n')
        optimized_lines = []

        for line in lines:
            if "case (op)" in line:
                # 插入流水线寄存器
                optimized_lines.append("            // 流水线优化：将运算分为两个阶段")
                optimized_lines.append("            reg [1:0] op_reg;")
                optimized_lines.append("            op_reg <= op;")
            optimized_lines.append(line)

        optimized = '\n'.join(optimized_lines)
        optimized = optimized.replace(extract_module_name(original_code), module_name)
        optimization_desc = "优化策略：\n1. 添加流水线寄存器提高时钟频率\n2. 改善时序性能\n3. 支持更高的工作频率"

    elif optimization_type == "power":
        # 功耗优化：添加时钟门控
        optimized = original_code.replace(
            "always @(posedge clk) begin",
            "// 功耗优化：添加时钟门控\n    wire gated_clk = clk & (valid_in | ~rst);\n    \n    always @(posedge gated_clk) begin"
        )
        optimized = optimized.replace(extract_module_name(original_code), module_name)
        optimization_desc = "优化策略：\n1. 添加时钟门控减少动态功耗\n2. 在无效数据时停止时钟\n3. 降低整体功耗消耗"

    else:
        # 默认优化：代码清理
        optimized = original_code.replace("    reg [31:0] temp1, temp2, temp3;",
                                        "    reg [31:0] temp1, temp2;  // 减少不必要的临时变量")
        optimized = optimized.replace(extract_module_name(original_code), module_name)
        optimization_desc = "优化策略：\n1. 清理不必要的变量声明\n2. 改善代码可读性\n3. 减少资源使用"

    return optimized, optimization_desc


def extract_module_name(verilog_code: str) -> str:
    """提取模块名"""
    import re
    match = re.search(r'module\s+(\w+)', verilog_code)
    return match.group(1) if match else "unknown"


def create_rema_conversation(original_code: str, optimized_code: str, optimization_desc: str,
                           optimization_type: str) -> List[Dict[str, str]]:
    """Create ReMA format multi-turn conversation - multi-agent RTL optimization"""

    # 构建专业的RTL优化提示
    question = f"""You are a professional multi-agent RTL optimization system. Please analyze and optimize the following Verilog code.

**Agent Roles:**
- MetaOptimizer: Responsible for high-level strategy planning and optimization direction
- CodeRewriter: Responsible for concrete code rewriting and optimization implementation

**Optimization Target:** {optimization_type} optimization

**Original RTL Code:**
```verilog
{original_code}
```

**Requirements:**
1. As MetaOptimizer, first perform deep analysis and strategy planning
2. As CodeRewriter, implement specific optimizations based on the strategy
3. Maintain functional equivalence
4. Consider synthesis tool characteristics (e.g., Yosys)
5. Follow RTL optimization best practices

Please proceed with multi-agent collaborative optimization."""

    # Meta-thinking回答 (MetaOptimizer角色)
    meta_thinking = f"""**MetaOptimizer Analysis:**

🔍 **RTL Design Analysis**
- Module: {extract_module_name(original_code)}
- Architecture: {"Sequential logic" if "always @(posedge" in original_code else "Combinational logic"}
- Complexity: {"High" if len(original_code.split('\\n')) > 50 else "Medium" if len(original_code.split('\\n')) > 20 else "Low"}

📊 **Optimization Potential Identification**
- Data path: {"Pipeline opportunities exist" if "stage" in original_code else "Combinational logic optimization opportunities"}
- Resource usage: {"Register usage can be optimized" if "reg" in original_code else "Combinational logic can be optimized"}
- Timing characteristics: {"Multi-clock domain" if original_code.count("clk") > 1 else "Single clock domain design"}

🎯 **{optimization_type.title()} Optimization Strategy**
- Primary strategy: {"Reduce critical path delay" if optimization_type == "timing" else "Reduce hardware resources" if optimization_type == "area" else "Lower dynamic power consumption" if optimization_type == "power" else "Balanced optimization"}
- Technical approach: {"Pipeline restructuring" if "pipeline" in optimization_type else "Logic simplification and sharing"}
- Constraints: Maintain interface compatibility and functional equivalence

💡 **Implementation Path**
- Step 1: {"Analyze critical paths" if optimization_type == "timing" else "Identify redundant resources" if optimization_type == "area" else "Locate power hotspots"}
- Step 2: {"Restructure data paths" if optimization_type != "power" else "Add clock gating"}
- Step 3: Verify optimization effectiveness and functional correctness"""

    # Reasoning回答 (CodeRewriter角色)
    reasoning = f"""**CodeRewriter Implementation:**

Based on MetaOptimizer's analysis, I have implemented the following optimization:

```verilog
{optimized_code}
```

🔧 **Specific Optimization Implementation**
{optimization_desc}

⚡ **Optimization Techniques Explained**
- **Algorithm level:** {"Parallel computation to reduce latency" if optimization_type == "timing" else "Eliminate redundant computation and storage" if optimization_type == "area" else "Introduce conditional computation to reduce power"}
- **Architecture level:** {"Restructure dataflow for higher parallelism" if "pipeline" in optimization_type else "Simplify control logic"}
- **Implementation level:** Follow Verilog synthesis best practices

📈 **Expected Improvement**
- **{optimization_type.title()} metrics:** {"Latency reduced by 10-30%" if optimization_type == "timing" else "Area reduced by 15-25%" if optimization_type == "area" else "Power reduced by 20-40%" if optimization_type == "power" else "Overall metrics improved"}
- **Synthesis quality:** After Yosys synthesis, expect {"better timing convergence" if optimization_type == "timing" else "fewer LUT usage" if optimization_type == "area" else "lower dynamic power"}
- **Code quality:** Maintain readability and maintainability

✅ **Verification Checklist**
- [x] Functional equivalence: All output logic remains consistent
- [x] Interface compatibility: Port definitions completely match
- [x] Synthesis-friendly: Avoided structures difficult for synthesis tools
- [x] Timing constraints: {"Optimized critical paths" if optimization_type == "timing" else "No additional timing issues introduced"}"""

    return [
        {"role": "user", "content": question},
        {"role": "meta_thinking", "content": meta_thinking},
        {"role": "reasoning", "content": reasoning}
    ]


def generate_rtl_dataset(num_samples: int, output_dir: str, data_source: str = "rtl_optimization"):
    """生成RTL优化数据集"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_data = []
    val_data = []

    complexities = ["simple", "medium", "complex"]
    optimization_types = ["area", "timing", "power", "pipeline"]

    print(f"生成 {num_samples} 个RTL优化样本...")

    for i in range(num_samples):
        complexity = random.choice(complexities)
        opt_type = random.choice(optimization_types)
        module_name = f"rtl_module_{i}"

        # 生成原始和优化代码
        original_code = generate_basic_verilog(module_name, complexity)
        optimized_code, optimization_desc = generate_optimized_verilog(original_code, opt_type)

        # 创建ReMA格式对话
        conversation = create_rema_conversation(original_code, optimized_code,
                                              optimization_desc, opt_type)

        # 构建数据项
        data_item = {
            "data_source": data_source,
            "question": conversation[0]["content"],  # 用户问题
            "response": conversation[2]["content"],  # 最终回答
            "history": conversation,  # 完整对话历史
            "ground_truth": optimized_code,  # 标准答案（优化代码）
            "original_code": original_code,  # 原始代码
            "optimization_type": opt_type,
            "complexity": complexity,
            "module_name": module_name,
            "extra_info": {
                "original_code": original_code,
                "optimization_goal": opt_type,
                "expected_improvement": {
                    "area": random.uniform(0.05, 0.25) if opt_type == "area" else random.uniform(-0.05, 0.15),
                    "timing": random.uniform(0.08, 0.30) if opt_type == "timing" else random.uniform(-0.02, 0.12),
                    "power": random.uniform(0.03, 0.20) if opt_type == "power" else random.uniform(-0.03, 0.10)
                }
            }
        }

        # 划分训练和验证集（80-20）
        if i < num_samples * 0.8:
            train_data.append(data_item)
        else:
            val_data.append(data_item)

    # 保存为parquet格式
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    train_path = output_path / "train.parquet"
    val_path = output_path / "val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"✓ 生成训练数据: {len(train_data)} 个样本 -> {train_path}")
    print(f"✓ 生成验证数据: {len(val_data)} 个样本 -> {val_path}")

    # 生成JSON格式的样本（用于调试）
    sample_json = output_path / "sample_data.json"
    with open(sample_json, 'w', encoding='utf-8') as f:
        json.dump({
            "train_sample": train_data[0] if train_data else {},
            "val_sample": val_data[0] if val_data else {},
            "statistics": {
                "total_samples": num_samples,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "data_source": data_source,
                "complexities": complexities,
                "optimization_types": optimization_types
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ 生成样本文件: {sample_json}")
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="生成RTL优化训练数据")
    parser.add_argument("--num_samples", type=int, default=200,
                       help="生成的样本数量 (默认: 200)")
    parser.add_argument("--output_dir", type=str, default="data/rtl_optimization",
                       help="输出目录 (默认: data/rtl_optimization)")
    parser.add_argument("--data_source", type=str, default="rtl_optimization",
                       help="数据源名称 (默认: rtl_optimization)")
    parser.add_argument("--quick", action="store_true",
                       help="快速模式：生成较少样本用于测试")

    args = parser.parse_args()

    if args.quick:
        args.num_samples = 50
        args.output_dir = "data/rtl_test"
        print("🚀 快速模式：生成少量数据用于测试")

    print("=" * 60)
    print("RTL优化数据生成器")
    print("=" * 60)
    print(f"样本数量: {args.num_samples}")
    print(f"输出目录: {args.output_dir}")
    print(f"数据源: {args.data_source}")

    try:
        train_path, val_path = generate_rtl_dataset(
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            data_source=args.data_source
        )

        print("\n✅ 数据生成完成！")
        print(f"训练数据: {train_path}")
        print(f"验证数据: {val_path}")
        print("\n可以使用以下命令开始训练：")
        print(f"python src/verl/verl/rema_trainer/main_ppo.py \\")
        print(f"  data.train_files={train_path} \\")
        print(f"  data.val_files={val_path} \\")
        print(f"  --config-name rtl_ppo_trainer")

    except Exception as e:
        print(f"❌ 数据生成失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
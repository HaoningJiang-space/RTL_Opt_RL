#!/usr/bin/env python3
"""
RTL奖励函数测试脚本
验证RTL奖励计算功能是否正常工作
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_reward_function():
    """测试RTL奖励函数"""
    print("=" * 60)
    print("RTL奖励函数测试")
    print("=" * 60)

    try:
        from verl.utils.reward_score.rtl_optimization import compute_score
        print("✓ 成功导入RTL奖励函数")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

    # 测试案例1: 基础Verilog代码
    test_cases = [
        {
            "name": "基础正确代码",
            "data_source": "rtl_optimization",
            "solution": """这是优化后的Verilog代码：

```verilog
module test_basic(
    input clk,
    input rst,
    input [7:0] data_in,
    output reg [7:0] data_out
);
    always @(posedge clk) begin
        if (rst)
            data_out <= 8'b0;
        else
            data_out <= data_in + 1;
    end
endmodule
```

优化说明：直接计算输出，减少延迟。""",
            "ground_truth": """module test_original(
    input clk,
    input rst,
    input [7:0] data_in,
    output reg [7:0] data_out
);
    reg [7:0] temp;
    always @(posedge clk) begin
        if (rst) begin
            temp <= 8'b0;
            data_out <= 8'b0;
        end else begin
            temp <= data_in + 1;
            data_out <= temp;
        end
    end
endmodule""",
            "expected_range": (0.3, 1.0)
        },

        {
            "name": "语法错误代码",
            "data_source": "rtl_optimization",
            "solution": """这是有问题的代码：

```verilog
module broken(
    input clk
    output data  // 缺少分号和类型
);
    always @(posedge clk
        data <= 1;  // 缺少右括号
endmodule  // 缺少end
```""",
            "ground_truth": "module good(input clk, output reg data); always @(posedge clk) data <= 1; endmodule",
            "expected_range": (0.0, 0.2)
        },

        {
            "name": "无代码内容",
            "data_source": "rtl_optimization",
            "solution": "这里没有任何代码，只是文字说明。",
            "ground_truth": "module test(input clk); endmodule",
            "expected_range": (0.0, 0.1)
        }
    ]

    all_passed = True

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: {test_case['name']}")
        print("-" * 40)

        try:
            score = compute_score(
                data_source=test_case["data_source"],
                solution_str=test_case["solution"],
                ground_truth=test_case["ground_truth"],
                extra_info={"original_code": test_case["ground_truth"]}
            )

            print(f"得分: {score:.3f}")

            expected_min, expected_max = test_case["expected_range"]

            if expected_min <= score <= expected_max:
                print(f"✓ 通过 (期望范围: {expected_min:.1f}-{expected_max:.1f})")
            else:
                print(f"✗ 失败 (期望范围: {expected_min:.1f}-{expected_max:.1f})")
                all_passed = False

        except Exception as e:
            print(f"✗ 异常: {e}")
            all_passed = False

    return all_passed


def test_verification_tools():
    """测试验证工具"""
    print("\n" + "=" * 60)
    print("验证工具测试")
    print("=" * 60)

    try:
        from verl.utils.reward_score.rtl_optimization import get_verification_tools
        tools = get_verification_tools()
        print("✓ 成功创建验证工具实例")

        # 测试工具检测
        print("\n可用工具:")
        for tool, available in tools.tools.items():
            status = "✓ 可用" if available else "✗ 不可用"
            print(f"  {tool}: {status}")

        # 测试语法验证
        test_code = """module test(input clk, output reg [7:0] data);
    always @(posedge clk) data <= 8'h42;
endmodule"""

        print(f"\n测试语法验证...")
        syntax_result = tools.verify_syntax(test_code)
        print(f"语法检查结果: {syntax_result['success']}")
        print(f"检查方法: {syntax_result['method']}")

        if tools.tools.get('yosys', False):
            print(f"\n测试综合...")
            synth_result = tools.synthesize_and_analyze(test_code)
            print(f"综合结果: {synth_result['success']}")
            if synth_result['success']:
                print(f"统计信息: {synth_result.get('stats', {})}")

        return True

    except Exception as e:
        print(f"✗ 验证工具测试失败: {e}")
        return False


def test_integration_with_rema():
    """测试与ReMA框架集成"""
    print("\n" + "=" * 60)
    print("ReMA框架集成测试")
    print("=" * 60)

    try:
        # 测试默认compute_score函数
        from verl.utils.reward_score import _default_compute_score
        print("✓ 成功导入ReMA默认奖励函数")

        # 测试RTL数据源
        test_solution = """优化后的代码：
```verilog
module simple(input clk, output reg data);
always @(posedge clk) data <= 1'b1;
endmodule
```"""

        test_ground_truth = "module original(input clk, output reg data); reg temp; always @(posedge clk) begin temp <= 1'b1; data <= temp; end endmodule"

        for data_source in ['rtl_optimization', 'rtl_math', 'rtl_generation']:
            print(f"\n测试数据源: {data_source}")
            score = _default_compute_score(
                data_source=data_source,
                solution_str=test_solution,
                ground_truth=test_ground_truth,
                extra_info={"original_code": test_ground_truth}
            )
            print(f"得分: {score:.3f}")

            if 0.0 <= score <= 1.0:
                print("✓ 分数范围正常")
            else:
                print(f"✗ 分数范围异常: {score}")
                return False

        return True

    except Exception as e:
        print(f"✗ ReMA集成测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("RTL多智能体优化系统 - 奖励函数测试")
    print("基于ReMA框架")

    test_results = []

    # 执行各项测试
    test_results.append(("RTL奖励函数", test_reward_function()))
    test_results.append(("验证工具", test_verification_tools()))
    test_results.append(("ReMA集成", test_integration_with_rema()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_passed = True
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！RTL奖励系统可以正常使用。")
        print("\n可以使用以下命令开始训练：")
        print("bash scripts/rtl/train_rtl_rema.sh --quick-test --generate-data")
        return 0
    else:
        print("❌ 存在测试失败，请检查配置和依赖。")
        return 1


if __name__ == "__main__":
    exit(main())
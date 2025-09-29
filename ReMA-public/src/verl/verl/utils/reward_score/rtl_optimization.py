"""
RTL Optimization Reward Calculation Module
Designed specifically for RTL code optimization tasks, integrated into ReMA framework's reward_score system
"""

import subprocess
import tempfile
import os
import re
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


logger = logging.getLogger(__name__)


class RTLVerificationTools:
    """RTL verification tools collection"""

    def __init__(self):
        self.tools = self._detect_tools()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)

    def _detect_tools(self) -> Dict[str, bool]:
        """Detect available verification tools"""
        tools = {}
        for tool in ['verilator', 'yosys', 'iverilog']:
            try:
                result = subprocess.run([tool, '--version'],
                                      capture_output=True, text=True, timeout=5)
                tools[tool] = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                tools[tool] = False
        return tools

    def verify_syntax(self, verilog_code: str) -> Dict[str, Any]:
        """Syntax verification"""
        if not self.tools.get('verilator', False):
            # Simple syntax check
            return {
                'success': 'module' in verilog_code and 'endmodule' in verilog_code,
                'method': 'simple'
            }

        temp_file = self.temp_dir / f"syntax_{hash(verilog_code) % 10000}.v"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(verilog_code)

            result = subprocess.run([
                'verilator', '--lint-only', '-Wall', str(temp_file)
            ], capture_output=True, text=True, timeout=30)

            return {
                'success': result.returncode == 0,
                'method': 'verilator',
                'stderr': result.stderr
            }
        except Exception as e:
            return {'success': False, 'method': 'verilator', 'error': str(e)}
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def synthesize_and_analyze(self, verilog_code: str) -> Dict[str, Any]:
        """Synthesize and analyze resource usage"""
        if not self.tools.get('yosys', False):
            return {'success': False, 'method': 'yosys_unavailable'}

        module_name = self._extract_module_name(verilog_code) or "top"
        temp_file = self.temp_dir / f"synth_{hash(verilog_code) % 10000}.v"
        script_file = self.temp_dir / f"script_{hash(verilog_code) % 10000}.ys"

        try:
            # Write files
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(verilog_code)

            yosys_script = f"""
read_verilog {temp_file}
hierarchy -check -top {module_name}
proc
opt
techmap
opt
stat -json
"""
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(yosys_script)

            # Run Yosys
            result = subprocess.run([
                'yosys', '-s', str(script_file)
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                stats = self._parse_yosys_output(result.stdout)
                return {
                    'success': True,
                    'method': 'yosys',
                    'stats': stats
                }
            else:
                return {
                    'success': False,
                    'method': 'yosys',
                    'stderr': result.stderr
                }

        except Exception as e:
            return {'success': False, 'method': 'yosys', 'error': str(e)}
        finally:
            for f in [temp_file, script_file]:
                if f.exists():
                    f.unlink()

    def _extract_module_name(self, verilog_code: str) -> Optional[str]:
        """Extract module name"""
        match = re.search(r'module\s+(\w+)', verilog_code)
        return match.group(1) if match else None

    def _parse_yosys_output(self, stdout: str) -> Dict[str, int]:
        """Parse Yosys output"""
        stats = {}
        for line in stdout.split('\n'):
            if 'Number of cells:' in line:
                try:
                    stats['cells'] = int(re.search(r'(\d+)', line).group(1))
                except (AttributeError, ValueError):
                    pass
            elif 'Number of wires:' in line:
                try:
                    stats['wires'] = int(re.search(r'(\d+)', line).group(1))
                except (AttributeError, ValueError):
                    pass
        return stats

    def __del__(self):
        """Clean up temporary directory"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass


# 全局验证工具实例
_verification_tools = None


def get_verification_tools():
    """获取验证工具实例（单例模式）"""
    global _verification_tools
    if _verification_tools is None:
        _verification_tools = RTLVerificationTools()
    return _verification_tools


def extract_verilog_code(text: str) -> str:
    """从文本中提取Verilog代码"""
    if not text:
        return ""

    # 查找代码块
    if "```verilog" in text:
        start = text.find("```verilog") + len("```verilog")
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()

    # 查找module关键字
    if "module" in text:
        lines = text.split("\n")
        verilog_lines = []
        in_module = False

        for line in lines:
            if "module" in line and not in_module:
                in_module = True
            if in_module:
                verilog_lines.append(line)
            if "endmodule" in line and in_module:
                break

        if verilog_lines:
            return "\n".join(verilog_lines).strip()

    return text.strip()


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    RTL优化奖励计算函数 - 符合ReMA框架接口

    Args:
        data_source: 数据源标识（如 'rtl_optimization', 'rtl_math', 等）
        solution_str: 模型生成的解答（包含优化后的RTL代码）
        ground_truth: 标准答案（可以是期望的优化结果或原始代码）
        extra_info: 额外信息（如优化目标、约束等）

    Returns:
        float: Reward score (0.0 to 1.0)
    """

    try:
        # Extract optimized Verilog code
        optimized_code = extract_verilog_code(solution_str)
        if not optimized_code:
            logger.warning("Unable to extract Verilog code from solution")
            return 0.0

        # Extract original code (from ground_truth or extra_info)
        original_code = ""
        if extra_info and isinstance(extra_info, dict):
            original_code = extra_info.get('original_code', '') or extract_verilog_code(ground_truth)
        else:
            original_code = extract_verilog_code(ground_truth) if ground_truth else ""

        # Get verification tools
        tools = get_verification_tools()

        # 1. Syntax verification (required, weight 40%)
        syntax_result = tools.verify_syntax(optimized_code)
        syntax_score = 1.0 if syntax_result['success'] else 0.0

        # If syntax is incorrect, return low score directly
        if syntax_score == 0.0:
            return 0.1  # Give some basic points to avoid complete zero

        # 2. Synthesis verification (weight 30%)
        synthesis_result = tools.synthesize_and_analyze(optimized_code)
        synthesis_score = 1.0 if synthesis_result['success'] else 0.5  # Give some points even if synthesis fails

        # 3. Optimization effect evaluation (weight 30%)
        optimization_score = 0.5  # Default score

        if original_code and synthesis_result['success'] and original_code != optimized_code:
            # Compare original code and optimized code
            original_result = tools.synthesize_and_analyze(original_code)
            if original_result['success']:
                optimization_score = calculate_improvement_score(
                    original_result['stats'],
                    synthesis_result['stats']
                )

        # Weighted calculation of total score
        total_score = (
            syntax_score * 0.4 +
            synthesis_score * 0.3 +
            optimization_score * 0.3
        )

        # Special rewards
        bonus = 0.0

        # Code quality reward
        if check_code_quality(optimized_code):
            bonus += 0.05

        # Multi-agent format reward (if applicable)
        if data_source.startswith('rtl_') and 'meta_thinking' in solution_str:
            bonus += 0.05

        final_score = min(1.0, total_score + bonus)

        logger.info(f"RTL reward calculation: syntax={syntax_score:.2f}, synthesis={synthesis_score:.2f}, "
                   f"optimization={optimization_score:.2f}, bonus={bonus:.2f}, total={final_score:.2f}")

        return final_score

    except Exception as e:
        logger.error(f"RTL reward calculation failed: {e}")
        return 0.0


def calculate_improvement_score(original_stats: Dict[str, int], optimized_stats: Dict[str, int]) -> float:
    """Calculate optimization improvement score"""

    if not original_stats or not optimized_stats:
        return 0.5

    improvements = []

    # Area improvement (based on cells count)
    orig_cells = original_stats.get('cells', 0)
    opt_cells = optimized_stats.get('cells', 0)

    if orig_cells > 0:
        area_improvement = max(-0.5, min(0.5, (orig_cells - opt_cells) / orig_cells))
        improvements.append(area_improvement)

    # Wire improvement (based on wires count)
    orig_wires = original_stats.get('wires', 0)
    opt_wires = optimized_stats.get('wires', 0)

    if orig_wires > 0:
        wire_improvement = max(-0.3, min(0.3, (orig_wires - opt_wires) / orig_wires))
        improvements.append(wire_improvement * 0.5)  # Lower weight

    if improvements:
        # Convert to 0-1 score
        avg_improvement = sum(improvements) / len(improvements)
        return max(0.0, min(1.0, 0.5 + avg_improvement))
    else:
        return 0.5


def check_code_quality(verilog_code: str) -> bool:
    """Check code quality"""
    quality_indicators = [
        'begin' in verilog_code and 'end' in verilog_code,  # Structured code
        len(verilog_code.split('\n')) < 200,  # Not excessively long
        verilog_code.count('always') <= 10,  # Not overly complex
        'clk' in verilog_code or 'clock' in verilog_code,  # Contains clock logic
    ]

    return sum(quality_indicators) >= 2


# ReMA framework specific RTL data source handling
def handle_rtl_data_sources(data_source: str) -> Dict[str, Any]:
    """Handle RTL-related data source configurations"""

    rtl_configs = {
        'rtl_optimization': {
            'syntax_weight': 0.4,
            'synthesis_weight': 0.3,
            'optimization_weight': 0.3
        },
        'rtl_math': {  # RTL mathematical reasoning tasks
            'syntax_weight': 0.5,
            'synthesis_weight': 0.2,
            'optimization_weight': 0.3
        },
        'rtl_generation': {  # RTL code generation tasks
            'syntax_weight': 0.6,
            'synthesis_weight': 0.4,
            'optimization_weight': 0.0
        }
    }

    return rtl_configs.get(data_source, rtl_configs['rtl_optimization'])
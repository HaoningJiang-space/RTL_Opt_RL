"""
RTL优化专用的奖励管理器
集成到ReMA框架中，为RTL代码优化提供专业的奖励信号
"""

import subprocess
import tempfile
import os
import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


class RTLRewardManager:
    """RTL优化奖励管理器，符合ReMA框架的奖励接口"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.work_dir = Path(tempfile.mkdtemp())
        self.work_dir.mkdir(exist_ok=True)

        # 设置日志
        self.logger = logging.getLogger(self.__class__.__name__)

        # 检测验证工具
        self.tools = self._detect_verification_tools()

        # 奖励权重配置
        self.weights = {
            'syntax': self.config.get('syntax_weight', 0.4),
            'synthesis': self.config.get('synthesis_weight', 0.4),
            'ppa': self.config.get('ppa_weight', 0.2)
        }

    def _detect_verification_tools(self) -> Dict[str, Optional[str]]:
        """检测可用的Verilog验证工具"""
        tools = {'verilator': None, 'yosys': None, 'iverilog': None}

        for tool in tools.keys():
            try:
                result = subprocess.run([tool, '--version'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    tools[tool] = tool
                    self.logger.info(f"检测到{tool}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.logger.warning(f"未找到{tool}")

        return tools

    def calculate_reward(self,
                        query: str,
                        response: str,
                        **kwargs) -> Dict[str, Any]:
        """
        计算RTL优化奖励 - 符合ReMA reward function接口

        Args:
            query: 原始RTL代码（作为prompt）
            response: 优化后的RTL代码
            **kwargs: 其他参数

        Returns:
            包含奖励信息的字典
        """

        try:
            # 提取Verilog代码
            original_code = self._extract_verilog_code(query)
            optimized_code = self._extract_verilog_code(response)

            if not original_code or not optimized_code:
                return {
                    'reward': 0.0,
                    'details': {'error': '无法提取有效的Verilog代码'},
                    'success': False
                }

            # 语法验证
            syntax_result = self._verify_syntax(optimized_code)
            syntax_score = 1.0 if syntax_result['success'] else 0.0

            # 综合验证（如果工具可用）
            synthesis_score = 0.0
            synthesis_result = {}
            if self.tools.get('yosys') and syntax_score > 0:
                synthesis_result = self._verify_synthesis(optimized_code)
                synthesis_score = 1.0 if synthesis_result.get('success', False) else 0.0

            # PPA比较（如果综合成功）
            ppa_score = 0.0
            ppa_result = {}
            if synthesis_score > 0 and original_code != optimized_code:
                ppa_result = self._compare_ppa(original_code, optimized_code)
                ppa_score = max(0.0, ppa_result.get('improvement_score', 0.0))

            # 计算总奖励
            total_reward = (
                syntax_score * self.weights['syntax'] +
                synthesis_score * self.weights['synthesis'] +
                ppa_score * self.weights['ppa']
            )

            return {
                'reward': total_reward,
                'details': {
                    'syntax_score': syntax_score,
                    'synthesis_score': synthesis_score,
                    'ppa_score': ppa_score,
                    'syntax_result': syntax_result,
                    'synthesis_result': synthesis_result,
                    'ppa_result': ppa_result
                },
                'success': syntax_score > 0
            }

        except Exception as e:
            self.logger.error(f"奖励计算失败: {e}")
            return {
                'reward': 0.0,
                'details': {'error': str(e)},
                'success': False
            }

    def _extract_verilog_code(self, text: str) -> str:
        """从文本中提取Verilog代码"""
        if not text:
            return ""

        # 寻找代码块
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

        # 寻找module关键字
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

    def _verify_syntax(self, verilog_code: str) -> Dict[str, Any]:
        """使用Verilator进行语法验证"""
        if not self.tools.get('verilator'):
            # 简单的语法检查
            has_module = "module" in verilog_code
            has_endmodule = "endmodule" in verilog_code
            return {
                'success': has_module and has_endmodule,
                'method': 'simple_check',
                'errors': [] if (has_module and has_endmodule) else ['Missing module/endmodule']
            }

        temp_file = self.work_dir / "temp_syntax.v"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(verilog_code)

            result = subprocess.run([
                'verilator', '--lint-only', '-Wall', str(temp_file)
            ], capture_output=True, text=True, timeout=30)

            return {
                'success': result.returncode == 0,
                'method': 'verilator',
                'stdout': result.stdout,
                'stderr': result.stderr,
                'errors': self._extract_errors(result.stderr)
            }

        except Exception as e:
            return {
                'success': False,
                'method': 'verilator',
                'error': str(e),
                'errors': [str(e)]
            }
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def _verify_synthesis(self, verilog_code: str) -> Dict[str, Any]:
        """使用Yosys进行综合验证"""
        if not self.tools.get('yosys'):
            return {'success': False, 'error': 'Yosys不可用'}

        temp_file = self.work_dir / "temp_synth.v"
        script_file = self.work_dir / "synth_script.ys"

        try:
            # 写入Verilog文件
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(verilog_code)

            # 提取模块名
            module_name = self._extract_module_name(verilog_code)
            if not module_name:
                module_name = "top"

            # 创建Yosys脚本
            yosys_script = f"""
read_verilog {temp_file}
hierarchy -check -top {module_name}
proc
opt
stat -json
"""
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(yosys_script)

            # 运行Yosys
            result = subprocess.run([
                'yosys', '-s', str(script_file)
            ], capture_output=True, text=True, timeout=60)

            success = result.returncode == 0
            stats = self._parse_yosys_stats(result.stdout) if success else {}

            return {
                'success': success,
                'method': 'yosys',
                'stats': stats,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except Exception as e:
            return {
                'success': False,
                'method': 'yosys',
                'error': str(e)
            }
        finally:
            for f in [temp_file, script_file]:
                if f.exists():
                    f.unlink()

    def _compare_ppa(self, original_code: str, optimized_code: str) -> Dict[str, Any]:
        """比较两个版本的PPA特性"""
        try:
            orig_result = self._verify_synthesis(original_code)
            opt_result = self._verify_synthesis(optimized_code)

            if not (orig_result.get('success') and opt_result.get('success')):
                return {'success': False, 'error': '无法综合代码进行PPA比较'}

            orig_stats = orig_result.get('stats', {})
            opt_stats = opt_result.get('stats', {})

            # 计算改善
            improvements = {}
            improvement_score = 0.0

            # 面积改善（基于cells数量）
            if orig_stats.get('cells', 0) > 0:
                area_improvement = (orig_stats['cells'] - opt_stats.get('cells', 0)) / orig_stats['cells']
                improvements['area'] = area_improvement
                improvement_score += max(0, area_improvement) * 0.5

            # 连线改善（基于wires数量）
            if orig_stats.get('wires', 0) > 0:
                wire_improvement = (orig_stats['wires'] - opt_stats.get('wires', 0)) / orig_stats['wires']
                improvements['wires'] = wire_improvement
                improvement_score += max(0, wire_improvement) * 0.3

            # 估算时序改善
            timing_improvement = self._estimate_timing_improvement(opt_stats, orig_stats)
            improvements['timing'] = timing_improvement
            improvement_score += max(0, timing_improvement) * 0.2

            return {
                'success': True,
                'improvements': improvements,
                'improvement_score': min(1.0, improvement_score),
                'original_stats': orig_stats,
                'optimized_stats': opt_stats
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _extract_module_name(self, verilog_code: str) -> Optional[str]:
        """提取模块名"""
        match = re.search(r'module\s+(\w+)', verilog_code)
        return match.group(1) if match else None

    def _extract_errors(self, stderr: str) -> List[str]:
        """从错误输出中提取错误信息"""
        errors = []
        for line in stderr.split('\n'):
            if 'error' in line.lower() or 'err:' in line.lower():
                errors.append(line.strip())
        return errors

    def _parse_yosys_stats(self, stdout: str) -> Dict[str, Any]:
        """解析Yosys统计信息"""
        stats = {}

        # 解析基本统计
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

    def _estimate_timing_improvement(self, opt_stats: Dict, orig_stats: Dict) -> float:
        """估算时序改善"""
        # 简化的时序估算，基于逻辑复杂度
        opt_cells = opt_stats.get('cells', 0)
        orig_cells = orig_stats.get('cells', 0)

        if orig_cells > 0:
            return max(-0.3, min(0.3, (orig_cells - opt_cells) / orig_cells * 0.2))
        return 0.0

    def __del__(self):
        """清理临时目录"""
        try:
            import shutil
            if hasattr(self, 'work_dir') and self.work_dir.exists():
                shutil.rmtree(self.work_dir)
        except Exception:
            pass


def rtl_reward_function(query: str, response: str, **kwargs) -> float:
    """
    RTL优化奖励函数 - 符合ReMA框架接口
    用于main_ppo.py中的custom_reward_function配置
    """

    # 创建全局奖励管理器实例（可以考虑优化为单例）
    if not hasattr(rtl_reward_function, '_manager'):
        rtl_reward_function._manager = RTLRewardManager()

    reward_result = rtl_reward_function._manager.calculate_reward(query, response, **kwargs)

    # ReMA期望返回float类型的奖励值
    return reward_result['reward']


def rtl_detailed_reward_function(query: str, response: str, **kwargs) -> Dict[str, Any]:
    """
    RTL优化详细奖励函数 - 返回详细信息
    """

    if not hasattr(rtl_detailed_reward_function, '_manager'):
        rtl_detailed_reward_function._manager = RTLRewardManager()

    return rtl_detailed_reward_function._manager.calculate_reward(query, response, **kwargs)
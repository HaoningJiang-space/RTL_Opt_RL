"""
VerilogVerifier - Verilog代码验证器
集成多种专业Verilog验证工具，为强化学习提供可靠的奖励信号
"""

import subprocess
import tempfile
import os
import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


class VerilogVerifier:
    """
    集成多种Verilog验证工具的验证器
    支持Verilator、Yosys、Icarus Verilog等工具
    """

    def __init__(self, work_dir: Optional[str] = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self.work_dir.mkdir(exist_ok=True)

        # 设置日志
        self.logger = logging.getLogger(self.__class__.__name__)

        # 验证工具路径
        self.tool_paths = self._detect_tools()

        # 参考PPA数据，用于计算改善奖励
        self.reference_ppa = None

    def _detect_tools(self) -> Dict[str, Optional[str]]:
        """检测可用的验证工具"""
        tools = {
            'verilator': None,
            'yosys': None,
            'iverilog': None
        }

        for tool in tools.keys():
            try:
                result = subprocess.run(
                    [tool, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    tools[tool] = tool  # 使用系统PATH中的工具
                    self.logger.info(f"检测到{tool}: 可用")
                else:
                    self.logger.warning(f"{tool}不可用")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.logger.warning(f"未找到{tool}")

        return tools

    def comprehensive_verify(
        self,
        verilog_code: str,
        reference_code: Optional[str] = None,
        module_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        综合验证Verilog代码，返回奖励信号

        Args:
            verilog_code: 待验证的Verilog代码
            reference_code: 参考代码（用于PPA对比）
            module_name: 模块名称

        Returns:
            包含验证结果和奖励分数的字典
        """

        results = {
            "syntax_check": {},
            "synthesis_check": {},
            "ppa_analysis": {},
            "verification_reward": 0.0,
            "detailed_scores": {},
            "error_messages": [],
            "warnings": []
        }

        try:
            # 1. 语法检查 (Verilator)
            if self.tool_paths['verilator']:
                syntax_result = self.verilator_check(verilog_code, module_name)
                results["syntax_check"] = syntax_result
                results["detailed_scores"]["syntax"] = 1.0 if syntax_result.get("success", False) else 0.0

            # 2. 综合检查和PPA估算 (Yosys)
            if self.tool_paths['yosys']:
                synth_result = self.yosys_check(verilog_code, module_name)
                results["synthesis_check"] = synth_result
                results["detailed_scores"]["synthesis"] = 1.0 if synth_result.get("success", False) else 0.0

                # PPA分析
                if synth_result.get("success", False) and reference_code:
                    ppa_comparison = self.compare_ppa(verilog_code, reference_code, module_name)
                    results["ppa_analysis"] = ppa_comparison
                    results["detailed_scores"]["ppa"] = ppa_comparison.get("improvement_score", 0.0)

            # 3. 编译检查 (Icarus Verilog)
            if self.tool_paths['iverilog']:
                compile_result = self.iverilog_check(verilog_code, module_name)
                results["detailed_scores"]["compilation"] = 1.0 if compile_result.get("success", False) else 0.0

            # 4. 计算综合奖励
            results["verification_reward"] = self.calculate_verification_reward(results)

        except Exception as e:
            self.logger.error(f"验证过程发生错误: {e}")
            results["error_messages"].append(str(e))
            results["verification_reward"] = 0.0

        return results

    def verilator_check(self, verilog_code: str, module_name: Optional[str] = None) -> Dict[str, Any]:
        """使用Verilator进行语法检查"""

        if not self.tool_paths['verilator']:
            return {"success": False, "error": "Verilator不可用"}

        # 写入临时文件
        temp_file = self.work_dir / "temp_verilator.v"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(verilog_code)

            # 运行Verilator语法检查
            cmd = [
                self.tool_paths['verilator'],
                '--lint-only',
                '-Wall',
                '--error-limit', '50',
                str(temp_file)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.work_dir)
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "warnings": self._extract_warnings(result.stderr),
                "errors": self._extract_errors(result.stderr)
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Verilator检查超时"}
        except Exception as e:
            return {"success": False, "error": f"Verilator检查失败: {e}"}
        finally:
            # 清理临时文件
            if temp_file.exists():
                temp_file.unlink()

    def yosys_check(self, verilog_code: str, module_name: Optional[str] = None) -> Dict[str, Any]:
        """使用Yosys进行综合检查和PPA估算"""

        if not self.tool_paths['yosys']:
            return {"success": False, "error": "Yosys不可用"}

        temp_file = self.work_dir / "temp_yosys.v"
        script_file = self.work_dir / "yosys_script.ys"

        try:
            # 写入Verilog文件
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(verilog_code)

            # 自动检测模块名
            if not module_name:
                module_name = self._extract_module_name(verilog_code)

            # 创建Yosys脚本
            yosys_script = f"""
read_verilog {temp_file}
hierarchy -check -top {module_name or "top"}
proc
opt
techmap
opt
stat -json
"""

            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(yosys_script)

            # 运行Yosys
            cmd = [self.tool_paths['yosys'], '-s', str(script_file)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.work_dir)
            )

            if result.returncode == 0:
                # 解析综合结果
                stats = self._parse_yosys_stats(result.stdout)
                return {
                    "success": True,
                    "stats": stats,
                    "area_estimate": stats.get("cells", 0),
                    "wire_count": stats.get("wires", 0),
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {
                    "success": False,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "error": "Yosys综合失败"
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Yosys检查超时"}
        except Exception as e:
            return {"success": False, "error": f"Yosys检查失败: {e}"}
        finally:
            # 清理临时文件
            for f in [temp_file, script_file]:
                if f.exists():
                    f.unlink()

    def iverilog_check(self, verilog_code: str, module_name: Optional[str] = None) -> Dict[str, Any]:
        """使用Icarus Verilog进行编译检查"""

        if not self.tool_paths['iverilog']:
            return {"success": False, "error": "Icarus Verilog不可用"}

        temp_file = self.work_dir / "temp_iverilog.v"
        out_file = self.work_dir / "temp_iverilog.vvp"

        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(verilog_code)

            # 编译
            cmd = [
                self.tool_paths['iverilog'],
                '-t', 'null',
                '-Wall',
                '-o', str(out_file),
                str(temp_file)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.work_dir)
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "warnings": self._extract_warnings(result.stderr),
                "errors": self._extract_errors(result.stderr)
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Icarus Verilog检查超时"}
        except Exception as e:
            return {"success": False, "error": f"Icarus Verilog检查失败: {e}"}
        finally:
            # 清理临时文件
            for f in [temp_file, out_file]:
                if f.exists():
                    f.unlink()

    def compare_ppa(
        self,
        optimized_code: str,
        reference_code: str,
        module_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """比较两个代码版本的PPA特性"""

        # 分别综合两个版本
        opt_result = self.yosys_check(optimized_code, module_name)
        ref_result = self.yosys_check(reference_code, module_name)

        if not (opt_result.get("success") and ref_result.get("success")):
            return {
                "success": False,
                "error": "无法综合代码进行PPA比较",
                "improvement_score": 0.0
            }

        # 提取PPA数据
        opt_stats = opt_result.get("stats", {})
        ref_stats = ref_result.get("stats", {})

        # 计算改善比例
        improvements = {}
        improvement_score = 0.0

        if ref_stats.get("cells", 0) > 0:
            area_improvement = (ref_stats["cells"] - opt_stats.get("cells", 0)) / ref_stats["cells"]
            improvements["area"] = area_improvement
            improvement_score += max(0, area_improvement) * 0.4  # 40%权重

        if ref_stats.get("wires", 0) > 0:
            wire_improvement = (ref_stats["wires"] - opt_stats.get("wires", 0)) / ref_stats["wires"]
            improvements["wires"] = wire_improvement
            improvement_score += max(0, wire_improvement) * 0.3  # 30%权重

        # 估算时序改善（基于逻辑深度）
        timing_improvement = self._estimate_timing_improvement(opt_stats, ref_stats)
        improvements["timing"] = timing_improvement
        improvement_score += max(0, timing_improvement) * 0.3  # 30%权重

        return {
            "success": True,
            "improvements": improvements,
            "improvement_score": min(1.0, improvement_score),  # 限制在1.0以内
            "optimized_stats": opt_stats,
            "reference_stats": ref_stats
        }

    def calculate_verification_reward(self, results: Dict[str, Any]) -> float:
        """计算综合验证奖励"""

        detailed_scores = results.get("detailed_scores", {})

        # 基础验证分数 (60%权重)
        syntax_score = detailed_scores.get("syntax", 0.0)
        synthesis_score = detailed_scores.get("synthesis", 0.0)
        compilation_score = detailed_scores.get("compilation", 0.0)

        base_score = (syntax_score * 0.4 + synthesis_score * 0.4 + compilation_score * 0.2)

        # PPA改善奖励 (40%权重)
        ppa_score = detailed_scores.get("ppa", 0.0)

        # 综合奖励计算
        total_reward = base_score * 0.6 + ppa_score * 0.4

        # 如果语法或综合失败，大幅降低奖励
        if syntax_score == 0.0 or synthesis_score == 0.0:
            total_reward *= 0.1

        return min(1.0, max(0.0, total_reward))

    def set_reference_ppa(self, reference_code: str, module_name: Optional[str] = None):
        """设置参考PPA数据"""
        ref_result = self.yosys_check(reference_code, module_name)
        if ref_result.get("success"):
            self.reference_ppa = ref_result.get("stats", {})
            self.logger.info("参考PPA数据已设置")
        else:
            self.logger.warning("无法设置参考PPA数据")

    def _extract_module_name(self, verilog_code: str) -> Optional[str]:
        """从Verilog代码中提取模块名"""
        match = re.search(r'module\s+(\w+)', verilog_code)
        return match.group(1) if match else None

    def _extract_warnings(self, stderr: str) -> List[str]:
        """从错误输出中提取警告"""
        warnings = []
        for line in stderr.split('\n'):
            if 'warning' in line.lower() or 'warn:' in line.lower():
                warnings.append(line.strip())
        return warnings

    def _extract_errors(self, stderr: str) -> List[str]:
        """从错误输出中提取错误"""
        errors = []
        for line in stderr.split('\n'):
            if 'error' in line.lower() or 'err:' in line.lower():
                errors.append(line.strip())
        return errors

    def _parse_yosys_stats(self, stdout: str) -> Dict[str, Any]:
        """解析Yosys统计信息"""
        stats = {}

        # 查找统计信息
        lines = stdout.split('\n')
        for i, line in enumerate(lines):
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
            elif 'Estimated number of LCs:' in line:
                try:
                    stats['logic_cells'] = int(re.search(r'(\d+)', line).group(1))
                except (AttributeError, ValueError):
                    pass

        # 尝试解析JSON统计
        json_match = re.search(r'\{[^{}]*"modules"[^{}]*\}', stdout, re.DOTALL)
        if json_match:
            try:
                json_stats = json.loads(json_match.group(0))
                stats.update(self._extract_json_stats(json_stats))
            except json.JSONDecodeError:
                pass

        return stats

    def _extract_json_stats(self, json_stats: Dict) -> Dict[str, Any]:
        """从JSON统计中提取关键信息"""
        extracted = {}

        modules = json_stats.get('modules', {})
        for module_name, module_data in modules.items():
            if 'num_cells' in module_data:
                extracted['cells'] = module_data['num_cells']
            if 'num_wires' in module_data:
                extracted['wires'] = module_data['num_wires']

        return extracted

    def _estimate_timing_improvement(self, opt_stats: Dict, ref_stats: Dict) -> float:
        """估算时序改善（简化版）"""
        # 基于逻辑单元数量的简化估算
        opt_cells = opt_stats.get('cells', opt_stats.get('logic_cells', 0))
        ref_cells = ref_stats.get('cells', ref_stats.get('logic_cells', 0))

        if ref_cells > 0 and opt_cells > 0:
            # 假设逻辑单元减少意味着逻辑深度可能减少
            return max(-0.5, min(0.5, (ref_cells - opt_cells) / ref_cells * 0.3))

        return 0.0

    def get_tool_status(self) -> Dict[str, bool]:
        """获取工具可用状态"""
        return {tool: path is not None for tool, path in self.tool_paths.items()}

    def cleanup(self):
        """清理工作目录"""
        try:
            import shutil
            if self.work_dir.exists():
                shutil.rmtree(self.work_dir)
        except Exception as e:
            self.logger.error(f"清理工作目录失败: {e}")

    def __del__(self):
        """析构函数"""
        self.cleanup()
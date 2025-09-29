#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABC工具接口实现
ABC tool interface implementation

该模块提供与ABC逻辑综合工具的接口，支持Verilog和AIG格式的相互转换，
以及各种逻辑优化操作，确保逻辑等效性。
This module provides interface with ABC logic synthesis tool, supporting
conversion between Verilog and AIG formats, and various logic optimization
operations with guaranteed logic equivalence.
"""

import os
import re
import subprocess
import tempfile
import shutil
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import logging

from ..utils.config import ABCConfig


class ABCInterface:
    """ABC工具接口类 / ABC tool interface class

    提供与ABC工具的完整接口，包括格式转换、逻辑优化、等效性验证等功能。
    Provides complete interface with ABC tool including format conversion,
    logic optimization, equivalence verification, etc.
    """

    def __init__(self, config: Optional[ABCConfig] = None):
        """初始化ABC接口 / Initialize ABC interface

        Args:
            config: ABC配置对象，如果为None则使用默认配置
                   ABC configuration object, use default if None
        """
        self.config = config or ABCConfig()
        self.logger = logging.getLogger(__name__)

        # 验证ABC是否可用 / Verify ABC availability
        self._verify_abc_installation()

        # 创建临时目录 / Create temporary directory
        os.makedirs(self.config.temp_dir, exist_ok=True)

    def _verify_abc_installation(self) -> None:
        """验证ABC工具是否正确安装 / Verify ABC tool installation"""
        try:
            subprocess.run(
                [self.config.abc_binary_path, "-h"],
                capture_output=True,
                timeout=10,
                check=True
            )
            self.logger.info("ABC工具验证成功 / ABC tool verification successful")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.error(f"ABC工具验证失败 / ABC tool verification failed: {e}")
            raise RuntimeError(f"ABC tool not available: {e}")

    def verilog_to_aig(self, verilog_file: str, output_aig: Optional[str] = None) -> str:
        """将Verilog文件转换为AIG格式 / Convert Verilog file to AIG format

        Args:
            verilog_file: 输入的Verilog文件路径 / Input Verilog file path
            output_aig: 输出的AIG文件路径，如果为None则自动生成
                       Output AIG file path, auto-generate if None

        Returns:
            str: 输出AIG文件的路径 / Path to output AIG file

        Raises:
            RuntimeError: 转换失败时抛出 / Raised when conversion fails
        """
        if not os.path.exists(verilog_file):
            raise FileNotFoundError(f"Verilog文件不存在 / Verilog file not found: {verilog_file}")

        if output_aig is None:
            output_aig = os.path.join(
                self.config.temp_dir,
                f"{Path(verilog_file).stem}.aig"
            )

        # 构建ABC脚本 / Build ABC script
        abc_script = f"""
        read_verilog {verilog_file}
        strash
        write_aig {output_aig}
        """

        try:
            self._run_abc_script(abc_script, "Verilog to AIG conversion")

            if not os.path.exists(output_aig):
                raise RuntimeError("AIG文件生成失败 / AIG file generation failed")

            self.logger.info(f"Verilog转AIG成功 / Verilog to AIG successful: {output_aig}")
            return output_aig

        except Exception as e:
            self.logger.error(f"Verilog转AIG失败 / Verilog to AIG failed: {e}")
            raise RuntimeError(f"Verilog to AIG conversion failed: {e}")

    def aig_to_verilog(self, aig_file: str, output_verilog: Optional[str] = None) -> str:
        """将AIG文件转换为Verilog格式 / Convert AIG file to Verilog format

        Args:
            aig_file: 输入的AIG文件路径 / Input AIG file path
            output_verilog: 输出的Verilog文件路径，如果为None则自动生成
                           Output Verilog file path, auto-generate if None

        Returns:
            str: 输出Verilog文件的路径 / Path to output Verilog file

        Raises:
            RuntimeError: 转换失败时抛出 / Raised when conversion fails
        """
        if not os.path.exists(aig_file):
            raise FileNotFoundError(f"AIG文件不存在 / AIG file not found: {aig_file}")

        if output_verilog is None:
            output_verilog = os.path.join(
                self.config.temp_dir,
                f"{Path(aig_file).stem}.v"
            )

        # 构建ABC脚本 / Build ABC script
        abc_script = f"""
        read_aig {aig_file}
        write_verilog {output_verilog}
        """

        try:
            self._run_abc_script(abc_script, "AIG to Verilog conversion")

            if not os.path.exists(output_verilog):
                raise RuntimeError("Verilog文件生成失败 / Verilog file generation failed")

            self.logger.info(f"AIG转Verilog成功 / AIG to Verilog successful: {output_verilog}")
            return output_verilog

        except Exception as e:
            self.logger.error(f"AIG转Verilog失败 / AIG to Verilog failed: {e}")
            raise RuntimeError(f"AIG to Verilog conversion failed: {e}")

    def apply_optimization(self, input_file: str, optimization: str,
                          input_format: str = "auto") -> Dict[str, Any]:
        """应用ABC优化操作 / Apply ABC optimization operation

        Args:
            input_file: 输入文件路径 / Input file path
            optimization: 优化操作名称 / Optimization operation name
            input_format: 输入格式 ("verilog", "aig", "auto") / Input format

        Returns:
            Dict[str, Any]: 优化结果字典，包含成功状态、输出文件、统计信息等
                           Optimization result dictionary with success status,
                           output files, statistics, etc.
        """
        try:
            # 自动检测输入格式 / Auto-detect input format
            if input_format == "auto":
                input_format = self._detect_file_format(input_file)

            # 转换为AIG格式进行优化 / Convert to AIG format for optimization
            if input_format == "verilog":
                aig_file = self.verilog_to_aig(input_file)
            else:
                aig_file = input_file

            # 执行优化 / Execute optimization
            optimized_aig, stats = self._apply_aig_optimization(aig_file, optimization)

            # 转换回原始格式 / Convert back to original format
            if input_format == "verilog":
                output_file = self.aig_to_verilog(optimized_aig)
            else:
                output_file = optimized_aig

            # 验证逻辑等效性 / Verify logic equivalence
            is_equivalent = self.verify_equivalence(
                aig_file if input_format == "aig" else self.verilog_to_aig(input_file),
                optimized_aig
            )

            return {
                "success": True,
                "optimized_file": output_file,
                "optimization": optimization,
                "input_format": input_format,
                "statistics": stats,
                "is_equivalent": is_equivalent,
                "original_aig": aig_file,
                "optimized_aig": optimized_aig
            }

        except Exception as e:
            self.logger.error(f"优化操作失败 / Optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "optimization": optimization,
                "input_file": input_file
            }

    def _apply_aig_optimization(self, aig_file: str, optimization: str) -> Tuple[str, Dict[str, Any]]:
        """在AIG文件上应用优化操作 / Apply optimization on AIG file

        Args:
            aig_file: AIG文件路径 / AIG file path
            optimization: 优化操作名称 / Optimization operation name

        Returns:
            Tuple[str, Dict]: 优化后的AIG文件路径和统计信息
                             Optimized AIG file path and statistics
        """
        # 生成输出文件名 / Generate output filename
        output_aig = os.path.join(
            self.config.temp_dir,
            f"{Path(aig_file).stem}_{optimization}_opt.aig"
        )

        # 获取原始统计信息 / Get original statistics
        original_stats = self.get_aig_statistics(aig_file)

        # 构建优化脚本 / Build optimization script
        abc_script = self._build_optimization_script(aig_file, output_aig, optimization)

        # 执行优化 / Execute optimization
        self._run_abc_script(abc_script, f"Optimization: {optimization}")

        # 获取优化后统计信息 / Get optimized statistics
        optimized_stats = self.get_aig_statistics(output_aig)

        # 计算改善指标 / Calculate improvement metrics
        improvement_stats = self._calculate_improvement(original_stats, optimized_stats)

        return output_aig, {
            "original": original_stats,
            "optimized": optimized_stats,
            "improvement": improvement_stats
        }

    def _build_optimization_script(self, input_aig: str, output_aig: str, optimization: str) -> str:
        """构建优化脚本 / Build optimization script

        Args:
            input_aig: 输入AIG文件 / Input AIG file
            output_aig: 输出AIG文件 / Output AIG file
            optimization: 优化操作 / Optimization operation

        Returns:
            str: ABC脚本内容 / ABC script content
        """
        if optimization in self.config.abc_commands:
            # 单个命令 / Single command
            command = self.config.abc_commands[optimization]
            script = f"""
            read_aig {input_aig}
            print_stats
            {command}
            print_stats
            write_aig {output_aig}
            """
        elif optimization in self.config.combo_sequences:
            # 组合优化序列 / Combination optimization sequence
            commands = []
            for cmd in self.config.combo_sequences[optimization]:
                commands.append(self.config.abc_commands[cmd])

            script = f"""
            read_aig {input_aig}
            print_stats
            {'; '.join(commands)}
            print_stats
            write_aig {output_aig}
            """
        else:
            # 默认：无操作 / Default: no operation
            script = f"""
            read_aig {input_aig}
            write_aig {output_aig}
            """

        return script

    def verify_equivalence(self, aig_file1: str, aig_file2: str) -> bool:
        """验证两个AIG文件的逻辑等效性 / Verify logic equivalence of two AIG files

        Args:
            aig_file1: 第一个AIG文件 / First AIG file
            aig_file2: 第二个AIG文件 / Second AIG file

        Returns:
            bool: 是否逻辑等效 / Whether logically equivalent
        """
        try:
            # 构建等效性检查脚本 / Build equivalence check script
            abc_script = f"""
            cec {aig_file1} {aig_file2}
            """

            result = self._run_abc_script(
                abc_script,
                "Equivalence verification",
                timeout=self.config.equivalence_timeout
            )

            # 解析结果 / Parse result
            # ABC的cec命令会输出"Networks are equivalent"或类似信息
            # ABC's cec command outputs "Networks are equivalent" or similar
            return ("equivalent" in result.stdout.lower() and
                    "not equivalent" not in result.stdout.lower())

        except Exception as e:
            self.logger.warning(f"等效性验证失败 / Equivalence verification failed: {e}")
            return False

    def get_aig_statistics(self, aig_file: str) -> Dict[str, Any]:
        """获取AIG文件的统计信息 / Get statistics of AIG file

        Args:
            aig_file: AIG文件路径 / AIG file path

        Returns:
            Dict[str, Any]: 统计信息字典 / Statistics dictionary
        """
        try:
            abc_script = f"""
            read_aig {aig_file}
            print_stats
            """

            result = self._run_abc_script(abc_script, "Get AIG statistics")
            stats = self._parse_abc_statistics(result.stdout)

            self.logger.debug(f"AIG统计信息 / AIG statistics: {stats}")
            return stats

        except Exception as e:
            self.logger.error(f"获取统计信息失败 / Failed to get statistics: {e}")
            return {"error": str(e)}

    def _parse_abc_statistics(self, abc_output: str) -> Dict[str, Any]:
        """解析ABC统计输出 / Parse ABC statistics output

        Args:
            abc_output: ABC输出文本 / ABC output text

        Returns:
            Dict[str, Any]: 解析后的统计信息 / Parsed statistics
        """
        stats = {
            "nodes": 0,
            "depth": 0,
            "inputs": 0,
            "outputs": 0,
            "latches": 0
        }

        try:
            # 使用正则表达式提取统计信息 / Extract statistics using regex
            patterns = {
                "nodes": r"(?:Nodes|AIG nodes|And):\s*(\d+)",
                "depth": r"(?:Depth|Levels):\s*(\d+)",
                "inputs": r"(?:Inputs|PI):\s*(\d+)",
                "outputs": r"(?:Outputs|PO):\s*(\d+)",
                "latches": r"(?:Latches|FF):\s*(\d+)"
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, abc_output, re.IGNORECASE)
                if match:
                    stats[key] = int(match.group(1))

        except Exception as e:
            self.logger.warning(f"统计信息解析部分失败 / Partial failure in statistics parsing: {e}")

        return stats

    def _calculate_improvement(self, original_stats: Dict[str, Any],
                             optimized_stats: Dict[str, Any]) -> Dict[str, float]:
        """计算优化改善指标 / Calculate optimization improvement metrics

        Args:
            original_stats: 原始统计信息 / Original statistics
            optimized_stats: 优化后统计信息 / Optimized statistics

        Returns:
            Dict[str, float]: 改善指标 / Improvement metrics
        """
        improvement = {}

        for key in ["nodes", "depth"]:
            if key in original_stats and key in optimized_stats:
                original_val = original_stats[key]
                optimized_val = optimized_stats[key]

                if original_val > 0:
                    improvement[f"{key}_reduction"] = (original_val - optimized_val) / original_val
                    improvement[f"{key}_absolute"] = original_val - optimized_val
                else:
                    improvement[f"{key}_reduction"] = 0.0
                    improvement[f"{key}_absolute"] = 0

        return improvement

    def _detect_file_format(self, file_path: str) -> str:
        """自动检测文件格式 / Auto-detect file format

        Args:
            file_path: 文件路径 / File path

        Returns:
            str: 文件格式 ("verilog" 或 "aig") / File format ("verilog" or "aig")
        """
        ext = Path(file_path).suffix.lower()

        if ext in ['.v', '.verilog']:
            return "verilog"
        elif ext in ['.aig']:
            return "aig"
        else:
            # 通过文件内容判断 / Judge by file content
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('module') or 'module' in first_line:
                        return "verilog"
                    elif first_line.startswith('aig'):
                        return "aig"
            except:
                pass

            # 默认假设为Verilog / Default assume Verilog
            return "verilog"

    def _run_abc_script(self, script: str, description: str = "",
                       timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        """运行ABC脚本 / Run ABC script

        Args:
            script: ABC脚本内容 / ABC script content
            description: 操作描述 / Operation description
            timeout: 超时时间 / Timeout

        Returns:
            subprocess.CompletedProcess: 执行结果 / Execution result

        Raises:
            RuntimeError: 执行失败时抛出 / Raised when execution fails
        """
        if timeout is None:
            timeout = self.config.optimization_timeout

        with tempfile.NamedTemporaryFile(mode='w', suffix='.abc', delete=False) as f:
            f.write(script)
            script_file = f.name

        try:
            self.logger.debug(f"执行ABC脚本 / Executing ABC script: {description}")

            result = subprocess.run(
                [self.config.abc_binary_path, "-f", script_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )

            self.logger.debug(f"ABC执行成功 / ABC execution successful: {description}")
            return result

        except subprocess.TimeoutExpired as e:
            self.logger.error(f"ABC执行超时 / ABC execution timeout: {description}")
            raise RuntimeError(f"ABC execution timeout for {description}: {e}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"ABC执行失败 / ABC execution failed: {description}\n"
                            f"stdout: {e.stdout}\nstderr: {e.stderr}")
            raise RuntimeError(f"ABC execution failed for {description}: {e}")

        finally:
            # 清理临时文件 / Clean up temporary file
            try:
                os.unlink(script_file)
            except:
                pass

    def cleanup(self):
        """清理临时文件 / Cleanup temporary files"""
        try:
            if os.path.exists(self.config.temp_dir):
                shutil.rmtree(self.config.temp_dir)
                self.logger.info("临时文件清理完成 / Temporary files cleaned up")
        except Exception as e:
            self.logger.warning(f"临时文件清理失败 / Temporary file cleanup failed: {e}")

    def __enter__(self):
        """上下文管理器入口 / Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出 / Context manager exit"""
        self.cleanup()
        # 参数用于上下文管理器协议，但在此实现中不需要处理
        # Parameters are for context manager protocol, not needed in this implementation


# 测试代码 / Test code
if __name__ == "__main__":
    # 设置日志 / Setup logging
    logging.basicConfig(level=logging.INFO)

    # 测试ABC接口 / Test ABC interface
    with ABCInterface() as abc:
        print("ABC接口初始化成功 / ABC interface initialized successfully")

        # 测试统计信息获取 / Test statistics retrieval
        # 这里需要一个实际的AIG文件来测试 / Need an actual AIG file for testing
        # test_aig = "test.aig"
        # if os.path.exists(test_aig):
        #     stats = abc.get_aig_statistics(test_aig)
        #     print(f"统计信息 / Statistics: {stats}")
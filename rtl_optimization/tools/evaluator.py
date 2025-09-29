#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合评估系统实现
Hybrid evaluation system implementation

该模块实现混合评估策略，结合快速GNN预测和精确综合评估，
为强化学习提供高效且准确的反馈机制。
This module implements hybrid evaluation strategy, combining fast GNN prediction
and accurate synthesis evaluation, providing efficient and accurate feedback
mechanism for reinforcement learning.
"""

import os
import time
import subprocess
import tempfile
from typing import Dict, Any, Optional
import logging
import numpy as np

import torch
import torch.nn as nn

from ..utils.config import EvaluationConfig
from ..tools.abc_interface import ABCInterface
from ..graph.gnn_model import RTLOptimizationGNN
from ..graph.rtl_graph import RTLNetlistGraph


class HybridEvaluationSystem:
    """混合评估系统 / Hybrid evaluation system

    结合GNN快速预测和真实综合评估的混合策略，解决训练效率和准确性的平衡问题。
    Hybrid strategy combining GNN fast prediction and real synthesis evaluation,
    solving the balance between training efficiency and accuracy.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None,
                 gnn_model: Optional[RTLOptimizationGNN] = None):
        """初始化混合评估系统 / Initialize hybrid evaluation system

        Args:
            config: 评估配置 / Evaluation configuration
            gnn_model: 预训练的GNN模型 / Pretrained GNN model
        """
        self.config = config or EvaluationConfig()
        self.logger = logging.getLogger(__name__)

        # ABC工具接口 / ABC tool interface
        self.abc_tool = ABCInterface()

        # RTL图构建器 / RTL graph builder
        self.graph_builder = RTLNetlistGraph()

        # GNN模型用于快速预测 / GNN model for fast prediction
        self.gnn_model = gnn_model
        if self.gnn_model is None:
            self.logger.warning("未提供GNN模型，将只使用真实综合评估 / "
                              "No GNN model provided, will only use real synthesis evaluation")

        # PPA预测器（独立的轻量级网络） / PPA predictor (independent lightweight network)
        self.ppa_predictor = self._build_ppa_predictor()

        # Yosys接口用于最终PPA评估 / Yosys interface for final PPA evaluation
        self.yosys_available = self._check_yosys_availability()

        # 评估缓存 / Evaluation cache
        self.evaluation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # 统计信息 / Statistics
        self.eval_stats = {
            "fast_evaluations": 0,
            "full_evaluations": 0,
            "total_time_fast": 0.0,
            "total_time_full": 0.0,
            "prediction_errors": []
        }

        self.logger.info("混合评估系统初始化完成 / Hybrid evaluation system initialized")

    def _build_ppa_predictor(self) -> nn.Module:
        """构建PPA预测网络 / Build PPA prediction network"""
        return nn.Sequential(
            nn.Linear(128, 256),  # 输入是图嵌入
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),    # 输出PPA (delay, area, power)
            nn.Sigmoid()          # 归一化到[0,1]
        )

    def _check_yosys_availability(self) -> bool:
        """检查Yosys是否可用 / Check if Yosys is available"""
        try:
            subprocess.run(["yosys", "-h"], capture_output=True, timeout=10, check=True)
            self.logger.info("Yosys工具可用 / Yosys tool available")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("Yosys工具不可用，将使用ABC统计作为PPA估算 / "
                              "Yosys tool not available, will use ABC statistics for PPA estimation")
            return False

    def evaluate(self, input_file: str, step_count: int,
                file_format: str = "auto") -> Dict[str, Any]:
        """主评估接口 / Main evaluation interface

        Args:
            input_file: 输入文件路径 / Input file path
            step_count: 当前步数 / Current step count
            file_format: 文件格式 / File format

        Returns:
            Dict[str, Any]: 评估结果 / Evaluation result
        """
        # 检查缓存 / Check cache
        cache_key = self._get_cache_key(input_file)
        if cache_key in self.evaluation_cache:
            self.cache_hits += 1
            self.logger.debug("使用缓存结果 / Using cached result")
            return self.evaluation_cache[cache_key]

        self.cache_misses += 1

        # 决定评估策略 / Decide evaluation strategy
        use_full_evaluation = self._should_use_full_evaluation(step_count, input_file)

        if use_full_evaluation:
            result = self.full_evaluate(input_file, file_format)
            self._update_predictor_calibration(input_file, result, file_format)
        else:
            result = self.quick_evaluate(input_file, file_format)

        # 更新缓存 / Update cache
        self.evaluation_cache[cache_key] = result

        return result

    def quick_evaluate(self, input_file: str, file_format: str = "auto") -> Dict[str, Any]:
        """快速评估（基于GNN预测）/ Quick evaluation (based on GNN prediction)"""
        start_time = time.time()

        try:
            # 构建图表示 / Build graph representation
            graph_data = self.graph_builder.build_from_file(input_file, file_format)

            if self.gnn_model is not None:
                # 使用GNN模型预测 / Use GNN model for prediction
                self.gnn_model.eval()
                with torch.no_grad():
                    outputs = self.gnn_model(graph_data)
                    graph_embedding = outputs["graph_embedding"]

                    # 使用独立的PPA预测器 / Use independent PPA predictor
                    predicted_ppa = self.ppa_predictor(graph_embedding)
                    confidence = outputs.get("confidence", torch.tensor([0.7]))

                result = {
                    "delay": predicted_ppa[0, 0].item(),
                    "area": predicted_ppa[0, 1].item(),
                    "power": predicted_ppa[0, 2].item(),
                    "confidence": confidence[0].item(),
                    "method": "gnn_prediction",
                    "evaluation_time": time.time() - start_time
                }
            else:
                # 降级到基于ABC统计的估算 / Fallback to ABC statistics-based estimation
                result = self._estimate_ppa_from_abc(input_file, file_format)
                result["method"] = "abc_estimation"
                result["confidence"] = 0.5  # 中等置信度
                result["evaluation_time"] = time.time() - start_time

            self.eval_stats["fast_evaluations"] += 1
            self.eval_stats["total_time_fast"] += result["evaluation_time"]

            return result

        except Exception as e:
            self.logger.error(f"快速评估失败 / Quick evaluation failed: {e}")
            # 降级到基本估算 / Fallback to basic estimation
            return {
                "delay": 0.5, "area": 0.5, "power": 0.5,
                "confidence": 0.3, "method": "fallback",
                "evaluation_time": time.time() - start_time,
                "error": str(e)
            }

    def full_evaluate(self, input_file: str, file_format: str = "auto") -> Dict[str, Any]:
        """完整评估（真实综合）/ Full evaluation (real synthesis)"""
        start_time = time.time()

        try:
            if self.yosys_available:
                result = self._yosys_synthesis_evaluation(input_file, file_format)
            else:
                result = self._abc_synthesis_evaluation(input_file, file_format)

            result["method"] = "synthesis"
            result["confidence"] = 1.0
            result["evaluation_time"] = time.time() - start_time

            self.eval_stats["full_evaluations"] += 1
            self.eval_stats["total_time_full"] += result["evaluation_time"]

            return result

        except Exception as e:
            self.logger.error(f"完整评估失败 / Full evaluation failed: {e}")
            # 降级到快速评估 / Fallback to quick evaluation
            return self.quick_evaluate(input_file, file_format)

    def _yosys_synthesis_evaluation(self, input_file: str, file_format: str) -> Dict[str, Any]:
        """使用Yosys进行综合评估 / Synthesis evaluation using Yosys"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 转换为Verilog格式（如果需要）/ Convert to Verilog format (if needed)
            if file_format == "aig":
                verilog_file = self.abc_tool.aig_to_verilog(input_file)
            else:
                verilog_file = input_file

            # 构建Yosys脚本 / Build Yosys script
            output_file = os.path.join(temp_dir, "output.json")
            yosys_script = f"""
            read_verilog {verilog_file}
            hierarchy -auto-top
            proc; opt; memory; opt
            techmap; opt
            stat -json {output_file}
            """

            script_file = os.path.join(temp_dir, "script.ys")
            with open(script_file, 'w') as f:
                f.write(yosys_script)

            # 运行Yosys / Run Yosys
            result = subprocess.run(
                ["yosys", "-s", script_file],
                capture_output=True,
                text=True,
                timeout=30,
                check=True
            )

            # 解析结果 / Parse results
            return self._parse_yosys_output(result.stdout, output_file)

    def _abc_synthesis_evaluation(self, input_file: str, file_format: str) -> Dict[str, Any]:
        """使用ABC进行综合评估 / Synthesis evaluation using ABC"""
        # 获取ABC统计信息 / Get ABC statistics
        if file_format == "verilog":
            aig_file = self.abc_tool.verilog_to_aig(input_file)
        else:
            aig_file = input_file

        stats = self.abc_tool.get_aig_statistics(aig_file)

        # 基于ABC统计信息估算PPA / Estimate PPA based on ABC statistics
        return self._convert_abc_stats_to_ppa(stats)

    def _estimate_ppa_from_abc(self, input_file: str, file_format: str) -> Dict[str, Any]:
        """基于ABC统计信息估算PPA / Estimate PPA based on ABC statistics"""
        try:
            if file_format == "verilog":
                aig_file = self.abc_tool.verilog_to_aig(input_file)
            else:
                aig_file = input_file

            stats = self.abc_tool.get_aig_statistics(aig_file)
            return self._convert_abc_stats_to_ppa(stats)

        except Exception as e:
            self.logger.warning(f"ABC统计获取失败 / ABC statistics retrieval failed: {e}")
            return {"delay": 0.5, "area": 0.5, "power": 0.5}

    def _convert_abc_stats_to_ppa(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """将ABC统计信息转换为PPA估算 / Convert ABC statistics to PPA estimation"""
        # 简化的PPA估算模型 / Simplified PPA estimation model
        nodes = stats.get("nodes", 100)
        depth = stats.get("depth", 10)
        inputs = stats.get("inputs", 10)
        outputs = stats.get("outputs", 10)

        # 归一化到[0,1]区间 / Normalize to [0,1] range
        # 这些是基于经验的简化估算公式 / These are experience-based simplified estimation formulas
        delay = min(depth / 50.0, 1.0)  # 深度影响延迟
        area = min(nodes / 1000.0, 1.0)  # 节点数影响面积
        power = min((nodes + inputs + outputs) / 1500.0, 1.0)  # 总复杂度影响功耗

        return {
            "delay": delay,
            "area": area,
            "power": power,
            "abc_stats": stats
        }

    def _parse_yosys_output(self, stdout: str, json_file: str) -> Dict[str, Any]:
        """解析Yosys输出 / Parse Yosys output"""
        try:
            import json

            # 从JSON文件读取统计信息 / Read statistics from JSON file
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    stats = json.load(f)

                # 提取关键指标 / Extract key metrics
                modules = stats.get("modules", {})
                if modules:
                    # 获取顶层模块统计 / Get top module statistics
                    top_module = list(modules.values())[0]
                    cells = top_module.get("num_cells", 0)
                    wires = top_module.get("num_wires", 0)

                    # 估算PPA / Estimate PPA
                    delay = min(cells / 100.0, 1.0)
                    area = min(cells / 500.0, 1.0)
                    power = min((cells + wires) / 600.0, 1.0)

                    return {
                        "delay": delay,
                        "area": area,
                        "power": power,
                        "yosys_stats": top_module
                    }

            # 降级到stdout解析 / Fallback to stdout parsing
            return self._parse_yosys_stdout(stdout)

        except Exception as e:
            self.logger.warning(f"Yosys输出解析失败 / Yosys output parsing failed: {e}")
            return {"delay": 0.5, "area": 0.5, "power": 0.5}

    def _parse_yosys_stdout(self, stdout: str) -> Dict[str, Any]:
        """解析Yosys标准输出 / Parse Yosys stdout"""
        import re

        # 提取关键统计信息 / Extract key statistics
        cells_match = re.search(r'Number of cells:\s+(\d+)', stdout)
        wires_match = re.search(r'Number of wires:\s+(\d+)', stdout)

        cells = int(cells_match.group(1)) if cells_match else 50
        wires = int(wires_match.group(1)) if wires_match else 50

        # 估算PPA / Estimate PPA
        delay = min(cells / 100.0, 1.0)
        area = min(cells / 500.0, 1.0)
        power = min((cells + wires) / 600.0, 1.0)

        return {
            "delay": delay,
            "area": area,
            "power": power,
            "cells": cells,
            "wires": wires
        }

    def _should_use_full_evaluation(self, step_count: int, input_file: str) -> bool:
        """判断是否应该使用完整评估 / Determine whether to use full evaluation"""
        # 定期完整评估 / Periodic full evaluation
        if step_count % self.config.full_eval_interval == 0:
            return True

        # 基于置信度的动态策略 / Dynamic strategy based on confidence
        if self.gnn_model is not None:
            try:
                # 快速获取置信度 / Quick confidence retrieval
                graph_data = self.graph_builder.build_from_file(input_file)
                with torch.no_grad():
                    outputs = self.gnn_model(graph_data)
                    confidence = outputs.get("confidence", torch.tensor([0.8]))

                # 置信度低时使用完整评估 / Use full evaluation when confidence is low
                if confidence[0].item() < self.config.prediction_confidence_threshold:
                    return True
            except:
                pass

        return False

    def _update_predictor_calibration(self, input_file: str, ground_truth: Dict[str, Any],
                                     file_format: str) -> None:
        """校正预测器 / Calibrate predictor"""
        if self.gnn_model is None:
            return

        try:
            # 获取图嵌入 / Get graph embedding
            graph_data = self.graph_builder.build_from_file(input_file, file_format)

            with torch.no_grad():
                outputs = self.gnn_model(graph_data)
                graph_embedding = outputs["graph_embedding"]

            # 计算预测值 / Compute prediction
            predicted_ppa = self.ppa_predictor(graph_embedding)

            # 构建目标值 / Build target values
            target_ppa = torch.tensor([
                ground_truth.get("delay", 0.5),
                ground_truth.get("area", 0.5),
                ground_truth.get("power", 0.5)
            ]).unsqueeze(0)

            # 计算误差并记录 / Compute error and record
            error = torch.mean(torch.abs(predicted_ppa - target_ppa)).item()
            self.eval_stats["prediction_errors"].append(error)

            # 在线学习更新（可选）/ Online learning update (optional)
            if hasattr(self, 'predictor_optimizer'):
                loss = nn.MSELoss()(predicted_ppa, target_ppa)
                self.predictor_optimizer.zero_grad()
                loss.backward()
                self.predictor_optimizer.step()

        except Exception as e:
            self.logger.warning(f"预测器校正失败 / Predictor calibration failed: {e}")

    def _get_cache_key(self, input_file: str) -> str:
        """生成缓存键 / Generate cache key"""
        # 基于文件内容的哈希 / Hash based on file content
        try:
            import hashlib
            with open(input_file, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except:
            return input_file

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """获取评估统计信息 / Get evaluation statistics"""
        total_evals = self.eval_stats["fast_evaluations"] + self.eval_stats["full_evaluations"]

        stats = {
            "total_evaluations": total_evals,
            "fast_evaluations": self.eval_stats["fast_evaluations"],
            "full_evaluations": self.eval_stats["full_evaluations"],
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }

        if total_evals > 0:
            stats["fast_eval_ratio"] = self.eval_stats["fast_evaluations"] / total_evals
            stats["avg_time_fast"] = self.eval_stats["total_time_fast"] / max(self.eval_stats["fast_evaluations"], 1)
            stats["avg_time_full"] = self.eval_stats["total_time_full"] / max(self.eval_stats["full_evaluations"], 1)

        if self.eval_stats["prediction_errors"]:
            stats["avg_prediction_error"] = np.mean(self.eval_stats["prediction_errors"])
            stats["prediction_error_std"] = np.std(self.eval_stats["prediction_errors"])

        return stats

    def reset_statistics(self) -> None:
        """重置统计信息 / Reset statistics"""
        self.eval_stats = {
            "fast_evaluations": 0,
            "full_evaluations": 0,
            "total_time_fast": 0.0,
            "total_time_full": 0.0,
            "prediction_errors": []
        }
        self.cache_hits = 0
        self.cache_misses = 0
        self.evaluation_cache.clear()

    def cleanup(self) -> None:
        """清理资源 / Cleanup resources"""
        self.abc_tool.cleanup()
        self.evaluation_cache.clear()


# 测试代码 / Test code
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # 测试混合评估系统 / Test hybrid evaluation system
    evaluator = HybridEvaluationSystem()

    print("混合评估系统测试 / Hybrid evaluation system test:")
    print(f"初始统计信息 / Initial statistics: {evaluator.get_evaluation_statistics()}")

    # 这里需要实际的RTL文件来测试 / Need actual RTL files for testing
    # test_file = "test.v"
    # if os.path.exists(test_file):
    #     result = evaluator.evaluate(test_file, step_count=1)
    #     print(f"评估结果 / Evaluation result: {result}")

    print("混合评估系统测试完成 / Hybrid evaluation system test completed")
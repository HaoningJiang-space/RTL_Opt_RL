# RTL Multi-Agent Optimization Framework
# 基于ReMA+VeRL的双智能体RTL优化系统

__version__ = "0.1.0"
__author__ = "RTL Optimization Team"

from .agents import BaseAgent, MetaOptimizerAgent, CodeRewriterAgent
from .training import MultiAgentRTLEnvironment, RTLMultiAgentTrainer
from .utils import VerilogVerifier, DataProcessor
from .evaluation import PPAEvaluator, OptimizationAnalyzer

__all__ = [
    "BaseAgent",
    "MetaOptimizerAgent",
    "CodeRewriterAgent",
    "MultiAgentRTLEnvironment",
    "RTLMultiAgentTrainer",
    "VerilogVerifier",
    "DataProcessor",
    "PPAEvaluator",
    "OptimizationAnalyzer"
]
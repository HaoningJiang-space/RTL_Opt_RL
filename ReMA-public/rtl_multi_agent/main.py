#!/usr/bin/env python3
"""
RTL多智能体优化系统主启动脚本
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rtl_multi_agent.agents import MetaOptimizerAgent, CodeRewriterAgent
from rtl_multi_agent.utils import VerilogVerifier, DataProcessor
from rtl_multi_agent.training import RTLMultiAgentTrainer
from rtl_multi_agent.evaluation import PPAEvaluator


def setup_logging(config: dict):
    """设置日志配置"""
    log_level = getattr(logging, config.get("logging", {}).get("level", "INFO"))
    log_file = config.get("logging", {}).get("log_file")

    # 创建日志目录
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 设置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

    return config


def initialize_agents(config: dict) -> dict:
    """初始化智能体"""
    agents = {}

    model_config = config.get("models", {})

    # 初始化MetaOptimizer
    meta_config = model_config.get("meta_optimizer", {})
    agents["meta_optimizer"] = MetaOptimizerAgent(
        model_name=meta_config.get("model_name", "deepseek-ai/deepseek-coder-6.7b-instruct"),
        max_length=meta_config.get("max_length", 4096),
        device=meta_config.get("device", "auto")
    )

    # 初始化CodeRewriter
    rewriter_config = model_config.get("code_rewriter", {})
    agents["code_rewriter"] = CodeRewriterAgent(
        model_name=rewriter_config.get("model_name", "deepseek-ai/deepseek-coder-6.7b-instruct"),
        max_length=rewriter_config.get("max_length", 4096),
        device=rewriter_config.get("device", "auto")
    )

    return agents


def load_or_generate_data(config: dict) -> list:
    """加载或生成训练数据"""
    data_config = config.get("data", {})
    data_processor = DataProcessor()

    train_data_path = data_config.get("train_data_path")

    # 如果没有指定数据路径或文件不存在，生成示例数据
    if not train_data_path or not Path(train_data_path).exists():
        if data_config.get("generate_sample_data", False):
            logging.info("生成示例数据用于训练")
            num_samples = data_config.get("num_sample_cases", 20)
            optimization_data = data_processor.generate_sample_data(num_samples)
        else:
            raise ValueError("未指定训练数据路径且未启用示例数据生成")
    else:
        logging.info(f"从文件加载训练数据: {train_data_path}")
        optimization_data = data_processor.load_optimization_data(train_data_path)

    # 数据增强
    if data_config.get("augment_data", False):
        optimization_data = data_processor.augment_data(optimization_data)

    # 训练/测试分割
    split_ratio = data_config.get("train_test_split", 0.8)
    data_split = data_processor.create_train_test_split(
        optimization_data,
        test_ratio=1 - split_ratio
    )

    return data_split["train"]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RTL多智能体优化系统")
    parser.add_argument(
        "--config",
        type=str,
        default="rtl_multi_agent/configs/default_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "demo"],
        default="train",
        help="运行模式"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="检查点路径（用于恢复训练或评估）"
    )

    args = parser.parse_args()

    try:
        # 加载配置
        print(f"加载配置文件: {args.config}")
        config = load_config(args.config)

        # 设置日志
        setup_logging(config)
        logger = logging.getLogger("main")

        logger.info("=" * 60)
        logger.info("RTL多智能体优化系统启动")
        logger.info("=" * 60)

        # 初始化组件
        logger.info("初始化智能体...")
        agents = initialize_agents(config)

        logger.info("初始化验证器...")
        verifier = VerilogVerifier()

        # 检查验证工具状态
        tool_status = verifier.get_tool_status()
        logger.info(f"验证工具状态: {tool_status}")

        logger.info("加载训练数据...")
        optimization_data = load_or_generate_data(config)
        logger.info(f"加载了 {len(optimization_data)} 条训练数据")

        # 初始化训练器
        logger.info("初始化训练器...")
        trainer = RTLMultiAgentTrainer(
            config=config,
            agents=agents,
            optimization_data=optimization_data,
            verifier=verifier
        )

        # 从检查点恢复（如果指定）
        if args.checkpoint:
            logger.info(f"从检查点恢复: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)

        # 执行指定模式
        if args.mode == "train":
            logger.info("开始训练...")
            training_results = trainer.train()
            logger.info("训练完成!")

            # 生成最终报告
            report = trainer.generate_final_report()
            logger.info("性能报告:")
            print(report)

        elif args.mode == "evaluate":
            logger.info("开始评估...")
            eval_results = trainer.evaluate(num_episodes=20)
            logger.info("评估完成!")
            logger.info(f"评估结果: {eval_results}")

        elif args.mode == "demo":
            logger.info("运行演示...")
            demo_results = run_demo(trainer, optimization_data[:3])
            logger.info("演示完成!")
            print("演示结果:", demo_results)

    except Exception as e:
        print(f"错误: {e}")
        logging.error(f"系统错误: {e}", exc_info=True)
        sys.exit(1)


def run_demo(trainer, demo_data: list) -> dict:
    """运行演示"""
    demo_results = []

    for i, case in enumerate(demo_data):
        print(f"\n{'='*50}")
        print(f"演示案例 {i+1}")
        print(f"{'='*50}")

        # 设置演示数据
        trainer.optimization_data = [case]

        # 运行一个episode
        episode_result = trainer.run_episode()

        demo_results.append({
            "case_id": case.get("case_id", f"demo_{i}"),
            "success": episode_result.get("success", False),
            "total_reward": episode_result.get("total_reward", 0.0),
            "steps": episode_result.get("total_steps", 0)
        })

        print(f"结果: 成功={episode_result.get('success', False)}, "
              f"奖励={episode_result.get('total_reward', 0.0):.3f}")

    return demo_results


if __name__ == "__main__":
    main()
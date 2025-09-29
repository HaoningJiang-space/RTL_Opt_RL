"""
BaseAgent基类
为RTL优化智能体提供统一的接口和基础功能
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import logging


class BaseAgent(ABC, nn.Module):
    """智能体基类，定义统一的接口和基础功能"""

    def __init__(
        self,
        model_name: str,
        agent_type: str,
        max_length: int = 4096,
        device: str = "auto"
    ):
        super().__init__()
        self.agent_type = agent_type
        self.model_name = model_name
        self.max_length = max_length

        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 初始化日志
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # 加载预训练的Verilog专用模型
        self._load_model()

        # 智能体特定配置
        self.agent_config = self._get_agent_config()

    def _load_model(self):
        """加载预训练模型"""
        try:
            self.logger.info(f"正在加载模型: {self.model_name}")

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载模型 - 使用CausalLM用于文本生成
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )

            self.model.eval()
            self.logger.info(f"模型加载成功，设备: {self.device}")

        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise

    @abstractmethod
    def _get_agent_config(self) -> Dict[str, Any]:
        """获取智能体特定配置"""
        pass

    @abstractmethod
    def generate_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """根据状态生成动作"""
        pass

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """生成文本的通用方法"""
        try:
            # Tokenize输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # 解码输出
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            self.logger.error(f"文本生成失败: {e}")
            return ""

    def extract_verilog_code(self, text: str) -> str:
        """从生成的文本中提取Verilog代码"""
        # 寻找代码块标记
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

        # 如果没有代码块，查找module关键字
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

            return "\n".join(verilog_lines).strip()

        return text.strip()

    def analyze_rtl_structure(self, rtl_code: str) -> Dict[str, Any]:
        """分析RTL代码结构"""
        analysis = {
            "module_name": "",
            "input_ports": [],
            "output_ports": [],
            "internal_signals": [],
            "always_blocks": 0,
            "assign_statements": 0,
            "complexity_score": 0.0
        }

        lines = rtl_code.split("\n")

        for line in lines:
            line = line.strip()

            # 提取模块名
            if line.startswith("module"):
                parts = line.split()
                if len(parts) >= 2:
                    analysis["module_name"] = parts[1].split("(")[0]

            # 统计端口
            if "input" in line and ";" in line:
                analysis["input_ports"].append(line)
            if "output" in line and ";" in line:
                analysis["output_ports"].append(line)

            # 统计内部信号
            if any(keyword in line for keyword in ["wire", "reg"]) and ";" in line:
                analysis["internal_signals"].append(line)

            # 统计always块和assign语句
            if line.startswith("always"):
                analysis["always_blocks"] += 1
            if line.startswith("assign"):
                analysis["assign_statements"] += 1

        # 计算复杂度分数（简单启发式）
        analysis["complexity_score"] = (
            len(analysis["input_ports"]) * 0.1 +
            len(analysis["output_ports"]) * 0.1 +
            len(analysis["internal_signals"]) * 0.2 +
            analysis["always_blocks"] * 0.3 +
            analysis["assign_statements"] * 0.1
        )

        return analysis

    def format_prompt(self, template: str, **kwargs) -> str:
        """格式化提示模板"""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            self.logger.error(f"提示模板格式化失败，缺少参数: {e}")
            return template

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "agent_type": self.agent_type,
            "device": str(self.device),
            "max_length": self.max_length,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None
        }

    def save_checkpoint(self, filepath: str):
        """保存智能体检查点"""
        checkpoint = {
            "agent_type": self.agent_type,
            "model_name": self.model_name,
            "agent_config": self.agent_config,
            "state_dict": self.state_dict() if hasattr(self, 'state_dict') else None
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"检查点已保存到: {filepath}")

    @classmethod
    def load_checkpoint(cls, filepath: str, **kwargs):
        """从检查点加载智能体"""
        checkpoint = torch.load(filepath)
        agent = cls(
            model_name=checkpoint["model_name"],
            agent_type=checkpoint["agent_type"],
            **kwargs
        )
        if checkpoint.get("state_dict"):
            agent.load_state_dict(checkpoint["state_dict"])
        return agent

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(type={self.agent_type}, model={self.model_name})"

    def __repr__(self) -> str:
        return self.__str__()
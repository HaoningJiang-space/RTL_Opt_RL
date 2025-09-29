"""
DataProcessor - 数据处理器
处理RTL优化序列数据，为训练做准备
"""

import json
import random
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging


class DataProcessor:
    """RTL优化数据处理器"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_optimization_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载优化序列数据"""
        data_path = Path(data_path)

        if not data_path.exists():
            self.logger.error(f"数据文件不存在: {data_path}")
            return []

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.suffix == '.json':
                    data = json.load(f)
                elif data_path.suffix == '.jsonl':
                    data = [json.loads(line) for line in f]
                else:
                    self.logger.error(f"不支持的文件格式: {data_path.suffix}")
                    return []

            self.logger.info(f"成功加载 {len(data)} 条优化数据")
            return self.process_raw_data(data)

        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return []

    def process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理原始数据"""
        processed_data = []

        for i, item in enumerate(raw_data):
            try:
                processed_item = self.process_single_item(item, i)
                if processed_item:
                    processed_data.append(processed_item)
            except Exception as e:
                self.logger.warning(f"处理第{i}条数据失败: {e}")

        self.logger.info(f"成功处理 {len(processed_data)} 条数据")
        return processed_data

    def process_single_item(self, item: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """处理单条数据"""

        # 必需字段检查
        required_fields = ["original_code", "optimized_code"]
        for field in required_fields:
            if field not in item:
                self.logger.warning(f"第{index}条数据缺少必需字段: {field}")
                return None

        # 构建标准化数据格式
        processed = {
            "case_id": item.get("case_id", f"case_{index}"),
            "original_code": item["original_code"],
            "optimized_code": item["optimized_code"],
            "optimization_sequence": item.get("optimization_sequence", []),
            "ppa_improvement": item.get("ppa_improvement", {}),
            "optimization_goal": item.get("optimization_goal", "balanced"),
            "constraints": item.get("constraints", {}),
            "metadata": item.get("metadata", {})
        }

        # 数据验证
        if not self.validate_item(processed):
            return None

        return processed

    def validate_item(self, item: Dict[str, Any]) -> bool:
        """验证数据项"""

        # 检查代码是否为空
        if not item["original_code"].strip() or not item["optimized_code"].strip():
            return False

        # 检查是否包含module声明
        if "module" not in item["original_code"] or "module" not in item["optimized_code"]:
            return False

        return True

    def create_train_test_split(
        self,
        data: List[Dict[str, Any]],
        test_ratio: float = 0.2,
        seed: int = 42
    ) -> Dict[str, List[Dict[str, Any]]]:
        """创建训练/测试分割"""

        random.seed(seed)
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)

        split_point = int(len(shuffled_data) * (1 - test_ratio))

        return {
            "train": shuffled_data[:split_point],
            "test": shuffled_data[split_point:]
        }

    def augment_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """数据增强"""
        augmented_data = data.copy()

        for item in data:
            # 创建逆向优化数据（从优化版本到原版本）
            if random.random() < 0.3:  # 30%概率
                reverse_item = {
                    **item,
                    "case_id": f"{item['case_id']}_reverse",
                    "original_code": item["optimized_code"],
                    "optimized_code": item["original_code"],
                    "ppa_improvement": self.reverse_ppa(item.get("ppa_improvement", {})),
                    "optimization_goal": "reverse_" + item.get("optimization_goal", "balanced")
                }
                augmented_data.append(reverse_item)

        self.logger.info(f"数据增强后: {len(augmented_data)} 条数据")
        return augmented_data

    def reverse_ppa(self, ppa: Dict[str, float]) -> Dict[str, float]:
        """反转PPA改善数据"""
        return {k: -v for k, v in ppa.items()}

    def save_processed_data(self, data: List[Dict[str, Any]], output_path: str):
        """保存处理后的数据"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"数据已保存到: {output_path}")
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")

    def generate_sample_data(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """生成示例数据用于测试"""

        sample_data = []

        for i in range(num_samples):
            sample = {
                "case_id": f"sample_{i}",
                "original_code": self.generate_sample_verilog(f"module_orig_{i}"),
                "optimized_code": self.generate_sample_verilog(f"module_opt_{i}"),
                "optimization_sequence": [
                    {"step": 1, "operation": "timing_optimization", "target": "critical_path"},
                    {"step": 2, "operation": "area_optimization", "target": "logic_blocks"}
                ],
                "ppa_improvement": {
                    "delay": random.uniform(0.05, 0.25),
                    "area": random.uniform(-0.05, 0.15),
                    "power": random.uniform(0.02, 0.18)
                },
                "optimization_goal": random.choice(["timing", "area", "power", "balanced"]),
                "constraints": {"max_area_increase": 0.1},
                "metadata": {"complexity": "medium", "domain": "test"}
            }
            sample_data.append(sample)

        return sample_data

    def generate_sample_verilog(self, module_name: str) -> str:
        """生成示例Verilog代码"""

        template = f"""module {module_name}(
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

endmodule"""

        return template
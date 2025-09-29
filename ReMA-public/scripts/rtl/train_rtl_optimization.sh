#!/bin/bash

# RTL优化训练脚本 - 基于ReMA框架
# 使用ReMA的PPO训练器进行RTL代码优化

set -e

# 获取脚本目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 默认配置
CONFIG_FILE="src/verl/verl/rema_trainer/config/rtl_ppo_trainer.yaml"
MODE="train"
WORKSPACE=${WORKSPACE:-$PROJECT_ROOT}
PROJECT_NAME=${PROJECT_NAME:-"rtl_optimization"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"rtl_rema_$(date +%Y%m%d_%H%M%S)"}
MODEL_PATH=${MODEL_PATH:-"deepseek-ai/deepseek-coder-6.7b-instruct"}

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
RTL优化训练脚本 - 基于ReMA框架

用法: $0 [选项]

选项:
  -c, --config CONFIG         配置文件路径 (默认: $CONFIG_FILE)
  -m, --mode MODE            运行模式 (train/eval/generation) (默认: $MODE)
  -w, --workspace PATH       工作空间路径 (默认: $WORKSPACE)
  -p, --project PROJECT      项目名称 (默认: $PROJECT_NAME)
  -e, --experiment EXP       实验名称 (默认: $EXPERIMENT_NAME)
  -M, --model MODEL_PATH     模型路径 (默认: $MODEL_PATH)
  --gpu GPU_COUNT            GPU数量 (默认: 自动检测)
  --quick-test               使用快速测试配置
  --generate-data            生成示例数据
  -h, --help                 显示帮助

示例:
  # 基础训练
  $0

  # 快速测试
  $0 --quick-test

  # 自定义配置
  $0 --config custom_config.yaml --experiment my_rtl_exp

  # 生成示例数据
  $0 --generate-data

环境变量:
  WORKSPACE      工作空间路径
  PROJECT_NAME   项目名称
  EXPERIMENT_NAME 实验名称
  MODEL_PATH     模型路径

EOF
}

# 解析命令行参数
QUICK_TEST=false
GENERATE_DATA=false
GPU_COUNT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -w|--workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        -p|--project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -e|--experiment)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        -M|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --gpu)
            GPU_COUNT="$2"
            shift 2
            ;;
        --quick-test)
            QUICK_TEST=true
            CONFIG_FILE="src/verl/verl/rema_trainer/config/rtl_quick_test.yaml"
            PROJECT_NAME="rtl_quick_test"
            EXPERIMENT_NAME="quick_test_$(date +%H%M%S)"
            shift
            ;;
        --generate-data)
            GENERATE_DATA=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 进入项目根目录
cd "$PROJECT_ROOT"

print_info "=================================="
print_info "RTL优化训练 - ReMA框架"
print_info "=================================="
print_info "项目目录: $PROJECT_ROOT"
print_info "配置文件: $CONFIG_FILE"
print_info "运行模式: $MODE"
print_info "项目名称: $PROJECT_NAME"
print_info "实验名称: $EXPERIMENT_NAME"
print_info "模型路径: $MODEL_PATH"

# 检查配置文件
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查Python环境
print_info "检查Python环境..."
if ! command -v python &> /dev/null; then
    print_error "Python未找到"
    exit 1
fi

python_version=$(python --version 2>&1)
print_success "Python版本: $python_version"

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    if [[ -z "$GPU_COUNT" ]]; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    fi
    print_success "检测到 $GPU_COUNT 个GPU"
else
    print_warning "未检测到NVIDIA GPU，将使用CPU训练"
    GPU_COUNT=0
fi

# 检查验证工具
print_info "检查Verilog验证工具..."
if command -v verilator &> /dev/null; then
    print_success "Verilator: 可用"
else
    print_warning "Verilator: 不可用"
fi

if command -v yosys &> /dev/null; then
    print_success "Yosys: 可用"
else
    print_warning "Yosys: 不可用"
fi

if command -v iverilog &> /dev/null; then
    print_success "Icarus Verilog: 可用"
else
    print_warning "Icarus Verilog: 不可用"
fi

# 创建必要的目录
print_info "创建目录..."
mkdir -p logs data/rtl_optimization data/rtl_test models

# 生成示例数据（如果需要）
if [[ "$GENERATE_DATA" == "true" ]] || [[ "$QUICK_TEST" == "true" ]]; then
    print_info "生成RTL优化示例数据..."
    python << 'EOF'
import os
import json
import pandas as pd
from pathlib import Path

def generate_sample_verilog(module_name, variant="basic"):
    """生成示例Verilog代码"""
    if variant == "basic":
        return f'''module {module_name}(
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
endmodule'''
    elif variant == "optimized":
        return f'''module {module_name}(
    input clk,
    input rst,
    input [7:0] data_in,
    output reg [7:0] data_out
);
    // 优化版本：直接计算，减少寄存器使用
    always @(posedge clk) begin
        if (rst) begin
            data_out <= 8'b0;
        end else begin
            data_out <= data_in + 1;
        end
    end
endmodule'''

def create_rtl_dataset(num_samples=100, output_dir="data/rtl_test"):
    """创建RTL优化数据集"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_data = []
    val_data = []

    for i in range(num_samples):
        module_name = f"test_module_{i}"
        original_code = generate_sample_verilog(module_name, "basic")
        optimized_code = generate_sample_verilog(module_name, "optimized")

        # 构建对话格式
        question = f"请优化以下Verilog代码，减少资源使用并提高性能：\n\n```verilog\n{original_code}\n```"
        answer = f"以下是优化后的Verilog代码：\n\n```verilog\n{optimized_code}\n```\n\n优化点：\n1. 减少了中间寄存器temp_reg的使用\n2. 直接计算输出，减少了一个时钟周期的延迟\n3. 节省了面积资源"

        data_point = {
            "question": question,
            "answer": answer,
            "original_code": original_code,
            "optimized_code": optimized_code,
            "optimization_type": "area_timing"
        }

        # 80-20分割
        if i < num_samples * 0.8:
            train_data.append(data_point)
        else:
            val_data.append(data_point)

    # 保存为parquet格式
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    train_df.to_parquet(f"{output_dir}/sample_train.parquet", index=False)
    val_df.to_parquet(f"{output_dir}/sample_val.parquet", index=False)

    print(f"生成了 {len(train_data)} 个训练样本和 {len(val_data)} 个验证样本")
    print(f"数据保存在: {output_dir}")

# 生成数据
create_rtl_dataset(50 if "$QUICK_TEST" == "true" else 200)
EOF
    print_success "示例数据生成完成"
fi

# 构建训练命令
if [[ "$GPU_COUNT" -gt 1 ]]; then
    PYTHON_CMD="python -m verl.rema_trainer.main_ppo"
else
    PYTHON_CMD="python src/verl/verl/rema_trainer/main_ppo.py"
fi

# 设置环境变量
export WORKSPACE="$WORKSPACE"
export PROJECT_NAME="$PROJECT_NAME"
export EXPERIMENT_NAME="$EXPERIMENT_NAME"
export MODEL_PATH="$MODEL_PATH"

# 构建完整命令
FULL_CMD="$PYTHON_CMD \\
    trainer.project_name=$PROJECT_NAME \\
    trainer.experiment_name=$EXPERIMENT_NAME \\
    trainer.nnodes=1 \\
    trainer.n_gpus_per_node=${GPU_COUNT:-1}"

# 添加配置文件参数
if [[ "$CONFIG_FILE" != "src/verl/verl/rema_trainer/config/ppo_trainer.yaml" ]]; then
    FULL_CMD="$FULL_CMD --config-path $(dirname $CONFIG_FILE) --config-name $(basename $CONFIG_FILE .yaml)"
fi

# 添加模型路径覆盖
FULL_CMD="$FULL_CMD \\
    actor_rollout_ref.model.path=$MODEL_PATH"

# 快速测试模式的特殊参数
if [[ "$QUICK_TEST" == "true" ]]; then
    FULL_CMD="$FULL_CMD \\
        trainer.total_epochs=3 \\
        trainer.total_training_steps=50 \\
        data.train_batch_size=16"
fi

print_info "执行命令:"
echo -e "${BLUE}$FULL_CMD${NC}"

# 显示重要提示
print_warning "开始训练前请确认:"
print_warning "1. 已正确设置模型路径和访问权限"
print_warning "2. 有足够的磁盘空间存储检查点"
print_warning "3. 网络连接正常（用于下载模型）"

if [[ "$QUICK_TEST" == "false" ]]; then
    print_warning "4. 这不是快速测试，训练可能需要较长时间"
fi

read -p "按Enter键继续，或Ctrl+C取消..."

print_info "开始RTL优化训练..."
echo

# 执行训练
eval $FULL_CMD
EOF
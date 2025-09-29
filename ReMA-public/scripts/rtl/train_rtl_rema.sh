#!/bin/bash

# RTL优化训练脚本 - 完全基于ReMA框架
# 使用ReMA的PPO训练器和自定义RTL reward函数

set -e

# 获取脚本目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 默认配置
CONFIG_NAME="rtl_ppo_trainer"
WORKSPACE=${WORKSPACE:-$PROJECT_ROOT}
PROJECT_NAME=${PROJECT_NAME:-"rtl_optimization"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"rtl_rema_$(date +%Y%m%d_%H%M%S)"}
MODEL_PATH=${MODEL_PATH:-"deepseek-ai/deepseek-coder-6.7b-instruct"}

# 训练参数
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
TOTAL_STEPS=${TOTAL_STEPS:-1000}

# 数据路径
DATA_DIR="$PROJECT_ROOT/data/rtl_optimization"
TRAIN_FILE="$DATA_DIR/train.parquet"
VAL_FILE="$DATA_DIR/val.parquet"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    cat << EOF
RTL优化训练脚本 - 基于ReMA框架

用法: $0 [选项]

基础选项:
  -c, --config CONFIG        配置文件名 (默认: $CONFIG_NAME)
  -p, --project PROJECT      项目名称 (默认: $PROJECT_NAME)
  -e, --experiment EXP       实验名称 (默认: auto-generated)
  -M, --model MODEL_PATH     模型路径 (默认: $MODEL_PATH)

训练选项:
  --nnodes N                节点数量 (默认: $NNODES)
  --gpus N                  每节点GPU数 (默认: auto-detect)
  --epochs N                训练轮数 (默认: $TOTAL_EPOCHS)
  --steps N                 训练步数 (默认: $TOTAL_STEPS)

数据选项:
  --data-dir DIR            数据目录 (默认: $DATA_DIR)
  --generate-data           自动生成训练数据
  --quick-test              快速测试模式

其他选项:
  --dry-run                 显示命令但不执行
  -h, --help                显示帮助

示例:
  # 标准训练
  $0

  # 快速测试
  $0 --quick-test --generate-data

  # 自定义配置
  $0 --config rtl_quick_test --epochs 5 --steps 100

环境变量:
  WORKSPACE          工作空间路径
  PROJECT_NAME       项目名称
  EXPERIMENT_NAME    实验名称
  MODEL_PATH         模型路径
  CUDA_VISIBLE_DEVICES GPU设备

EOF
}

# 解析命令行参数
GENERATE_DATA=false
QUICK_TEST=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_NAME="$2"
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
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --gpus)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --epochs)
            TOTAL_EPOCHS="$2"
            shift 2
            ;;
        --steps)
            TOTAL_STEPS="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            TRAIN_FILE="$DATA_DIR/train.parquet"
            VAL_FILE="$DATA_DIR/val.parquet"
            shift 2
            ;;
        --generate-data)
            GENERATE_DATA=true
            shift
            ;;
        --quick-test)
            QUICK_TEST=true
            CONFIG_NAME="rtl_quick_test"
            PROJECT_NAME="rtl_quick_test"
            DATA_DIR="$PROJECT_ROOT/data/rtl_test"
            TRAIN_FILE="$DATA_DIR/train.parquet"
            VAL_FILE="$DATA_DIR/val.parquet"
            TOTAL_EPOCHS=3
            TOTAL_STEPS=50
            GENERATE_DATA=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
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
print_info "配置名称: $CONFIG_NAME"
print_info "项目名称: $PROJECT_NAME"
print_info "实验名称: $EXPERIMENT_NAME"
print_info "模型路径: $MODEL_PATH"
print_info "节点数量: $NNODES"
print_info "GPU数量: $GPUS_PER_NODE"
print_info "训练轮数: $TOTAL_EPOCHS"
print_info "训练步数: $TOTAL_STEPS"
print_info "数据目录: $DATA_DIR"

# 检查配置文件
CONFIG_PATH="src/verl/verl/rema_trainer/config/${CONFIG_NAME}.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
    print_error "配置文件不存在: $CONFIG_PATH"
    print_info "可用配置文件："
    ls src/verl/verl/rema_trainer/config/*.yaml | xargs basename -s .yaml
    exit 1
fi
print_success "配置文件: $CONFIG_PATH"

# 检查Python环境
print_info "检查Python环境..."
if ! command -v python &> /dev/null; then
    print_error "Python未找到"
    exit 1
fi

python_version=$(python --version 2>&1)
print_success "Python: $python_version"

# 检查关键依赖
print_info "检查依赖..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || { print_error "PyTorch未安装"; exit 1; }
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || { print_error "Transformers未安装"; exit 1; }
python -c "import verl; print('VeRL: 可用')" || { print_error "VeRL框架未安装"; exit 1; }

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    print_success "检测到 $GPUS_PER_NODE 个GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | nl -v0 | while read gpu_info; do
        print_info "GPU $gpu_info"
    done
else
    print_warning "未检测到NVIDIA GPU"
    GPUS_PER_NODE=0
fi

# 检查验证工具
print_info "检查Verilog验证工具..."
for tool in verilator yosys iverilog; do
    if command -v $tool &> /dev/null; then
        version=$(timeout 5s $tool --version 2>&1 | head -1 || echo "版本未知")
        print_success "$tool: $version"
    else
        print_warning "$tool: 不可用"
    fi
done

# 生成数据（如果需要）
if [[ "$GENERATE_DATA" == "true" ]] || [[ ! -f "$TRAIN_FILE" ]] || [[ ! -f "$VAL_FILE" ]]; then
    print_info "生成RTL训练数据..."

    if [[ "$QUICK_TEST" == "true" ]]; then
        python scripts/data/generate_rtl_data.py --quick --output_dir "$DATA_DIR"
    else
        python scripts/data/generate_rtl_data.py --num_samples 500 --output_dir "$DATA_DIR"
    fi

    if [[ $? -eq 0 ]]; then
        print_success "数据生成完成"
    else
        print_error "数据生成失败"
        exit 1
    fi
else
    print_success "使用现有训练数据"
fi

# 验证数据文件
if [[ ! -f "$TRAIN_FILE" ]]; then
    print_error "训练数据不存在: $TRAIN_FILE"
    exit 1
fi

if [[ ! -f "$VAL_FILE" ]]; then
    print_error "验证数据不存在: $VAL_FILE"
    exit 1
fi

print_success "训练数据: $TRAIN_FILE"
print_success "验证数据: $VAL_FILE"

# 创建必要的目录
mkdir -p logs models

# 构建训练命令
TRAIN_CMD="python -m verl.rema_trainer.main_ppo"

# 构建参数列表
TRAIN_ARGS=(
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.nnodes=$NNODES"
    "trainer.n_gpus_per_node=$GPUS_PER_NODE"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.total_training_steps=$TOTAL_STEPS"
    "data.train_files=$TRAIN_FILE"
    "data.val_files=$VAL_FILE"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "--config-path=src/verl/verl/rema_trainer/config"
    "--config-name=$CONFIG_NAME"
)

# 添加额外的训练参数
if [[ "$QUICK_TEST" == "true" ]]; then
    TRAIN_ARGS+=(
        "data.train_batch_size=16"
        "actor_rollout_ref.rollout.n=2"
        "actor_rollout_ref.rollout.max_num_turns=5"
    )
fi

# 显示完整命令
print_info "训练命令:"
echo -e "${BLUE}$TRAIN_CMD \\"
for arg in "${TRAIN_ARGS[@]}"; do
    echo "    $arg \\"
done
echo -e "${NC}"

# 执行检查
print_info "执行前检查..."

# 测试reward函数
print_info "测试RTL奖励函数..."
python -c "
from src.verl.verl.utils.reward_score.rtl_optimization import compute_score
test_code = '''module test(input clk, output reg [7:0] out); always @(posedge clk) out <= 8'h42; endmodule'''
score = compute_score('rtl_optimization', 'Optimized code: ' + test_code, test_code)
print(f'RTL奖励函数测试: {score:.3f}')
assert 0 <= score <= 1, '奖励分数应在0-1范围内'
print('✓ RTL奖励函数正常工作')
" || { print_error "RTL奖励函数测试失败"; exit 1; }

if [[ "$DRY_RUN" == "true" ]]; then
    print_warning "DRY RUN模式 - 命令未执行"
    exit 0
fi

# 最终确认
print_warning "即将开始训练，请确认："
print_warning "1. GPU内存充足"
print_warning "2. 磁盘空间充足"
print_warning "3. 网络连接正常"
if [[ "$QUICK_TEST" == "false" ]]; then
    print_warning "4. 这不是测试，将进行完整训练"
fi

read -p "按Enter继续，或Ctrl+C取消..." -t 10 || echo ""

print_info "🚀 开始RTL优化训练..."
echo

# 执行训练
exec $TRAIN_CMD "${TRAIN_ARGS[@]}"
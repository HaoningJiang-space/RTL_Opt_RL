#!/bin/bash

# RTL多智能体优化系统训练脚本

set -e

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 默认参数
CONFIG_FILE="rtl_multi_agent/configs/default_config.yaml"
MODE="train"
CHECKPOINT=""
CONDA_ENV="rema"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

# 显示帮助
show_help() {
    echo "RTL多智能体优化系统训练脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -c, --config CONFIG_FILE    配置文件路径 (默认: $CONFIG_FILE)"
    echo "  -m, --mode MODE             运行模式 (train/evaluate/demo, 默认: $MODE)"
    echo "  -r, --checkpoint PATH       从检查点恢复训练"
    echo "  -e, --env ENV_NAME          Conda环境名称 (默认: $CONDA_ENV)"
    echo "  -h, --help                  显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --config rtl_multi_agent/configs/quick_test_config.yaml"
    echo "  $0 --mode evaluate --checkpoint ./experiments/checkpoint_epoch_10"
    echo "  $0 --mode demo"
}

# 解析命令行参数
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
        -r|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -e|--env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_message $RED "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 进入项目根目录
cd "$PROJECT_ROOT"

# 检查配置文件
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_message $RED "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

print_message $BLUE "=" * 60
print_message $BLUE "RTL多智能体优化系统训练"
print_message $BLUE "=" * 60

print_message $GREEN "配置文件: $CONFIG_FILE"
print_message $GREEN "运行模式: $MODE"
print_message $GREEN "项目目录: $PROJECT_ROOT"

if [[ -n "$CHECKPOINT" ]]; then
    print_message $GREEN "检查点路径: $CHECKPOINT"
fi

# 检查并激活Conda环境
if command -v conda >/dev/null 2>&1; then
    print_message $YELLOW "激活Conda环境: $CONDA_ENV"
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if conda info --envs | grep -q "^$CONDA_ENV "; then
        conda activate "$CONDA_ENV"
        print_message $GREEN "成功激活环境: $CONDA_ENV"
    else
        print_message $YELLOW "环境 $CONDA_ENV 不存在，使用当前环境"
    fi
else
    print_message $YELLOW "Conda未安装，使用当前Python环境"
fi

# 检查Python和依赖
print_message $YELLOW "检查Python环境..."
python --version
echo "Python路径: $(which python)"

# 检查关键依赖
print_message $YELLOW "检查依赖..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || print_message $RED "PyTorch未安装"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || print_message $RED "Transformers未安装"

# 检查验证工具
print_message $YELLOW "检查验证工具..."
verilator --version 2>/dev/null && print_message $GREEN "Verilator: 可用" || print_message $YELLOW "Verilator: 不可用"
yosys -V 2>/dev/null | head -1 && print_message $GREEN "Yosys: 可用" || print_message $YELLOW "Yosys: 不可用"
iverilog -V 2>/dev/null | head -1 && print_message $GREEN "Icarus Verilog: 可用" || print_message $YELLOW "Icarus Verilog: 不可用"

# 创建必要的目录
print_message $YELLOW "创建目录..."
mkdir -p logs data models experiments

# 构建Python命令
PYTHON_CMD="python rtl_multi_agent/main.py --config $CONFIG_FILE --mode $MODE"

if [[ -n "$CHECKPOINT" ]]; then
    PYTHON_CMD="$PYTHON_CMD --checkpoint $CHECKPOINT"
fi

# 显示即将执行的命令
print_message $BLUE "执行命令: $PYTHON_CMD"

# 执行训练
print_message $GREEN "开始执行..."
echo ""

# 使用exec替换当前shell，这样可以正确传递信号
exec $PYTHON_CMD
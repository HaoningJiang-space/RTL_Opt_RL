# RTL多智能体优化系统

基于ReMA框架的RTL代码优化强化学习系统，利用多智能体协作实现高质量的Verilog代码优化。

## 🏗️ 系统架构

### 核心组件

1. **奖励系统** (`src/verl/verl/utils/reward_score/rtl_optimization.py`)
   - 集成Verilator、Yosys、Icarus Verilog验证工具
   - 多维度奖励计算：语法(40%) + 综合(30%) + 优化效果(30%)
   - 符合ReMA框架的reward接口规范

2. **配置系统**
   - `src/verl/verl/rema_trainer/config/rtl_ppo_trainer.yaml` - 标准训练配置
   - `src/verl/verl/rema_trainer/config/rtl_quick_test.yaml` - 快速测试配置

3. **数据生成** (`scripts/data/generate_rtl_data.py`)
   - 自动生成ReMA格式的多轮对话数据
   - 支持不同复杂度和优化类型的RTL代码

4. **训练脚本** (`scripts/rtl/train_rtl_rema.sh`)
   - 完全基于ReMA框架的训练流程
   - 自动环境检测和数据生成

## 🚀 快速开始

### 1. 环境准备

```bash
# 1. 创建conda环境
conda create -n rema_rtl python=3.10
conda activate rema_rtl

# 2. 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets

# 3. 安装ReMA框架依赖
cd src/verl
pip install -e .
cd ../360-LLaMA-Factory
pip install -e .
cd ../..

# 4. 安装验证工具（可选但推荐）
# Ubuntu/Debian:
sudo apt-get install verilator yosys iverilog

# macOS:
brew install verilator yosys icarus-verilog
```

### 2. 快速测试

```bash
# 运行快速测试
bash scripts/rtl/train_rtl_rema.sh --quick-test --generate-data

# 或者分步执行：
# 1. 生成测试数据
python scripts/data/generate_rtl_data.py --quick

# 2. 测试奖励函数
python scripts/test/test_rtl_reward.py

# 3. 运行训练
bash scripts/rtl/train_rtl_rema.sh --quick-test
```

### 3. 标准训练

```bash
# 生成训练数据
python scripts/data/generate_rtl_data.py --num_samples 1000

# 开始训练
bash scripts/rtl/train_rtl_rema.sh \
    --project rtl_optimization_v1 \
    --experiment my_rtl_exp \
    --epochs 20 \
    --steps 2000
```

## 📊 奖励机制详解

### 奖励计算公式

```
总奖励 = 语法分数 × 0.4 + 综合分数 × 0.3 + 优化效果分数 × 0.3 + 奖励分
```

### 验证层级

1. **语法验证** (Verilator)
   - 检查Verilog语法正确性
   - 必须通过才能获得基础分数

2. **综合验证** (Yosys)
   - 验证代码可综合性
   - 提取资源使用统计

3. **优化效果评估**
   - 比较原始代码与优化代码的PPA指标
   - 计算面积、时序、功耗改善

### 支持的数据源

- `rtl_optimization`: 通用RTL优化任务
- `rtl_math`: RTL数学推理任务
- `rtl_generation`: RTL代码生成任务
- `verilog_optimization`: Verilog优化任务

## 🔧 高级配置

### 模型配置

推荐的Verilog专用模型：

```yaml
# 配置文件中的模型设置
actor_rollout_ref:
  model:
    path: deepseek-ai/deepseek-coder-6.7b-instruct
    # 备选模型：
    # path: henryen/OriGen_Fix
    # path: Nellyw888/VeriReason-Qwen2.5-7b-RTLCoder-Verilog-GRPO-reasoning-tb
```

### 训练参数调优

```yaml
# ReMA核心参数
actor_rollout_ref:
  actor:
    clip_mode: turn      # turn-level clipping
    agg_mode: trajectory # trajectory aggregation
    optim:
      lr: 5e-6          # 较小学习率适合代码任务

  rollout:
    max_num_turns: 15   # 支持多轮优化
    n: 8               # rollout数量
```

### 自定义奖励权重

```yaml
# 在配置文件中调整奖励权重
reward_model:
  verification_tools:
    enable_verilator: true
    enable_yosys: true
    enable_iverilog: true

  syntax_weight: 0.4    # 语法权重
  synthesis_weight: 0.4  # 综合权重
  ppa_weight: 0.2       # PPA权重
```

## 📝 数据格式

### 训练数据格式

```json
{
  "data_source": "rtl_optimization",
  "question": "请优化以下Verilog代码...",
  "response": "优化后的代码：...",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "meta_thinking", "content": "..."},
    {"role": "reasoning", "content": "..."}
  ],
  "ground_truth": "优化后的Verilog代码",
  "extra_info": {
    "original_code": "原始Verilog代码",
    "optimization_goal": "timing",
    "expected_improvement": {...}
  }
}
```

### 多轮对话结构

1. **用户提问**: 提供原始RTL代码和优化需求
2. **Meta-thinking**: 高层次分析和策略制定
3. **Reasoning**: 具体优化实现和代码生成

## 🛠️ 开发和调试

### 测试奖励函数

```python
from verl.utils.reward_score.rtl_optimization import compute_score

score = compute_score(
    data_source="rtl_optimization",
    solution_str="优化后的代码...",
    ground_truth="原始代码...",
    extra_info={"original_code": "..."}
)
print(f"奖励分数: {score}")
```

### 调试训练过程

```bash
# 启用详细日志
export VERL_LOG_LEVEL=DEBUG

# 保存中间结果
bash scripts/rtl/train_rtl_rema.sh \
    --config rtl_quick_test \
    --dry-run  # 只显示命令不执行
```

### 监控训练

- 检查 `logs/` 目录下的日志文件
- 使用W&B监控训练进度（如果配置了）
- 查看 `models/` 目录下的检查点

## 📈 评估和对比

### 与基线对比

系统自动与以下基线对比：
- ABC综合工具默认优化
- Yosys默认综合流程
- 人工专家优化结果

### 性能指标

- **成功率**: 语法和综合通过率
- **优化效果**: PPA指标改善程度
- **代码质量**: 可读性和维护性评分

## 🔍 故障排除

### 常见问题

1. **验证工具不可用**
   ```bash
   # 检查工具安装
   which verilator yosys iverilog

   # 安装缺失工具
   sudo apt-get install verilator yosys iverilog
   ```

2. **GPU内存不足**
   ```bash
   # 减少批量大小
   bash scripts/rtl/train_rtl_rema.sh --config rtl_quick_test
   ```

3. **模型下载失败**
   ```bash
   # 使用本地模型或镜像
   export HF_ENDPOINT=https://hf-mirror.com
   ```

### 调试命令

```bash
# 测试系统完整性
python scripts/test/test_rtl_reward.py

# 验证数据格式
python scripts/data/generate_rtl_data.py --quick

# 检查配置
python -c "from omegaconf import OmegaConf; print(OmegaConf.load('src/verl/verl/rema_trainer/config/rtl_ppo_trainer.yaml'))"
```

## 🤝 贡献和扩展

### 添加新的优化类型

1. 在 `generate_rtl_data.py` 中添加新的优化策略
2. 在 `rtl_optimization.py` 中扩展奖励计算逻辑
3. 更新配置文件中的相关参数

### 集成新的验证工具

1. 在 `RTLVerificationTools` 类中添加新工具检测
2. 实现对应的验证方法
3. 更新奖励计算权重

## 📚 参考资料

- [ReMA论文](https://arxiv.org/abs/2503.09501): ReMA: Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning
- [VeRL框架](https://github.com/volcengine/verl): Volcano Engine Reinforcement Learning for LLM
- [RTL优化最佳实践](./docs/rtl_optimization_guide.md)

## 📄 许可证

本项目基于ReMA框架开发，遵循Apache 2.0许可证。
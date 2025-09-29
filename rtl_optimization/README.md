# RTL优化系统 / RTL Optimization System

基于图神经网络和强化学习的RTL网表优化系统，使用ABC工具进行逻辑等效优化。

RTL netlist optimization system based on Graph Neural Networks and Reinforcement Learning, using ABC tool for logic-equivalent optimization.

## 🌟 主要特性 / Key Features

- **🔧 ABC集成**: 使用成熟的ABC工具执行逻辑等效优化
- **🧠 智能决策**: 基于强化学习的优化策略学习
- **⚡ 高效评估**: 混合评估系统，结合GNN预测和真实综合
- **📊 图表示**: 专门针对RTL网表设计的轻量级图表示
- **🔄 完整流程**: 从RTL解析到优化决策的端到端解决方案

## 🏗️ 系统架构 / System Architecture

```
rtl_optimization/
├── utils/              # 配置和工具类
│   ├── config.py      # 系统配置
│   └── __init__.py
├── tools/              # 核心工具
│   ├── abc_interface.py    # ABC工具接口
│   ├── evaluator.py        # 混合评估系统
│   └── __init__.py
├── graph/              # 图表示和GNN模型
│   ├── rtl_graph.py        # RTL图构建
│   ├── gnn_model.py        # GNN模型
│   └── __init__.py
├── rl/                 # 强化学习组件
│   ├── environment.py      # RL环境
│   ├── ppo_agent.py        # PPO算法
│   └── __init__.py
├── examples/           # 使用示例
│   ├── basic_usage.py      # 基本使用示例
│   └── __init__.py
└── __init__.py         # 主模块入口
```

## 📋 依赖要求 / Dependencies

### 必需依赖 / Required Dependencies
```bash
# Python 基础依赖
numpy>=1.21.0
torch>=1.12.0
torch-geometric>=2.1.0

# 图处理
networkx>=2.6

# 其他
gymnasium>=0.26.0  # 或 gym>=0.21.0
```

### 外部工具 / External Tools
```bash
# ABC逻辑综合工具 (必需)
abc

# Yosys综合工具 (可选，用于更精确的PPA评估)
yosys
```

## 🚀 快速开始 / Quick Start

### 1. 安装依赖 / Install Dependencies

```bash
# 安装Python依赖
pip install torch torch-geometric networkx gymnasium numpy

# 安装ABC (Ubuntu/Debian)
sudo apt-get install abc

# 或从源码编译ABC
git clone https://github.com/berkeley-abc/abc.git
cd abc && make
```

### 2. 基本使用 / Basic Usage

```python
from rtl_optimization import quick_start

# RTL文件列表
rtl_files = [
    "path/to/design1.v",
    "path/to/design2.v",
    # ...
]

# 快速开始训练
result = quick_start(
    rtl_files=rtl_files,
    num_episodes=1000,
    config_path=None  # 使用默认配置
)

print(f"训练完成，平均奖励: {result['final_stats']['recent_avg_reward']}")
```

### 3. 详细使用 / Detailed Usage

```python
from rtl_optimization import (
    get_config,
    create_optimization_pipeline
)

# 1. 加载配置
config = get_config("config.json")  # 或使用默认配置

# 2. 创建优化流水线
pipeline = create_optimization_pipeline(rtl_files, config)

# 3. 训练智能体
training_history = pipeline["agent"].train(num_episodes=1000)

# 4. 评估性能
eval_result = pipeline["agent"].evaluate(num_episodes=50)
print(f"评估结果: {eval_result}")
```

## 🔧 配置系统 / Configuration System

系统采用分层配置设计，支持以下主要配置模块：

### 图配置 / Graph Configuration
```python
config.graph.total_node_feature_dim = 24  # 节点特征维度
config.graph.edge_types = ["connection", "timing", "fanout"]
```

### GNN配置 / GNN Configuration
```python
config.gnn.hidden_dim = 128        # 隐藏层维度
config.gnn.num_conv_layers = 2     # 图卷积层数
config.gnn.dropout_rate = 0.1      # Dropout率
```

### RL配置 / RL Configuration
```python
config.rl.max_steps_per_episode = 20    # 每回合最大步数
config.rl.learning_rate_policy = 3e-4   # 策略网络学习率
config.rl.clip_ratio = 0.2              # PPO截断比例
```

### ABC配置 / ABC Configuration
```python
config.abc.abc_commands = {
    "rewrite": "rewrite -l",
    "refactor": "refactor -l",
    # ... 更多ABC命令
}
```

## 🎯 核心组件详解 / Core Components

### 1. ABC接口 / ABC Interface

```python
from rtl_optimization.tools import ABCInterface

with ABCInterface() as abc:
    # Verilog转AIG
    aig_file = abc.verilog_to_aig("design.v")

    # 应用优化
    result = abc.apply_optimization(aig_file, "rewrite")

    # 验证等效性
    is_equiv = abc.verify_equivalence(original, optimized)
```

### 2. RTL图表示 / RTL Graph Representation

```python
from rtl_optimization.graph import RTLNetlistGraph

graph_builder = RTLNetlistGraph()

# 构建图
graph_data = graph_builder.build_from_file("design.v", "verilog")

# 获取统计信息
stats = graph_builder.get_graph_statistics(graph_data)
print(f"节点数: {stats['num_nodes']}, 边数: {stats['num_edges']}")
```

### 3. GNN模型 / GNN Model

```python
from rtl_optimization.graph import RTLOptimizationGNN

# 创建模型
model = RTLOptimizationGNN()

# 预测PPA
ppa_result = model.predict_ppa(graph_data)
print(f"预测延迟: {ppa_result['delay']}")

# 获取图嵌入
embedding = model.get_graph_embedding(graph_data)
```

### 4. 混合评估系统 / Hybrid Evaluation System

```python
from rtl_optimization.tools import HybridEvaluationSystem

evaluator = HybridEvaluationSystem()

# 快速评估（GNN预测）
quick_result = evaluator.quick_evaluate("design.v")

# 完整评估（真实综合）
full_result = evaluator.full_evaluate("design.v")

# 混合评估（智能选择）
hybrid_result = evaluator.evaluate("design.v", step_count=5)
```

### 5. 强化学习环境 / RL Environment

```python
from rtl_optimization.rl import RTLOptimizationEnvironment

env = RTLOptimizationEnvironment(rtl_dataset)

# 重置环境
state = env.reset()

# 执行动作
next_state, reward, done, info = env.step(action)

# 获取回合总结
summary = env.get_episode_summary()
```

### 6. PPO算法 / PPO Algorithm

```python
from rtl_optimization.rl import RTLOptimizationPPO

agent = RTLOptimizationPPO(env)

# 训练
training_history = agent.train(num_episodes=1000)

# 评估
eval_result = agent.evaluate(num_episodes=50)

# 保存模型
agent.save_checkpoint("model.pth")
```

## 📊 支持的优化操作 / Supported Optimization Operations

系统支持以下ABC优化命令：

| 命令 | 描述 | 优化目标 |
|------|------|----------|
| `rewrite` | AIG重写 | 节点数减少 |
| `refactor` | 重构优化 | 深度和节点平衡 |
| `balance` | 平衡AIG深度 | 延迟优化 |
| `resub` | 替换优化 | 面积优化 |
| `compress2` | 综合压缩 | 整体优化 |
| `choice` | 选择计算 | 结构优化 |
| `fraig` | 功能归约 | 冗余消除 |
| `dch` | 深度选择计算 | 高级优化 |
| `if` | FPGA映射 | FPGA优化 |
| `mfs` | 最大扇入简化 | 扇入优化 |
| `lutpack` | LUT打包 | FPGA面积优化 |

## 🧪 实验和评估 / Experiments and Evaluation

### 运行示例 / Running Examples

```bash
# 基本使用示例
cd rtl_optimization/examples
python basic_usage.py

# 自定义实验
python custom_experiment.py --config config.json --episodes 1000
```

### 评估指标 / Evaluation Metrics

- **PPA指标**: 延迟 (Delay)、面积 (Area)、功耗 (Power)
- **训练指标**: 平均奖励、收敛速度、成功率
- **效率指标**: 评估速度、缓存命中率

### 基线对比 / Baseline Comparison

系统支持与以下基线方法对比：
- Yosys默认优化
- ABC默认优化序列
- 随机搜索
- IronMan基线（如果可用）

## 🔬 技术细节 / Technical Details

### 图表示设计 / Graph Representation Design

- **节点类型**: 门 (Gate)、寄存器 (Register)、端口 (Port)、连线 (Wire)
- **节点特征**: 24维向量 (类型编码 + 门类型 + 数值特征 + 时序特征)
- **边类型**: 连接关系、时序路径、扇出关系

### 奖励函数设计 / Reward Function Design

```python
总奖励 = PPA改善奖励 + ABC改善奖励 + 动作特异性奖励 + 等效性奖励 + 约束惩罚
```

- **PPA改善**: 基于延迟、面积、功耗的加权改善
- **ABC改善**: 基于节点数和深度减少的中间反馈
- **动作特异性**: 不同优化命令的特定奖励
- **等效性保证**: ABC自动保证的逻辑等效性奖励

### 混合评估策略 / Hybrid Evaluation Strategy

- **快速评估**: GNN预测，毫秒级响应
- **完整评估**: 真实综合，秒级精确结果
- **智能切换**: 基于置信度和训练进度的动态选择

## 🐛 故障排除 / Troubleshooting

### 常见问题 / Common Issues

1. **ABC不可用**
   ```bash
   # 检查ABC安装
   which abc
   abc -h
   ```

2. **PyTorch Geometric安装问题**
   ```bash
   # 根据PyTorch版本安装
   pip install torch-geometric
   ```

3. **内存不足**
   ```python
   # 减少批量大小
   config.rl.batch_size = 32
   config.rl.buffer_size = 1024
   ```

4. **训练不收敛**
   ```python
   # 调整学习率
   config.rl.learning_rate_policy = 1e-4
   config.rl.learning_rate_value = 5e-4
   ```

### 调试模式 / Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
```

## 📈 性能优化建议 / Performance Optimization Tips

1. **并行处理**: 使用多个环境并行训练
2. **缓存优化**: 启用评估缓存减少重复计算
3. **批量优化**: 合理设置批量大小
4. **模型压缩**: 使用更小的GNN模型进行快速原型

## 🤝 贡献指南 / Contributing

欢迎贡献代码、报告问题或提出改进建议！

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证 / License

[MIT License](LICENSE)

## 📚 参考文献 / References

1. IronMan系列论文 (GLSVLSI'21, TCAD'22)
2. CircuitNet数据集 (2024)
3. ABC工具文档
4. PyTorch Geometric文档

## 📞 联系方式 / Contact

如有问题或建议，请通过以下方式联系：

- 提交Issue: [GitHub Issues](https://github.com/your-repo/issues)
- 邮件: 12210308@mail.sustech.edu.cn

---


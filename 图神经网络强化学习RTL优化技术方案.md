
# CLAUDE.md - Project Context for Claude Code

You are an elite software engineer assisting an analog circuit engineer who has some machine learning knowledge. You will help them improve a reinforcement learning task framework and related engineering projects.

## Language and Format Requirements

**Language Rules:**
- Use Chinese for all interactions and discussions with the user
- Professional terms may be kept in English when appropriate
- All printed content in code should be in English
- Add appropriate Chinese and English comments in generated code

**Format Rules:**
- No emojis allowed in any files or code (except markdown files)
- Use clear, professional formatting

## Work Principles

You must follow these three-tier principles based on task complexity:

**Tier 1 - Simple Tasks:**
- When given a clear, single, simple task instruction, execute code modifications directly
- Provide code with appropriate comments

**Tier 2 - Complex Tasks:**
- When the task involves multiple components, is complex, or when a technical solution is proposed
- DO NOT generate code immediately
- Instead, discuss the specific technical approach with the user
- Refine the user's proposal and identify unclear technical details
- Generate a complete engineering plan only after clarification
- Your engineering plan must include:
  - Custom function names with input/output specifications
  - Data flow between different functions
  - File architecture updates if needed
- Use thinking/megathink/ultrathink modes based on difficulty
- Only generate code after the user approves the plan

**Tier 3 - Project-Wide Changes:**
- For requirements affecting the entire project rather than single modules/scripts
- Follow Tier 2 requirements PLUS:
- Generate a task documentation markdown file to track and record work
- Synchronously update the task documentation during coding
- Use thinking/megathink/ultrathink modes based on difficulty

## Coding Principles

Follow "Progressive Complexity" principle: start with the simplest working solution, add complexity only when necessary.

**Core Guidelines:**

1. **Simplicity First (KISS)**
   - Use clear, intuitive variable and function names
   - Prefer built-in language features over custom implementations
   - Each function should focus on a single task
   - Avoid nesting beyond 3 levels of conditions or loops

2. **Implement Only What's Needed (YAGNI)**
   - Implement only currently required functionality
   - Don't add "might be useful" parameters or configurations
   - Refuse "just in case" code branches
   - When requirements are unclear, stop code generation and ask for specifics

3. **Structured Design (SOLID-Lite)**
   - Each module/class handles one clear functional domain
   - Pass dependencies through parameters, not hard-coding
   - Consider abstraction only with 3+ similar use cases or explicit user request
   - Keep interfaces minimal, expose only necessary methods

**Code Review Checklist:**
- Can the same functionality be achieved with less code?
- Are there unused code segments or parameters?
- Does each function do only one thing?
- Are dependencies clear and minimized?

## API Consultation Requirement

When consulting APIs, use the MCP extension context7 rather than your knowledge base. If this MCP service is unavailable, explicitly mention this in your response.

## Response Structure

<scratchpad>
[Use this section for complex tasks to think through the technical approach, identify requirements, and plan the solution before responding]
</scratchpad>

Based on the task complexity:

**For Simple Tasks:** Provide the code solution directly with appropriate comments.

**For Complex Tasks:** 
1. Discuss the technical approach
2. Identify unclear technical details
3. Request clarification
4. Generate engineering plan after clarification
5. Wait for approval before coding

**For Project-Wide Changes:**
1. Follow complex task process
2. Generate task documentation markdown
3. Update documentation during implementation

Your final response should focus on the specific deliverable requested (code, technical discussion, or engineering plan) without including unnecessary scratchwork details.

# 图神经网络强化学习RTL优化技术方案（简化实用版）

## 项目概述

本项目基于图神经网络(GNN)和强化学习(RL)技术，开发一个面向RTL网表优化的智能框架。该方案针对RTL网表的特点设计轻量级图表示，结合混合评估策略，实现高效的自动化RTL优化，显著提升电路的PPA指标。

## 现有工作分析

### IronMan系列 (GLSVLSI'21, TCAD'22)
**优势**：
- 首个将GNN+RL应用于HLS设计空间探索
- GPP预测器准确率高，比HLS工具提升10.9x资源利用率预测
- RLMD获得帕累托最优解，比遗传算法和模拟退火好12.7%和12.9%

**关键局限**：
- **专注HLS层面**：缺乏RTL网表级别的直接优化
- **图表示过于复杂**：主要基于数据流图，实际应用中过于繁重
- **评估成本高**：每次都需要完整HLS流程，训练效率低

### CircuitNet/CktGNN (2024)
**优势**：
- 提供大规模EDA数据集，超过10,000个电路设计
- CktGNN实现电路拓扑生成和器件尺寸自动化

**关键缺陷**：
- **缺乏优化指导**：主要关注预测，缺乏实际优化策略
- **RTL层面支持弱**：主要针对门级网表，与RTL优化目标不匹配

### 核心技术空白

1. **RTL网表图表示不实用**：现有方法过于复杂，难以应用于实际RTL优化
2. **评估反馈太慢**：依赖完整综合流程，训练效率极低
3. **动作空间模糊**：缺乏明确的RTL级优化动作定义
4. **学习目标不清晰**：多目标优化的学习机制不明确

## 核心技术方案

### 阶段1：轻量级RTL网表图表示 (月1-2)

#### 1.1 简化图构建引擎

**设计理念**：针对RTL网表特点，构建轻量级但信息充分的图表示

```python
class RTLNetlistGraph:
    """面向RTL网表的轻量级图表示"""

    def __init__(self):
        # 简化的节点类型（4种核心类型）
        self.node_types = {
            "gate": "逻辑门节点",      # AND, OR, MUX, ADD等
            "reg": "寄存器节点",       # FF, Latch等
            "port": "端口节点",       # 输入/输出端口
            "wire": "连线节点"        # 内部连线
        }

        # 简化的边类型（3种核心关系）
        self.edge_types = {
            "connection": "连接关系", # 基本的信号连接
            "timing": "时序路径",     # 关键时序路径
            "fanout": "扇出关系"      # 信号扇出
        }

    def build_from_netlist(self, netlist_file):
        """从RTL网表文件构建图"""
        # 1. 解析网表（支持Verilog门级网表）
        netlist = self.parse_verilog_netlist(netlist_file)

        # 2. 创建节点（关注关键属性）
        nodes = self.create_nodes_with_features(netlist)

        # 3. 创建边（重点是数据依赖）
        edges = self.create_dependency_edges(netlist)

        # 4. 标记关键路径（用于优化决策）
        self.mark_critical_paths(nodes, edges)

        return self.build_pytorch_geometric_graph(nodes, edges)

    def create_nodes_with_features(self, netlist):
        """创建带有关键特征的节点"""
        nodes = []
        for component in netlist.components:
            if component.type == "gate":
                node = {
                    "id": component.id,
                    "type": "gate",
                    "features": {
                        "gate_type": self.encode_gate_type(component.gate_name),
                        "input_width": component.input_width,
                        "estimated_delay": self.lookup_gate_delay(component.gate_name),
                        "estimated_area": self.lookup_gate_area(component.gate_name),
                        "fanout_count": len(component.outputs),
                        "is_critical": False  # 后续标记
                    }
                }
            elif component.type == "register":
                node = {
                    "id": component.id,
                    "type": "reg",
                    "features": {
                        "bit_width": component.width,
                        "clock_domain": component.clock,
                        "has_reset": component.reset is not None,
                        "has_enable": component.enable is not None
                    }
                }
            nodes.append(node)
        return nodes
```

#### 1.2 高效图神经网络

**专门针对RTL网表优化的GNN架构**：
```python
class RTLOptimizationGNN(torch.nn.Module):
    """RTL优化专用图神经网络"""

    def __init__(self, node_feature_dim=32, hidden_dim=128):
        super().__init__()

        # 节点类型嵌入
        self.node_type_embedding = torch.nn.Embedding(4, 16)  # 4种节点类型

        # 特征编码器
        self.feature_encoder = torch.nn.Linear(node_feature_dim, hidden_dim)

        # 图卷积层（简化，只用2层）
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # 全局池化
        self.global_pool = GlobalMeanPool()

        # 输出头
        self.ppa_predictor = torch.nn.Linear(hidden_dim, 3)  # 预测PPA
        self.value_head = torch.nn.Linear(hidden_dim, 1)     # RL值函数

    def forward(self, graph):
        """前向传播"""
        # 节点特征编码
        x = self.feature_encoder(graph.x)

        # 图卷积
        x = F.relu(self.conv1(x, graph.edge_index))
        x = F.relu(self.conv2(x, graph.edge_index))

        # 图级嵌入
        graph_embedding = self.global_pool(x, graph.batch)

        # 多任务输出
        ppa_prediction = self.ppa_predictor(graph_embedding)
        state_value = self.value_head(graph_embedding)

        return {
            "graph_embedding": graph_embedding,
            "ppa_prediction": ppa_prediction,
            "state_value": state_value
        }
```

### 阶段2：明确的强化学习优化框架 (月3-4)

#### 2.1 实用的强化学习环境设计

**核心思路**：明确定义状态、动作、奖励，确保学习目标清晰

```python
class RTLOptimizationEnvironment(gym.Env):
    """RTL优化强化学习环境（简化实用版）"""

    def __init__(self, rtl_dataset, fast_evaluator):
        super().__init__()
        self.rtl_dataset = rtl_dataset
        self.fast_evaluator = fast_evaluator  # 混合评估器
        self.gnn_model = RTLOptimizationGNN()

        # 基于ABC的优化动作空间（12种核心优化操作）
        self.actions = {
            0: "rewrite",              # ABC: rewrite - AIG重写
            1: "refactor",             # ABC: refactor - 重构优化
            2: "balance",              # ABC: balance - 平衡AIG
            3: "resub",                # ABC: resub - 替换优化
            4: "compress2",            # ABC: compress2 - 压缩优化
            5: "choice",               # ABC: choice - 选择优化
            6: "fraig",                # ABC: fraig - 功能性归约
            7: "dch",                  # ABC: dch - 选择计算
            8: "if",                   # ABC: if - FPGA映射
            9: "mfs",                  # ABC: mfs - 最大扇入简化
            10: "lutpack",             # ABC: lutpack - LUT打包
            11: "no_operation"         # 跳过当前步骤
        }
        self.action_space = gym.spaces.Discrete(len(self.actions))

        # 状态空间：图嵌入（128维）+ PPA当前值（3维）+ 步数（1维）
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(132,), dtype=np.float32
        )

    def reset(self):
        """重置环境"""
        # 选择RTL设计
        self.current_netlist = random.choice(self.rtl_dataset)
        self.current_graph = self.build_graph(self.current_netlist)

        # 获取基线指标（用真实综合）
        self.baseline_ppa = self.fast_evaluator.full_evaluate(self.current_netlist)
        self.current_ppa = self.baseline_ppa.copy()

        # 重置计数器
        self.step_count = 0
        self.optimization_history = []

        return self._get_observation()

    def step(self, action):
        """执行优化动作"""
        self.step_count += 1

        # 解析动作
        action_type = self.actions[action]

        # 应用ABC优化动作
        if action_type != "no_operation":
            # 将图转换为网表
            current_netlist = self.graph_to_netlist(self.current_graph)

            # 应用ABC优化
            abc_result = self.fast_evaluator.abc_tool.apply_optimization(current_netlist, action_type)

            if abc_result["success"]:
                # 优化成功，更新图表示
                new_netlist = abc_result["optimized_netlist"]
                new_graph = self.netlist_to_graph(new_netlist)

                # 快速评估新的PPA
                new_ppa = self.fast_evaluator.evaluate(new_graph, self.step_count)

                # 计算奖励（包含ABC统计信息）
                reward = self._calculate_reward_with_abc_stats(
                    self.current_ppa, new_ppa, action_type, abc_result["stats"]
                )

                # 更新状态
                self.current_graph = new_graph
                self.current_ppa = new_ppa

            else:
                # ABC优化失败，惩罚
                reward = -0.2
                new_ppa = self.current_ppa
                print(f"ABC optimization failed: {abc_result['error']}")
        else:
            # 跳过操作
            reward = -0.05  # 轻微惩罚，鼓励采取行动
            new_ppa = self.current_ppa

        # 记录历史
        self.optimization_history.append({
            "action": action_type,
            "ppa": new_ppa,
            "reward": reward
        })

        # 判断结束
        done = (self.step_count >= 20) or self._early_termination()

        return self._get_observation(), reward, done, {"ppa": new_ppa}

    def _get_observation(self):
        """获取当前状态观测"""
        # 图嵌入
        gnn_output = self.gnn_model(self.current_graph)
        graph_embedding = gnn_output["graph_embedding"].flatten()

        # PPA归一化
        normalized_ppa = self._normalize_ppa(self.current_ppa, self.baseline_ppa)

        # 步数归一化
        normalized_step = self.step_count / 20.0

        # 拼接状态
        state = torch.cat([
            graph_embedding,
            torch.tensor(normalized_ppa, dtype=torch.float32),
            torch.tensor([normalized_step], dtype=torch.float32)
        ])

        return state.numpy()

    def _calculate_reward_with_abc_stats(self, old_ppa, new_ppa, action_type, abc_stats):
        """基于ABC优化统计的奖励函数"""
        # 1. PPA改善奖励（主要奖励）
        timing_improvement = (old_ppa["delay"] - new_ppa["delay"]) / old_ppa["delay"]
        area_improvement = (old_ppa["area"] - new_ppa["area"]) / old_ppa["area"]
        power_improvement = (old_ppa["power"] - new_ppa["power"]) / old_ppa["power"]

        # 加权PPA奖励
        ppa_reward = 0.5 * timing_improvement + 0.3 * area_improvement + 0.2 * power_improvement

        # 2. ABC级别的改善奖励（中间奖励，提升学习效率）
        abc_improvement = 0
        if abc_stats and "improvement" in abc_stats:
            nodes_reduction = abc_stats["improvement"]["nodes_reduction"]
            depth_reduction = abc_stats["improvement"]["depth_reduction"]

            # ABC层面的改善也给予奖励（权重较小，作为中间反馈）
            abc_improvement = 0.1 * nodes_reduction + 0.1 * depth_reduction

        # 3. 动作特异性奖励
        action_specific_reward = self._get_action_specific_reward(action_type, abc_stats)

        # 4. 约束违反惩罚
        constraint_penalty = 0
        if new_ppa.get("timing_violation", False):
            constraint_penalty = -1.0

        # 5. 逻辑等效性保证奖励（ABC保证了这一点）
        equivalence_bonus = 0.05

        total_reward = (ppa_reward + abc_improvement + action_specific_reward +
                       equivalence_bonus + constraint_penalty)

        return total_reward

    def _get_action_specific_reward(self, action_type, abc_stats):
        """针对不同ABC动作的特异性奖励"""
        action_rewards = {
            "rewrite": 0.02,      # 基础重写，小奖励
            "refactor": 0.03,     # 重构优化，稍高奖励
            "balance": 0.02,      # 平衡深度，小奖励
            "compress2": 0.04,    # 压缩优化，较高奖励
            "fraig": 0.03,        # 功能归约，中等奖励
            "if": 0.02,           # FPGA映射，小奖励
            "mfs": 0.04,          # 最大扇入简化，较高奖励
        }

        base_reward = action_rewards.get(action_type, 0.01)

        # 如果ABC统计显示显著改善，增加奖励
        if abc_stats and "improvement" in abc_stats:
            improvement = abc_stats["improvement"]
            if improvement["nodes_reduction"] > 0.1 or improvement["depth_reduction"] > 0.1:
                base_reward *= 1.5  # 50%额外奖励

        return base_reward
```

#### 2.2 简化的PPO训练算法

**专注于实用性的PPO实现**：
```python
class RTLOptimizationPPO:
    """RTL优化专用PPO算法"""

    def __init__(self, state_dim=132, action_dim=8):
        # 策略网络：简单的MLP（输入包含图嵌入）
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
            torch.nn.Softmax(dim=-1)
        )

        # 值函数网络
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)

        # PPO参数
        self.clip_ratio = 0.2
        self.entropy_coef = 0.01

    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_net(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        return {
            "action": action.item(),
            "log_prob": action_dist.log_prob(action).item(),
            "value": self.value_net(state_tensor).item()
        }

    def update(self, batch_data):
        """更新策略和值函数"""
        states = torch.FloatTensor(batch_data["states"])
        actions = torch.LongTensor(batch_data["actions"])
        old_log_probs = torch.FloatTensor(batch_data["log_probs"])
        returns = torch.FloatTensor(batch_data["returns"])
        advantages = torch.FloatTensor(batch_data["advantages"])

        # 计算新的动作概率和值
        action_probs = self.policy_net(states)
        action_dist = torch.distributions.Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)
        values = self.value_net(states).squeeze()

        # PPO策略损失
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 熵损失（鼓励探索）
        entropy_loss = -action_dist.entropy().mean()

        # 值函数损失
        value_loss = F.mse_loss(values, returns)

        # 总损失
        total_policy_loss = policy_loss + self.entropy_coef * entropy_loss

        # 更新网络
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item()
        }

#### 2.3 ABC优化动作执行引擎

**基于ABC的逻辑等效优化**：
```python
class ABCInterface:
    """ABC工具接口，执行具体的优化动作"""

    def __init__(self):
        # ABC命令映射
        self.abc_commands = {
            "rewrite": "rewrite -l",           # AIG重写，保持逻辑等效
            "refactor": "refactor -l",         # 重构优化
            "balance": "balance -l",           # 平衡AIG深度
            "resub": "resub -l",               # 替换优化
            "compress2": "compress2",          # 综合压缩优化
            "choice": "choice",                # 选择计算优化
            "fraig": "fraig",                  # 功能性归约
            "dch": "dch",                      # 深度选择计算
            "if": "if -K 6",                   # FPGA技术映射
            "mfs": "mfs",                      # 最大扇入简化
            "lutpack": "lutpack"               # LUT打包优化
        }

        # 优化序列组合（常用的多步优化）
        self.combo_sequences = {
            "light_opt": ["rewrite", "refactor"],
            "heavy_opt": ["rewrite", "refactor", "balance", "resub"],
            "fpga_opt": ["rewrite", "balance", "if", "lutpack"],
            "area_opt": ["compress2", "choice", "mfs"]
        }

    def apply_optimization(self, rtl_netlist, action_name):
        """应用ABC优化动作"""
        try:
            # 1. 将RTL转换为AIG格式
            aig_file = self.convert_rtl_to_aig(rtl_netlist)

            # 2. 构建ABC脚本
            abc_script = self.build_abc_script(action_name)

            # 3. 执行ABC优化
            optimized_aig = self.run_abc_optimization(aig_file, abc_script)

            # 4. 转换回RTL网表
            optimized_netlist = self.convert_aig_to_rtl(optimized_aig)

            # 5. 验证逻辑等效性
            equivalence_check = self.verify_equivalence(rtl_netlist, optimized_netlist)

            if equivalence_check:
                return {
                    "success": True,
                    "optimized_netlist": optimized_netlist,
                    "optimization": action_name,
                    "stats": self.get_optimization_stats(aig_file, optimized_aig)
                }
            else:
                return {
                    "success": False,
                    "error": "Equivalence check failed",
                    "original_netlist": rtl_netlist
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "original_netlist": rtl_netlist
            }

    def build_abc_script(self, action_name):
        """构建ABC优化脚本"""
        if action_name in self.abc_commands:
            # 单个命令
            command = self.abc_commands[action_name]
            script = f"""
            read_aig input.aig
            {command}
            write_aig output.aig
            print_stats
            """
        elif action_name in self.combo_sequences:
            # 组合优化序列
            commands = "; ".join([self.abc_commands[cmd] for cmd in self.combo_sequences[action_name]])
            script = f"""
            read_aig input.aig
            {commands}
            write_aig output.aig
            print_stats
            """
        else:
            # 默认：无操作
            script = f"""
            read_aig input.aig
            write_aig output.aig
            """

        return script

    def run_abc_optimization(self, input_aig, abc_script):
        """运行ABC优化"""
        import subprocess
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            # 写入脚本文件
            script_file = os.path.join(temp_dir, "optimize.abc")
            with open(script_file, 'w') as f:
                f.write(abc_script)

            # 复制输入文件
            input_file = os.path.join(temp_dir, "input.aig")
            output_file = os.path.join(temp_dir, "output.aig")

            subprocess.run(["cp", input_aig, input_file], check=True)

            # 运行ABC
            result = subprocess.run(
                ["abc", "-f", script_file],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=True
            )

            # 读取优化结果
            if os.path.exists(output_file):
                return output_file
            else:
                raise Exception("ABC optimization failed to produce output")

    def verify_equivalence(self, original_netlist, optimized_netlist):
        """验证逻辑等效性"""
        # 使用ABC的等效性检查
        try:
            # 转换为AIG
            original_aig = self.convert_rtl_to_aig(original_netlist)
            optimized_aig = self.convert_rtl_to_aig(optimized_netlist)

            # ABC等效性检查脚本
            equiv_script = f"""
            read_aig {original_aig}
            &r {optimized_aig}
            &equiv
            """

            # 运行等效性检查
            result = subprocess.run(
                ["abc", "-c", equiv_script],
                capture_output=True,
                text=True
            )

            # 解析结果
            return "Networks are equivalent" in result.stdout

        except Exception as e:
            print(f"Equivalence check failed: {e}")
            return False

    def get_optimization_stats(self, original_aig, optimized_aig):
        """获取优化统计信息"""
        def get_aig_stats(aig_file):
            result = subprocess.run(
                ["abc", "-c", f"read_aig {aig_file}; print_stats"],
                capture_output=True,
                text=True
            )
            # 解析统计信息（节点数、深度等）
            return self.parse_abc_stats(result.stdout)

        original_stats = get_aig_stats(original_aig)
        optimized_stats = get_aig_stats(optimized_aig)

        return {
            "original": original_stats,
            "optimized": optimized_stats,
            "improvement": {
                "nodes_reduction": (original_stats["nodes"] - optimized_stats["nodes"]) / original_stats["nodes"],
                "depth_reduction": (original_stats["depth"] - optimized_stats["depth"]) / original_stats["depth"]
            }
        }

    def parse_abc_stats(self, abc_output):
        """解析ABC统计输出"""
        import re

        # 提取关键统计信息
        nodes_match = re.search(r'Nodes:\s*(\d+)', abc_output)
        depth_match = re.search(r'Depth:\s*(\d+)', abc_output)

        return {
            "nodes": int(nodes_match.group(1)) if nodes_match else 0,
            "depth": int(depth_match.group(1)) if depth_match else 0
        }
```


### 阶段3：混合评估系统 (月5-6)

#### 3.1 快速反馈评估系统（核心创新）

**解决反馈速度问题的关键技术**：
```python
class HybridEvaluationSystem:
    """混合评估系统：结合GNN预测和真实综合"""

    def __init__(self):
        # 预训练的PPA预测器
        self.ppa_predictor = self._build_ppa_predictor()

        # ABC工具接口
        self.abc_tool = ABCInterface()
        self.synthesis_tool = YosysInterface()  # 用于最终PPA评估

        # 评估策略
        self.full_eval_interval = 10  # 每10步做一次真实综合
        self.prediction_confidence_threshold = 0.8

    def _build_ppa_predictor(self):
        """构建PPA预测网络"""
        return torch.nn.Sequential(
            torch.nn.Linear(128, 256),  # 输入是图嵌入
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),    # 输出PPA
            torch.nn.Sigmoid()          # 归一化到[0,1]
        )

    def quick_evaluate(self, rtl_graph):
        """快速评估（基于GNN预测）"""
        # 获取图嵌入
        gnn_output = self.gnn_model(rtl_graph)
        graph_embedding = gnn_output["graph_embedding"]

        # 预测PPA
        predicted_ppa = self.ppa_predictor(graph_embedding)
        confidence = gnn_output.get("confidence", 0.7)

        return {
            "delay": predicted_ppa[0].item(),
            "area": predicted_ppa[1].item(),
            "power": predicted_ppa[2].item(),
            "confidence": confidence,
            "method": "prediction"
        }

    def full_evaluate(self, rtl_netlist):
        """完整评估（真实综合）"""
        # 运行Yosys综合
        synthesis_result = self.synthesis_tool.synthesize(rtl_netlist)

        return {
            "delay": synthesis_result["delay"],
            "area": synthesis_result["area"],
            "power": synthesis_result["power"],
            "confidence": 1.0,
            "method": "synthesis",
            "timing_violation": synthesis_result["timing_violations"] > 0
        }

    def evaluate(self, rtl_graph, step_count):
        """混合评估策略"""
        # 决策：用预测还是真实综合
        if (step_count % self.full_eval_interval == 0) or \
           (self._need_verification(rtl_graph)):
            # 用真实综合
            rtl_netlist = self.graph_to_netlist(rtl_graph)
            result = self.full_evaluate(rtl_netlist)

            # 用结果校正预测器
            self._calibrate_predictor(rtl_graph, result)

            return result
        else:
            # 用快速预测
            return self.quick_evaluate(rtl_graph)

    def _need_verification(self, rtl_graph):
        """判断是否需要验证"""
        # 预测置信度低时需要验证
        quick_result = self.quick_evaluate(rtl_graph)
        return quick_result["confidence"] < self.prediction_confidence_threshold

    def _calibrate_predictor(self, rtl_graph, ground_truth):
        """校正预测器"""
        # 用真实结果微调预测器
        gnn_output = self.gnn_model(rtl_graph)
        graph_embedding = gnn_output["graph_embedding"]

        predicted_ppa = self.ppa_predictor(graph_embedding)
        target_ppa = torch.tensor([
            ground_truth["delay"],
            ground_truth["area"],
            ground_truth["power"]
        ])

        # 计算损失并更新
        loss = F.mse_loss(predicted_ppa, target_ppa)

        optimizer = torch.optim.Adam(self.ppa_predictor.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 阶段4：实验验证与评估 (月7-8)

#### 4.1 实验设计框架

**务实的实验设计**：
```python
class RTLOptimizationExperiment:
    """RTL优化实验设计"""

    def __init__(self):
        self.test_datasets = {
            "rtl_rewriter_bench": "你的RTLRewriter-Bench（55个短案例+5个长案例）",
            "circuitnet_subset": "CircuitNet数据集子集",
            "synthetic_designs": "合成的简单测试用例"
        }

        self.baseline_methods = {
            "ironman_pro": "IronMan-Pro基线",
            "manual_optimization": "人工优化",
            "yosys_default": "Yosys默认优化",
            "random_search": "随机搜索"
        }

    def run_experiments(self, our_method):
        """运行实验"""
        results = {}

        # 1. 基础性能对比
        results["basic_comparison"] = self.compare_with_baselines(our_method)

        # 2. 消融研究
        results["ablation_study"] = self.conduct_ablation_study(our_method)

        # 3. 可扩展性测试
        results["scalability"] = self.test_scalability(our_method)

        return results

    def compare_with_baselines(self, our_method):
        """与基线方法对比"""
        comparison_results = {}

        for dataset_name, dataset in self.test_datasets.items():
            dataset_results = {}

            for baseline_name, baseline in self.baseline_methods.items():
                wins, losses, ties = 0, 0, 0

                for test_case in dataset:
                    our_result = our_method.optimize(test_case)
                    baseline_result = baseline.optimize(test_case)

                    if self.is_better(our_result, baseline_result):
                        wins += 1
                    elif self.is_better(baseline_result, our_result):
                        losses += 1
                    else:
                        ties += 1

                dataset_results[baseline_name] = {
                    "wins": wins, "losses": losses, "ties": ties,
                    "win_rate": wins / (wins + losses + ties)
                }

            comparison_results[dataset_name] = dataset_results

        return comparison_results

    def conduct_ablation_study(self, method):
        """消融研究"""
        ablation_configs = {
            "no_gnn": "用简单MLP替代GNN",
            "no_hybrid_eval": "只用真实综合，不用预测",
            "simple_reward": "简化奖励函数",
            "random_actions": "随机动作选择"
        }

        ablation_results = {}
        for config_name, config_desc in ablation_configs.items():
            modified_method = self.create_ablated_method(method, config_name)
            performance = self.evaluate_method(modified_method)
            ablation_results[config_name] = {
                "description": config_desc,
                "performance": performance
            }

        return ablation_results
```

#### 4.2 简化的基线对比

**重点对比的基线方法**：
```python
class BaselineComparison:
    """实用的基线对比"""

    def __init__(self):
        self.baselines = {
            "ironman_pro": self.setup_ironman_baseline(),
            "yosys_optimization": self.setup_yosys_baseline(),
            "manual_expert": self.setup_manual_baseline()
        }

    def setup_ironman_baseline(self):
        """设置IronMan基线"""
        # 基于IronMan开源代码实现
        return IronManBaseline(
            config_file="ironman_default.yaml",
            model_path="pretrained_ironman.pth"
        )

    def run_comparison(self, our_method, test_cases):
        """运行对比实验"""
        results = {}

        for baseline_name, baseline in self.baselines.items():
            print(f"对比 {baseline_name}...")

            baseline_results = []
            our_results = []

            for test_case in test_cases:
                # 运行基线
                baseline_ppa = baseline.optimize(test_case)
                baseline_results.append(baseline_ppa)

                # 运行我们的方法
                our_ppa = our_method.optimize(test_case)
                our_results.append(our_ppa)

            # 计算改善百分比
            improvements = self.calculate_improvements(our_results, baseline_results)

            results[baseline_name] = {
                "timing_improvement": np.mean(improvements["timing"]),
                "area_improvement": np.mean(improvements["area"]),
                "power_improvement": np.mean(improvements["power"]),
                "win_rate": np.mean(improvements["overall"] > 0)
            }

        return results
```

## 开源仓库与实现资源

### 核心仓库：
1. **IronMan** (https://github.com/lydiawunan/IronMan) - 基础GNN+RL框架
2. **CircuitNet** (https://github.com/circuitnet/CircuitNet) - EDA数据集
3. **你的RTLRewriter-Bench** - 评测基准

### 技术栈：
- **ABC** - 核心逻辑优化引擎（保证逻辑等效性）
- **PyTorch Geometric** - 图神经网络框架
- **Stable-Baselines3** - 强化学习算法库
- **Yosys** - 开源综合工具（PPA评估）
- **NetworkX** - 图处理和分析

## 详细实施计划

### 第1-2个月：建立基础框架
**目标**：实现基本的RTL图表示和GNN模型

**具体任务**：
1. **Week 1-2**: 熟悉IronMan代码，理解其GNN+RL架构
2. **Week 3-4**: 基于RTLRewriter-Bench数据实现RTL图构建
3. **Week 5-6**: 实现简化的RTL图神经网络
4. **Week 7-8**: 建立基础的综合工具接口（Yosys）

**交付物**：
- RTL网表解析器
- 基础图神经网络模型
- Yosys接口模块

### 第3-4个月：强化学习环境
**目标**：实现RL环境和训练算法

**具体任务**：
1. **Week 9-10**: 设计RL环境，定义状态/动作/奖励
2. **Week 11-12**: 实现PPO训练算法
3. **Week 13-14**: 开发混合评估系统（关键创新）
4. **Week 15-16**: 初步训练和调试

**交付物**：
- RL训练环境
- 混合评估系统
- 初步训练的模型

### 第5-6个月：系统集成和优化
**目标**：完善系统，提升性能

**具体任务**：
1. **Week 17-18**: 优化训练稳定性
2. **Week 19-20**: 实现在线学习机制
3. **Week 21-22**: 系统集成和端到端测试
4. **Week 23-24**: 性能调优

**交付物**：
- 完整的优化系统
- 性能调优报告

### 第7-8个月：实验验证和论文
**目标**：实验验证，撰写论文

**具体任务**：
1. **Week 25-26**: 大规模实验验证
2. **Week 27-28**: 与基线方法对比
3. **Week 29-30**: 论文写作
4. **Week 31-32**: 投稿和开源发布

**交付物**：
- 实验结果报告
- 会议论文
- 开源代码

## 预期成果与创新点

### 技术成果
- **轻量级RTL图表示**：专门针对RTL网表设计，实用性强
- **混合评估系统**：解决训练效率问题的关键创新
- **实用的RL框架**：明确的状态/动作/奖励设计

### 学术贡献
- **一作论文**：DAC/ICCAD 2025投稿目标
- **开源项目**：完整的RTL优化框架
- **基准扩展**：增强RTLRewriter-Bench的评估能力

### 核心创新点
1. **ABC-驱动的RL动作空间**：使用成熟的ABC优化命令作为动作，保证逻辑等效性
2. **混合评估策略**：结合GNN预测和真实综合，训练效率提升10x
3. **多层奖励反馈**：结合PPA改善和ABC中间统计，提供更丰富的学习信号
4. **RTL网表专用图表示**：轻量级但信息充分的图结构设计

这个简化的方案更加实用，充分利用了你的GNN和RL专长，同时避免了过度复杂的设计。重点是**混合评估系统**这一核心创新，能够解决现有方法训练效率低的根本问题。


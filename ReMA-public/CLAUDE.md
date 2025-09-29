
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

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReMA (Reinforced Multi-Agents) is an open-source framework for Multi-Agent Multi-Turn Reinforcement Learning designed to train LLMs with meta-thinking capabilities. The framework implements hierarchical agents: a high-level meta-thinking agent for strategic oversight and a low-level reasoning agent for detailed execution.

## Repository Structure

```
├── src/
│   ├── verl/                    # Core RL framework
│   │   ├── verl/
│   │   │   ├── rema_trainer/           # Shared parameter training
│   │   │   ├── rema_separated_trainer/ # Separate parameter training
│   │   │   ├── models/                 # Model implementations
│   │   │   ├── utils/                  # Utility functions
│   │   │   └── tests/                  # Test suite
│   │   └── pyproject.toml
│   └── 360-LLaMA-Factory/       # SFT (Supervised Fine-Tuning) component
├── scripts/
│   ├── rl/rema/                 # Training scripts for shared parameters
│   ├── rl/separated_rema/       # Training scripts for separate parameters
│   └── eval/                    # Evaluation scripts
├── data/                        # Training and evaluation datasets
│   ├── MATH/                    # Math problem datasets
│   ├── multi_turn_mamrp/        # Multi-turn datasets
│   └── overall_math/            # Math evaluation sets
└── requirements.txt
```

## Installation Commands

```bash
# Create and activate conda environment
conda create -n rema python=3.10
conda activate rema

# Install flash attention for faster training
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Install SFT component
cd src/360-LLaMA-Factory
pip install -e .

# Install RL component
cd ../verl
pip install -e .

# Install additional requirements
cd ../..
pip install -r requirements.txt
```

## Common Development Commands

### Training
```bash
# Run ReMA training with shared parameters
bash scripts/rl/rema/example_7b.sh

# Run ReMA training with separated parameters
bash scripts/rl/separated_rema/example_7b.sh

# Evaluation only (add to training script)
+trainer.val_before_train=True \
+trainer.val_only=True \
```

### Testing and Quality Checks
```bash
# Run tests (for 360-LLaMA-Factory component)
cd src/360-LLaMA-Factory
make test

# Code quality checks
make quality

# Auto-format code
make style

# Manual commands
ruff check . --fix
ruff format .
pytest -vv tests/
```

## Training Modes

### ReMA with Shared Parameters
- Both agents share the same parameters
- End-to-end training with unified learning
- Configuration: `src/verl/verl/rema_trainer/config/ppo_trainer.yaml`

### ReMA with Separate Parameters
- High-level and low-level agents have separate parameters
- One agent frozen while other is trainable
- Configuration: `src/verl/verl/rema_separated_trainer/config/ppo_trainer.yaml`
- Enable with: `algorithm.switch_agent.enable=True`

## Key Configuration Parameters

### Training Script Variables (must be set)
- `WORKSPACE`: Your workspace directory
- `PROJECT_NAME`: W&B project name
- `EXPERIMENT_NAME`: Experiment identifier
- `MODEL_PATH`: Path to base model

### Critical Training Parameters
- `data.max_prompt_length`: Max prompt length per turn
- `data.max_response_length`: Max response length per turn
- `actor_rollout_ref.rollout.max_num_turns`: Maximum conversation turns
- `actor_rollout_ref.actor.clip_mode`: `"turn"` or `"batch"` ratio clipping
- `actor_rollout_ref.actor.agg_mode`: `"trajectory"`, `"batch"`, or `"turn"`
- `algorithm.switch_agent.freq`: Agent switching frequency (separated mode)

## Architecture Notes

The framework consists of two main components:
1. **360-LLaMA-Factory**: Handles supervised fine-tuning and uses ruff for linting
2. **verl**: Core RL training engine with PPO implementation and multi-agent support

The training pipeline supports:
- Multi-turn conversations with up to 30 turns
- Dynamic batch sizing for memory efficiency
- Turn-level ratio clipping (more efficient than token-level)
- Reward masking for unfinished sequences
- Integration with W&B for experiment tracking

## Data Format

Training data should be in parquet format with required fields:
- `question`: The prompt/question field
- Response generation handled by the RL pipeline
- Evaluation datasets processed as complete batches by inference engines
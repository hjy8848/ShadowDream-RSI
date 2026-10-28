# ShadowDream-RSI

ShadowDream-RSI 研究条件能力评测、策略程序进化搜索和执行模型学习。

## 当前实现

- ALFWorld 完整任务运行与 witness 前缀重放由本项目自己的环境适配和 Docker policy worker 执行。checkpoint 接管支持空策略状态与最近五条公开历史的 `cold-start`，也支持完整公开历史的 `history` 协议；候选看不到环境事实或 witness 后缀。
- 所有通过完整运行的候选与父节点做配对 checkpoint 评测，记录各点变化、条件能力增益及退化。
- P-only Editor（直接输出 Python，也接受既有 JSON 格式）、合法性检查、全 Archive 调度、父子存档及 Greedy、Random、UCB、Progress-UCB、Shadow-UCB。搜索反馈可选择绝对完整表现或相对扩展前历史最优的非负提升；每次扩展记录两者。
- 从同一父节点的多次修改构造 Editor GRPO 样本，并通过本项目的 VeRL V1 入口训练；GPU 更新尚未验证。
- 从真实成功轨迹构造 Service 动作 SFT 数据，并使用基于 Transformers/PEFT 的 LoRA SFT 训练入口；可单独生成直接 witness SFT 数据作为对照。Service 学习不依赖搜索收益显著。
- 在按任务 ID 和场景隔离的 Selection 集重评全部存档策略，并可用旧模型作为保留候选；统一 direct-action 接口用于衡量 Service 权重。
- 提供单次在线 Service GRPO 更新：用相同的合法动作 token 约束采样及重算 log probability；采样和训练使用 Transformers/PEFT。
- 提供搜索、可选模型更新、独立 Selection 和跨周期状态的编排入口；完整训练闭环尚未验证。
- 提供按已完成评测统计的成本对照及多搜索 seed 的配对 bootstrap 区间，并核对任务、根策略、代码和模型指纹；可用 `search.env_step_budget` 设置环境步数硬上限，未完成的候选不进入存档。
- 本地记录模型输出、候选策略、完整与 checkpoint 轨迹、搜索反馈、环境步数和模型 token。探索预算可限制 Editor 调用和环境步数，其他成本用于报告。

项目仍在开发：Editor 的 VeRL GPU 更新、完整预算匹配实验、历史接管协议和跨周期训练闭环仍需要验证；当前代码不能证明研究假设。SFT 已在本地小模型和真实 witness 上完成一次参数更新验证。

## 依赖和设置

本项目不依赖 Remedy。需要 Python 3.11、ALFWorld/TextWorld 数据、Docker 和策略 worker 使用的镜像。安装基础依赖；Editor VeRL GRPO 还需要兼容的 CUDA 环境和 `verl` extra：

```bash
python -m pip install -e .
python -m pip install -e '.[verl]'  # 仅用于 Editor GRPO
export ALFWORLD_DATA=/path/to/alfworld
```

复制 `configs/example.json` 并填写模型路径、revision 和环境配置。ALFWorld 配置随 Python 包提供；也可在 `alfworld.config_path` 指定自己的 YAML。任务文件是 JSONL，每行包含 task ID、游戏文件、场景 ID、任务说明、goal atoms、witness actions 和 source split；checkpoint 文件为 `{"task-id": [2, 5]}`，prefix 长度必须小于 witness 长度和步数预算。固定 `seeds`，提前划分训练、验证和测试任务；不同实验不应复用输出目录。

```bash
shadowdream shadow --config configs/run.json --tasks tasks.jsonl \
  --checkpoints checkpoints.json --parent parent.json --child child.json --out shadow.json
shadowdream search --config configs/run.json --tasks tasks.jsonl \
  --checkpoints checkpoints.json --parent parent.json --out runs/shadow \
  --attempts 10 --method shadow_ucb --feedback improvement
shadowdream editor-grpo --config configs/run.json --run-dir runs/shadow --out runs/editor-update
shadowdream select --config configs/run.json --run-dir runs/shadow --tasks selection.jsonl --out runs/selected.json
shadowdream test --config configs/run.json --selection runs/selected.json --tasks heldout.jsonl --out runs/test.json --action-mode direct
shadowdream report --run-dir runs/shadow --metric env_steps --budget 1000 --out runs/report.json
# 多 seed 比较：重复 --run-dir，指定基线和目标的 method / feedback。
shadowdream cycle --config configs/run.json --parent parent.json --tasks tasks.jsonl \
  --selection-tasks selection.jsonl --checkpoints checkpoints.json --out runs/cycle1 --attempts 10
shadowdream cycle --state runs/cycle1/state.json --tasks tasks.jsonl \
  --selection-tasks selection.jsonl --checkpoints checkpoints.json --out runs/cycle2 --attempts 10
shadowdream service-data --run-dir runs/shadow --out runs/service-sft.jsonl
shadowdream witness-data --config configs/run.json --tasks tasks.jsonl --out runs/witness-sft.jsonl
shadowdream service-sft --config configs/run.json --dataset runs/service-sft.jsonl \
  --out runs/service-adapter --device cuda:0
shadowdream service-grpo --config configs/grpo.json --tasks train.jsonl --out runs/service-grpo --device cuda:0
```

`absolute` 和 `improvement` 必须作为独立运行进行比较。搜索样本不可替代独立 Selection 或最终测试；运行数据目录不得混入测试集。GPU 时间需要通过实际训练/运行环境另行计量。

测试：`PYTHONPATH=src python -m unittest discover -s tests`。

准备 SFT 数据时，`service-data` 和 `witness-data` 均可传入 `--token-budget`；前者还需提供 `--config`。系统按相同 tokenizer 和完整样本计算输入 token，实际使用量可能低于上限，并写入训练指标。

Service GRPO 要求训练配置使用正温度、`top_p=1` 和至少两个不同的 rollout seed。每次更新会检查采样与重算的受约束 token log probability；默认容差 0.001，可用 `training.service_logprob_tolerance` 明确调整并记录。奖励无组内差异时梯度可能为零。可设置 `training.service_env_step_budget` 限制在线交互，预算截断的不完整任务组不会用于训练，已消耗步数仍计入成本。

Editor GRPO 使用 `top_p=1` 且不使用重复惩罚；训练前重算生成 token 的未截断 log probability 并与采样记录核对；默认容差为 0.001，可通过 `training.editor_logprob_tolerance` 显式调整，训练结果会保存误差和容差。半精度硬件可能需要根据实测数值差异设置容差。`shadow.protocol` 可设为 `cold-start` 或 `history`，两种协议不可混作同一项比较。

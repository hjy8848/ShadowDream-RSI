# ShadowDream-RSI v0.1 真实运行验证报告

**验证日期：** 2026-10-28（Asia/Shanghai）
**目标版本：** `main` / `333b5d4a53f56ec244fec895072d20e3e4d18aff`
**代码指纹：** `61a9feba049860b9c8f59f5086bf7e4f8bd28c9b500738d7c13247becbe7f9dc`
**验证主机：** `zju-47`，10.130.138.47
**总体状态：** 用户要求停止验证后已停止；未完成项目明确标记为未验证。远端本次验证进程、Ray worker 与容器均无残留；验证时选用的 GPU 3、6 已释放。

## 验证分类与结论

| 类别 | 状态 | 证据摘要 |
|---|---|---|
| 代码单元测试 | 已通过 | 33 项测试分别在独立依赖环境与 `.venv-verl-cu128` Remedy 路径隔离环境通过。 |
| 独立依赖安装与 Remedy 隔离 | 已验证（有环境说明） | 项目核心依赖以 editable 方式安装成功；运行进程中 `experiment_rsi` 不可导入且 `sys.path` 不含 Remedy 源码。独立 venv 的 `pip check` 仍报告其继承的 `searchr1` 基础环境中两个无关包元数据冲突；指定 `.venv-verl-cu128` 的 `pip check` 通过。 |
| ALFWorld / Shadow 真实集成 | 部分已验证 | 完整真实 episode、witness 重放和父子 checkpoint 评测已运行。Shadow 的成功率配对差符合 v2 定义；本次父子都未成功，因此 Shadow gain 为 0。`progress=0.0` 与 `0.5` 是额外观察到的稠密进展差异，不能单独当作 Shadow 成功增益。 |
| Shadow-UCB + 3B Editor / 1.5B Service 搜索 | 未通过；后续验证已停止 | 首次尝试真实加载模型并跑完 root episode；Editor 收到 183,248-token prompt（超出模型 32,768 上限），首个候选解析失败，进化树只有 root。没有重跑。 |
| Editor GRPO / Service GRPO GPU 更新 | 未验证 | 首次 Cycle 在 Editor 生成候选阶段停止，未进入任何 GRPO 更新。 |
| 连续两轮 Cycle 状态继承 | 未验证 | 未完成 Cycle 1，未启动 Cycle 2。 |
| 研究效果实验 | 未验证 | 本次工程验证不是 Shadow-UCB 相对普通 UCB 的效果比较，也不据此推断算法有效性。 |

## 硬件、模型与运行环境

- GPU：8× NVIDIA RTX A6000，单卡 48 GiB。验证任务在独占的物理 GPU 6（Editor）与 GPU 3（Service）上运行；其它卡上存在其他用户任务，未触碰。
- Python：`/home/hejingyu/projects/Remedy/.venv-verl-cu128/bin/python`，Python 3.11.14；PyTorch 2.8.0+cu128、Transformers 4.55.2、PEFT 0.21.2、VeRL `0.9.0.dev0`。
- 模型：`/home/shenyl/hf/model/Qwen/Qwen2.5-3B-Instruct`（Editor）；`/home/shenyl/hf/model/Qwen/Qwen2.5-1.5B-Instruct`（Service）。
- ALFWorld 数据：`/home/hejingyu/.cache/alfworld/alfworld`；环境由项目配置的 `python:3.11-slim` Docker worker 执行。
- Remedy 隔离：通过临时 `sitecustomize.py` 过滤 editable install 注入的 Remedy 源码路径。验证命令输出为 `experiment_rsi None`、`remedy_path []`。没有修改原 `.venv-verl-cu128`。
- GPU BF16 矩阵乘 smoke test 在两张 A6000 上成功；日志：`/data/zju-47/hejingyu/shadowdream-validation-20261028-333b5d4/logs/cuda-device-smoke.log`。

## 单元测试

```text
33 tests: OK
```

- 独立 venv：`/data/zju-47/hejingyu/shadowdream-validation-20261028-333b5d4/logs/unit-tests-independent-venv.log`
- `.venv-verl-cu128`（Remedy 路径隔离）：`/data/zju-47/hejingyu/shadowdream-validation-20261028-333b5d4/logs/unit-tests-verl-env.log`
- 此分类只表示代码单元测试通过，不等同于真实 ALFWorld、模型推理或 GPU 训练通过。

## 独立依赖与 Remedy 运行时隔离

代码从目标 Git SHA 归档到远端验证目录后，在独立验证目录中执行 `pip install -e <验证目录>/project`，安装日志：`/data/zju-47/hejingyu/shadowdream-validation-20261028-333b5d4/logs/core-dependencies.log`。项目未声明 Remedy 运行依赖；输入任务文件也未包含 Remedy 数据/导入引用。

独立 venv 使用 `searchr1` 的 `--system-site-packages`，其 `pip check` 报告了继承环境已有的两项版本冲突（`torchaudio` 对 `torch==2.8.0` 的约束、旧 `verl 0.1` 对 `transformers<4.48` 的约束；实际继承的 torch/Transformers 版本分别为 2.4.0/4.57.3）。这些包不由项目安装引入。完整输出见 `logs/independent-venv-pip-check.log`。指定训练环境 `.venv-verl-cu128` 的 `pip check` 通过，见 `logs/slow-preflight.log`。

模型运行使用用户提供的 `.venv-verl-cu128`，并在每次进程中检查 `importlib.util.find_spec('experiment_rsi') is None` 且 `sys.path` 不包含 `Remedy/src`。因此这里复用的是训练依赖环境，不是 Remedy 代码运行时依赖。

## 真实 ALFWorld 与 Shadow 集成验证

### Witness 重放

四个实际 ALFWorld/PDDL witness 在项目 `AlfworldEpisode` 与原生游戏环境中重放，均达到 `won=True`、`goal_reached=True`、`progress=1.0`。日志：`/data/zju-47/hejingyu/shadowdream-validation-20261028-333b5d4/logs/witness-preflight.log`。

### 完整 Service episode

- 任务：`d6c4d664c10d962d`，seed 7；Qwen2.5-1.5B-Instruct，物理 GPU 5。
- 真实模型生成 16 个动作，执行到 `max_steps=16`；`candidate_error=null`，`success=false`，`progress=0.0`，16 次模型调用、5933 个模型 token。
- 这是模型在该任务上的负向表现，但 episode 与模型推理完整运行，不是运行时失败。
- 日志：`logs/real-direct-episode.log`；完整轨迹：`runs/direct-episode.json`。

### Witness checkpoint 父子 Shadow 配对评测

- 真实 ALFWorld 任务 `d6c4d664c10d962d`、checkpoint prefix 长度 2、seeds 7/8；分别运行 parent 和 child，共 4 个真实 episode。
- 四个 episode 均完成剩余 14 步，`candidate_error=null`。Parent 两个 seed 的 `success=false, progress=0.0`；child 两个 seed 的 `success=false, progress=0.5`。
- 当前输出聚合结果为 `shadow.vector=[(task, 2, 0.0)]`、`gain=0.0`、`regression=0.0`。这是符合 v2 中 `C(S,T,k;ξ)` 成功指示配对差的结果：parent 与 child 都未完成任务，故 Shadow success delta 为零。两个 child episode 的 `progress=0.5` 相比 parent 的 `0.0` 是稠密进展指标差异，留在 episode 记录中，应作为 `task_progress` 对照解释；本次样本不足以声称条件完成能力提升。
- 代码路径会对每一个语法有效、完整任务执行无候选错误的候选运行配对 Shadow，不以完整任务成绩过滤，因此完整任务直接有效的修改也会评测并记录 Shadow。每个 task/checkpoint 的均值向量、正增益/退化汇总以及 parent/child 每 seed episode 均写入输出。
- 命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 \
PYTHONNOUSERSITE=1 ALFWORLD_DATA=/home/hejingyu/.cache/alfworld/alfworld \
PYTHONPATH="$ROOT/runtime-isolation:$ROOT/project/src" \
/home/hejingyu/projects/Remedy/.venv-verl-cu128/bin/python -m shadowdream.cli shadow \
  --config "$ROOT/configs/run-single.json" \
  --tasks "$ROOT/inputs/scene304.jsonl" \
  --checkpoints "$ROOT/inputs/scene304-checkpoints.json" \
  --parent "$ROOT/inputs/parent.json" --child "$ROOT/inputs/child.json" \
  --out "$ROOT/runs/shadow-paired.json"
```

- 完整记录：`/data/zju-47/hejingyu/shadowdream-validation-20261028-333b5d4/runs/shadow-paired.json`；launcher log：`logs/real-shadow-paired.log`。

## Shadow-UCB 搜索与训练更新

首次 Cycle 1 尝试使用 4 seeds、`max_steps=16`、4 个 Editor attempts。真实 root episode 完成：4 个 seeds 均 `success=false, progress=0.0`，累计 64 个模型调用；Editor prompt 达 183,248 tokens，超过 Qwen2.5-3B-Instruct 的 32,768-token 模型上下文配置。第一个真实生成样本为 512 tokens，解析失败：`unmatched ']' (<unknown>, line 1)`；因此只有 root 节点，没有有效 Shadow-UCB 子节点，也没有进入 GRPO。

本次自启动进程在首个无效候选后由验证方终止（exit 143）；完整部分产物保存在 `/data/zju-47/hejingyu/shadowdream-validation-20261028-333b5d4/runs/cycle-01-aborted-overlong-context`，launcher log 为 `logs/cycle-01.log`。本次模型权重从 NFS 挂载 `/data/zju-130` 读取；3B 权重加载约 14 分钟，执行阶段还扫描了模型目录用于快照哈希。用户随后要求“全部停止”，因此没有执行较短配置重跑。

后续实现修正针对该失败点：Editor 提示将只携带压缩的公开轨迹摘要，不再携带原始 prompt/token/logprob 数组；模型上下文不足时会在生成前给出明确失败，而不是把超长输入送入 Editor。候选 parser 会在执行前校验唯一的同步 `step(view, state)` 接口。搜索 UCB 的 Shadow 先验会同时计入增益与退化，保留两项原始统计；运行预算会区分并汇总 Editor 与 Service 的调用数和 token 数。Service GRPO 指标新增可训练参数 L1 变化与 `parameter_update` 字段。上述本地修正尚未运行测试或真实 GPU 验证。

运行日志：`/data/zju-47/hejingyu/shadowdream-validation-20261028-333b5d4/logs/cycle-01.log`。

## 连续 Cycle 与快照继承

未完成 Cycle 1，未生成可供下一轮继承的 cycle `state.json`，也未启动 Cycle 2。因此 Service snapshot、程序版本或训练 Adapter 的跨 Cycle 继承均未验证。

## 研究效果实验

本次真实运行验证只检查系统能否执行指定环境、搜索与训练路径。除非另行完成预注册/配对的普通 UCB 对照并报告多任务结果，否则不对 Shadow-UCB 的搜索效果、Service Learning 的收益或二者之间的因果关系作结论。

## 远端证据目录

验证根目录：`/data/zju-47/hejingyu/shadowdream-validation-20261028-333b5d4/`。其中 `logs/` 保存命令输出，`inputs/` 保存任务、witness/checkpoint 与程序输入，`runs/` 保存轨迹和结构化结果；Ray/VeRL 临时文件与完整 Cycle 输出在周期结束后归档并记录。

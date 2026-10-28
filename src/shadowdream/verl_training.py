"""Train Editor LoRA adapters with VeRL's V1 GRPO implementation."""
import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


VERL_COMMIT = "9268aad322df93cc61e9c168d66d395256dbe87a"


def _model_path(config):
    path = Path(config["model"]["path"]).expanduser()
    if path.exists():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=config["model"]["path"], revision=config["model"]["revision"])


def dataset_rows(samples, tokenizer):
    groups = OrderedDict()
    for index, sample in enumerate(samples):
        if not sample["prompt_ids"] or not sample["generated_ids"]:
            raise ValueError("replay requires nonempty prompt and response tokens")
        if len(sample["generated_ids"]) != len(sample["behavior_logprobs"]):
            raise ValueError("response tokens and behavior log-probabilities must align")
        trajectory_id = sample["task_metrics"].get("trajectory_id", str(index))
        groups.setdefault(sample["group_id"], OrderedDict()).setdefault(trajectory_id, []).append(sample)

    rows = []
    for trajectories in groups.values():
        if len(trajectories) < 2:
            raise ValueError("Editor GRPO requires at least two generations per group")
        replay = []
        first = next(iter(trajectories.values()))[0]
        for trajectory in trajectories.values():
            rewards = {sample["reward"] for sample in trajectory}
            weights = [sample["task_metrics"].get("sample_weight", 1.0) for sample in trajectory]
            if len(rewards) != 1:
                raise ValueError("calls from one Editor trajectory must share a reward")
            if abs(sum(weights) - 1.0) > 1e-6 or any(weight <= 0 for weight in weights):
                raise ValueError("Editor call weights must be positive and sum to one per trajectory")
            replay.append([
                {
                    "prompt_ids": sample["prompt_ids"],
                    "response_ids": sample["generated_ids"],
                    "response_logprobs": sample["behavior_logprobs"],
                    "reward": float(sample["reward"]),
                    "sample_weight": weight,
                }
                for sample, weight in zip(trajectory, weights)
            ])
        messages = first["task_metrics"].get("prompt_messages")
        if messages is None:
            messages = [{"role": "user", "content": tokenizer.decode(
                first["prompt_ids"], skip_special_tokens=False)}]
        rows.append({
            "prompt": messages,
            "data_source": "shadowdream",
            "reward_model": {"ground_truth": "{}"},
            "extra_info": {"index": len(rows), "problem": json.dumps(messages, ensure_ascii=False)},
            "agent_name": "shadowdream_editor_replay",
            "__rollout_n__": len(replay),
            "replay_trajectories": json.dumps(replay),
        })
    return rows


def build_verl_overrides(config, dataset_path, model_path, adapter_path,
                         row_count, call_count, prompt_length, response_length, checkpoint_dir):
    lora = config["model"]["lora"]
    dtype = config["model"]["dtype"]
    model_dtype = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[dtype]
    tracking = config.get("tracking", {})
    token_budget = prompt_length + response_length
    return [
        f"data.train_files={dataset_path}",
        f"data.val_files={dataset_path}",
        f"data.train_batch_size={row_count}",
        f"data.val_batch_size={row_count}",
        f"data.max_prompt_length={prompt_length}",
        f"data.max_response_length={response_length}",
        "data.filter_overlong_prompts=False",
        "data.shuffle=False",
        "data.dataloader_num_workers=0",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"actor_rollout_ref.model.path={model_path}",
        f"actor_rollout_ref.model.lora_adapter_path={adapter_path or 'null'}",
        f"actor_rollout_ref.model.lora_rank={lora['r']}",
        f"actor_rollout_ref.model.lora_alpha={lora['alpha']}",
        "actor_rollout_ref.model.target_modules=[{}]".format(",".join(lora["target_modules"])),
        "actor_rollout_ref.model.enable_gradient_checkpointing={}".format(
            str(config["training"]["gradient_checkpointing"]).lower()),
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        f"actor_rollout_ref.actor.fsdp_config.model_dtype={model_dtype}",
        f"actor_rollout_ref.actor.fsdp_config.dtype={dtype}",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={call_count}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={token_budget}",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
        "actor_rollout_ref.actor.use_kl_loss=False",
        f"actor_rollout_ref.actor.optim.lr={config['training']['learning_rate']}",
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]",
        "+actor_rollout_ref.actor.checkpoint.save_lora_only=True",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.n=1",
        f"actor_rollout_ref.rollout.temperature={config['model']['temperature']}",
        f"actor_rollout_ref.rollout.top_p={config['model']['top_p']}",
        f"actor_rollout_ref.rollout.dtype={dtype}",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.gpu_memory_utilization={}".format(
            config["training"].get("rollout_gpu_memory_utilization", 0.4)),
        f"actor_rollout_ref.rollout.max_model_len={token_budget}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={token_budget}",
        "actor_rollout_ref.rollout.max_num_seqs=8",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        "actor_rollout_ref.rollout.agent.default_agent_loop=shadowdream_editor_replay",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path={}".format(
            Path(__file__).with_name("verl_agent_loop.yaml").resolve()),
        "trainer.use_v1=True",
        "trainer.n_gpus_per_node={}".format(config["training"].get("grpo_gpus_per_node", 1)),
        "trainer.nnodes=1",
        "trainer.total_epochs=1",
        "trainer.total_training_steps=1",
        "trainer.save_freq=1",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        "trainer.resume_mode=disable",
        "trainer.logger=[{}]".format(",".join(tracking.get("logger", ["console"]))),
        "trainer.project_name={}".format(tracking.get("project_name", "shadowdream")),
        "trainer.experiment_name={}".format(tracking.get("trainer_experiment_name", "editor-grpo")),
        f"trainer.default_local_dir={checkpoint_dir}",
        "ray_kwargs.ray_init.num_cpus=8",
        "+ray_kwargs.ray_init.include_dashboard=False",
        "transfer_queue.backend.SimpleStorage.num_data_storage_units=1",
    ]


def train_with_verl(config, samples, out_dir, objective="grpo", adapter_path=None):
    if sys.version_info < (3, 11):
        raise RuntimeError("VeRL V1 requires Python 3.11 or newer")
    if objective != "grpo" or not samples or {sample["role"] for sample in samples} != {"edit"}:
        raise ValueError("Editor training requires nonempty GRPO samples")
    version = str(adapter_path) if adapter_path else "base"
    if {sample["adapter_version"] for sample in samples} != {version}:
        raise ValueError("training adapter must match the collection adapter")
    if {sample["model_revision"] for sample in samples} != {str(config["model"]["revision"])}:
        raise ValueError("training model revision must match the collection revision")
    if {sample["sampling_parameters"]["temperature"] for sample in samples} != {config["model"]["temperature"]}:
        raise ValueError("one update must use the collection temperature")
    if config["model"]["temperature"] <= 0 or config["model"]["top_p"] != 1.0:
        raise ValueError("Editor GRPO requires stochastic untruncated sampling")
    if config["model"]["lora"]["dropout"] != 0:
        raise ValueError("Editor VeRL training requires zero LoRA dropout")

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_path = _model_path(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    rows = dataset_rows(samples, tokenizer)
    dataset_path = output / "samples.parquet"
    pq.write_table(pa.Table.from_pylist(rows), dataset_path)
    calls = [call for row in rows for session in json.loads(row["replay_trajectories"]) for call in session]
    checkpoint_dir = output / "verl_checkpoints"
    overrides = build_verl_overrides(
        config, dataset_path, model_path, adapter_path, len(rows), len(calls),
        max(len(sample["prompt_ids"]) for sample in samples),
        max(len(sample["generated_ids"]) for sample in samples), checkpoint_dir,
    )
    env = os.environ.copy()
    env["RAY_TMPDIR"] = str(output / "ray")
    env["TRITON_CACHE_DIR"] = str(output / "triton")
    env["HYDRA_FULL_ERROR"] = "1"
    command = [sys.executable, "-m", "shadowdream.verl_runtime", *overrides,
               f"hydra.run.dir={output / 'hydra'}"]
    with (output / "trainer.log").open("w", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    iteration = int((checkpoint_dir / "latest_checkpointed_iteration.txt").read_text().strip())
    checkpoint = checkpoint_dir / f"global_step_{iteration}" / "actor"
    export = output / "adapter_export"
    with (output / "export.log").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                        "--local_dir", str(checkpoint), "--target_dir", str(export)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    adapter = export / "lora_adapter"
    from safetensors.torch import load_file
    import torch

    weights = load_file(str(adapter / "adapter_model.safetensors"))
    if any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("VeRL exported non-finite adapter weights")
    if adapter_path:
        before = load_file(str(Path(adapter_path) / "adapter_model.safetensors"))
        change = sum(float((value.float() - before[name].float()).abs().sum())
                     for name, value in weights.items())
    else:
        change = sum(float(value.float().abs().sum()) for name, value in weights.items()
                     if "lora_B" in name)
    result = {"backend": "verl", "verl_commit": VERL_COMMIT, "objective": "grpo",
              "groups": len(rows), "model_calls": len(calls), "trainer_step": iteration,
              "trainable_parameter_l1_change": change, "parameter_update": change > 0,
              "adapter": str(adapter)}
    (output / "train.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

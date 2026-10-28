"""Launch a mask-aware VeRL update from an on-policy Service collection."""
import json
import os
import subprocess
import sys
from pathlib import Path

from experiment_rsi.model import ModelSession
from experiment_rsi.verl_training import _model_path, build_verl_overrides
from .service_verl_data import export_parquet
from .snapshot import snapshot_id


def train_service_verl(config, collection_dir, output_dir, adapter_path=None):
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    import torch

    collection = Path(collection_dir)
    metadata = json.loads((collection / 'collection.json').read_text(encoding='utf-8'))
    replay = json.loads((collection / 'masked_replay.json').read_text(encoding='utf-8'))
    if metadata['temperature'] != config['model']['temperature'] or metadata['top_p'] != 1.0:
        raise ValueError('collection and actor sampling parameters do not match')
    if config['model']['lora']['dropout'] != 0:
        raise ValueError('Service VeRL requires zero LoRA dropout')
    actor = ModelSession(config, adapter_path=adapter_path)
    snapshot = snapshot_id(actor, dict(config, service_adapter=adapter_path))
    if snapshot != metadata['service_snapshot'] or any(g['service_snapshot'] != snapshot for g in replay):
        raise ValueError('Service actor snapshot differs from rollout snapshot')
    del actor
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_path = _model_path(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset_path = output / 'samples.parquet'
    row_count = export_parquet(collection / 'masked_replay.json', dataset_path, tokenizer)
    calls = [step for group in replay for episode in group['episodes'] for step in episode['steps']]
    if not calls:
        raise ValueError('no Service model calls to train')
    checkpoint_dir = output / 'verl_checkpoints'
    overrides = build_verl_overrides(config, dataset_path, model_path, model_path,
        adapter_path, row_count, len(calls), max(len(c['prompt_ids']) for c in calls),
        max(len(c['response_ids']) for c in calls), checkpoint_dir, 'grpo')
    for index, value in enumerate(overrides):
        if value.startswith('actor_rollout_ref.rollout.agent.default_agent_loop='):
            overrides[index] = 'actor_rollout_ref.rollout.agent.default_agent_loop=shadowdream_service_replay'
        elif value.startswith('actor_rollout_ref.rollout.agent.agent_loop_config_path='):
            overrides[index] = f'actor_rollout_ref.rollout.agent.agent_loop_config_path={Path(__file__).with_name("verl_service_agent.yaml").resolve()}'
    overrides.extend(['actor_rollout_ref.model.external_lib=shadowdream.verl_service_hook',
                      'actor_rollout_ref.model.use_fused_kernels=False',
                      'actor_rollout_ref.actor.calculate_entropy=False',
                      'actor_rollout_ref.actor.entropy_coeff=0.0'])
    env = os.environ.copy()
    env['HYDRA_FULL_ERROR'] = '1'
    env['RAY_TMPDIR'] = str(output / 'ray')
    env['TRITON_CACHE_DIR'] = str(output / 'triton')
    command = [sys.executable, '-m', 'experiment_rsi.verl_runtime', *overrides,
               f'hydra.run.dir={output / "hydra"}']
    with (output / 'trainer.log').open('w', encoding='utf-8') as log:
        log.write(json.dumps(command) + '\n')
        log.flush()
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    iteration = int((checkpoint_dir / 'latest_checkpointed_iteration.txt').read_text().strip())
    checkpoint = checkpoint_dir / f'global_step_{iteration}' / 'actor'
    export = output / 'adapter_export'
    with (output / 'export.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                        '--local_dir', str(checkpoint), '--target_dir', str(export)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    adapter = export / 'lora_adapter'
    weights = load_file(str(adapter / 'adapter_model.safetensors'))
    if any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError('VeRL exported non-finite adapter weights')
    if adapter_path:
        before = load_file(str(Path(adapter_path) / 'adapter_model.safetensors'))
        change = sum(float((value.float() - before[key].float()).abs().sum()) for key, value in weights.items())
    else:
        change = sum(float(value.float().abs().sum()) for key, value in weights.items() if 'lora_B' in key)
    result = {'backend': 'verl_masked', 'service_snapshot': snapshot, 'groups': row_count,
              'model_calls': len(calls), 'trainer_step': iteration, 'adapter': str(adapter),
              'trainable_parameter_l1_change': change, 'parameter_update': change > 0}
    (output / 'train.json').write_text(json.dumps(result, indent=2) + '\n')
    return result

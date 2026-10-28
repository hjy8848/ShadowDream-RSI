"""Build paired Editor GRPO samples from real search attempts."""
import json
import math
from collections import defaultdict
from pathlib import Path

from .verl_training import train_with_verl


def editor_samples(run_dir, alpha=1.0, shadow_weight=1.0, regression_weight=1.0,
                   invalid_penalty=1.0):
    root = Path(run_dir)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    group_size = manifest['config']['editor'].get('group_size', 1)
    if group_size < 2:
        raise ValueError('Editor GRPO requires at least two edits per group')
    groups = defaultdict(list)
    for path in sorted((root / 'events').glob('*.json')):
        event = json.loads(path.read_text(encoding='utf-8'))
        groups[event['group_id']].append(event)
    samples = []
    for group_id, events in groups.items():
        if len(events) != group_size or any(e['status'] == 'budget_exhausted' for e in events):
            continue
        if any(e['status'] not in {'accepted', 'invalid'} for e in events):
            raise ValueError('unknown Editor event status')
        if (len({e['parent_id'] for e in events}) != 1
            or any(e['prompt_messages'] != events[0]['prompt_messages'] for e in events)
            or any(e['generation']['prompt_ids'] != events[0]['generation']['prompt_ids'] for e in events)):
            raise ValueError('Editor group must share its parent and prompt')
        parent = json.loads((root / 'nodes' / (events[0]['parent_id'] + '.json')).read_text())
        for event in events:
            generation = event['generation']
            sampling = generation['sampling_parameters']
            if sampling['top_p'] != 1.0 or sampling.get('repetition_penalty') != 1.0:
                raise ValueError('Editor GRPO requires untruncated sampling with no repetition penalty')
            if event['status'] == 'accepted':
                child = json.loads((root / 'nodes' / (event['node_id'] + '.json')).read_text())
                reward = (alpha * (child['full_score'] - parent['full_score'])
                          + shadow_weight * child['shadow']['gain']
                          - regression_weight * child['shadow']['regression'])
            else:
                reward = -invalid_penalty
            samples.append({'group_id': group_id, 'role': 'edit',
                'prompt_ids': generation['prompt_ids'], 'generated_ids': generation['generated_ids'],
                'generated_text': generation['generated_text'],
                'behavior_logprobs': generation['behavior_logprobs'],
                'reward': reward, 'model_revision': generation['model_revision'],
                'adapter_version': generation['adapter_version'],
                'sampling_parameters': generation['sampling_parameters'],
                'training_eligible': True, 'behavior_snapshot': manifest['editor_snapshot'],
                'task_metrics': {'trajectory_id': str(event['attempt']), 'sample_weight': 1.0,
                                 'prompt_messages': event['prompt_messages']}})
    if not samples:
        raise ValueError('no complete Editor groups; collect at least two edits per parent')
    return samples


def verify_behavior(session, samples, tolerance=1e-3):
    """Compare stored behavior against the actor distribution used by VeRL."""
    import torch
    session.model.eval()
    largest = 0.0
    with torch.no_grad():
        for sample in samples:
            temperature = sample['sampling_parameters']['temperature']
            if temperature <= 0:
                raise ValueError('Editor GRPO requires stochastic sampling')
            prompt, response = sample['prompt_ids'], sample['generated_ids']
            if not prompt or not response or len(response) != len(sample['behavior_logprobs']):
                raise ValueError('invalid Editor rollout token or logprob lengths')
            device = next(session.model.parameters()).device
            ids = torch.tensor([prompt + response], dtype=torch.long, device=device)
            logits = session.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
            indices = torch.tensor(response, device=device)
            distribution = torch.log_softmax(logits[len(prompt)-1:len(prompt)-1+len(response)].float() / temperature, dim=-1)
            actual = distribution.gather(1, indices[:, None]).squeeze(1)
            expected = torch.tensor(sample['behavior_logprobs'], device=device)
            error = (actual - expected).abs().max().item()
            if not math.isfinite(error):
                raise ValueError('non-finite Editor behavior or actor logprobs')
            largest = max(largest, error)
            if error > tolerance:
                raise ValueError(f'Editor behavior and actor logprobs differ by {error:.6g}')
    return largest


def train_editor(config, samples, output_dir, adapter_path=None):
    from .model import ModelSession
    from .snapshot import snapshot_id
    actor = ModelSession(config, adapter_path=adapter_path)
    current_snapshot = snapshot_id(actor, {'model': config['model'], 'service_adapter': adapter_path})
    if {sample['behavior_snapshot'] for sample in samples} != {current_snapshot}:
        raise ValueError('Editor training snapshot differs from rollout snapshot')
    tolerance = config['training'].get('editor_logprob_tolerance', 1e-3)
    if tolerance <= 0:
        raise ValueError('Editor logprob tolerance must be positive')
    error = verify_behavior(actor, samples, tolerance=tolerance)
    del actor
    result = train_with_verl(config, samples, output_dir, 'grpo', adapter_path=adapter_path)
    result['behavior_logprob_max_error'] = error
    result['behavior_logprob_tolerance'] = tolerance
    summary = Path(output_dir) / 'train.json'
    summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result

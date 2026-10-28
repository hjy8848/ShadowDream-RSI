"""Prepare on-policy Service trajectories for a mask-aware VeRL replay loop.

These rows do not make the stock VeRL actor mask-aware. They must only be used
with an actor that applies `allowed_token_ids` during log-prob recomputation.
"""
import json
from pathlib import Path


def rows_from_masked_replay(replay, tokenizer):
    rows = []
    for index, group in enumerate(replay):
        episodes = group['episodes']
        if len(episodes) < 2:
            raise ValueError('Service GRPO requires at least two trajectories per task')
        if len({episode['seed'] for episode in episodes}) != len(episodes):
            raise ValueError('rollout seeds must be distinct within a task group')
        calls = []
        for episode in episodes:
            if episode['task_id'] != group['task_id'] or not episode['steps']:
                raise ValueError('every trajectory needs model actions from the same task')
            weight = 1.0 / len(episode['steps'])
            trajectory = []
            for step in episode['steps']:
                n = len(step['response_ids'])
                if not step['prompt_ids'] or n == 0 or n != len(step['behavior_logprobs']) or n != len(step['allowed_token_ids']):
                    raise ValueError('prompt, response, behavior logprobs and legal masks must align')
                if any(token not in allowed for token, allowed in zip(step['response_ids'], step['allowed_token_ids'])):
                    raise ValueError('response token is outside its recorded legal mask')
                trajectory.append({'prompt_ids': step['prompt_ids'], 'response_ids': step['response_ids'],
                    'response_logprobs': step['behavior_logprobs'],
                    'allowed_token_ids': step['allowed_token_ids'],
                    'reward': episode['reward']['total'], 'sample_weight': weight})
            calls.append(trajectory)
        prompt = [{'role': 'user', 'content': tokenizer.decode(calls[0][0]['prompt_ids'], skip_special_tokens=False)}]
        rows.append({'prompt': prompt, 'data_source': 'shadowdream_service',
            'reward_model': {'ground_truth': '{}'},
            'extra_info': {'index': index, 'problem': json.dumps(prompt, ensure_ascii=False),
                           'service_snapshot': group['service_snapshot'], 'task_id': group['task_id']},
            'agent_name': 'shadowdream_service_replay', '__rollout_n__': len(calls),
            'replay_trajectories': json.dumps(calls)})
    if not rows:
        raise ValueError('no complete Service GRPO groups')
    return rows


def export_parquet(replay_path, output_path, tokenizer):
    import pyarrow as pa
    import pyarrow.parquet as pq
    replay = json.loads(Path(replay_path).read_text(encoding='utf-8'))
    rows = rows_from_masked_replay(replay, tokenizer)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    pq.write_table(pa.Table.from_pylist(rows), output)
    return len(rows)

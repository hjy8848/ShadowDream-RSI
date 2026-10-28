"""Cost-indexed search curves from recorded actual evaluations."""
import json
import random
from statistics import mean
from pathlib import Path


def load_run(directory):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    initial = json.loads((root / 'nodes/root.json').read_text(encoding='utf-8'))
    best = initial['full_score']
    totals = {'editor_calls': 0, 'editor_tokens': 0,
              'env_steps': initial['cost']['env_steps'],
              'service_tokens': initial['cost']['model_tokens']}
    curve = [{**totals, 'best_score': best, 'node_id': 'root'}]
    for path in sorted((root / 'events').glob('*.json')):
        event = json.loads(path.read_text(encoding='utf-8'))
        totals['editor_calls'] += event['editor_calls']
        totals['editor_tokens'] += event['editor_tokens']
        for field in ('full_cost', 'shadow_cost'):
            if field in event:
                totals['env_steps'] += event[field]['env_steps']
                totals['service_tokens'] += event[field]['model_tokens']
        if event['status'] == 'accepted':
            node = json.loads((root / 'nodes' / (event['node_id'] + '.json')).read_text(encoding='utf-8'))
            best = max(best, node['full_score'])
        curve.append({**totals, 'best_score': best,
                      'node_id': event.get('node_id'), 'status': event['status']})
    cfg = manifest['config']
    comparison = {'service_snapshot': manifest['service_snapshot'],
                  'editor_snapshot': manifest['editor_snapshot'],
                  'code_fingerprint': manifest['code_fingerprint'],
                  'task_fingerprints': manifest['task_fingerprints'],
                  'root_service_sha256': manifest['root_service_sha256'],
                  'checkpoints': manifest['checkpoints'],
                  'shadow_protocol': cfg.get('shadow', {}).get('protocol', 'cold-start'),
                  'max_steps': cfg['run']['max_steps'],
                  'service_sampling': {k: cfg['model'][k] for k in ('temperature', 'top_p')},
                  'editor_sampling': cfg['editor']}
    return {'method': manifest['method'], 'feedback': manifest['feedback'],
            'rng_seed': manifest['rng_seed'],
            'comparison': comparison,
            'task_ids': manifest['task_ids'], 'scene_ids': manifest['scene_ids'],
            'seeds': manifest['seeds'], 'curve': curve}


def compare_at_budget(runs, metric, budget):
    if metric not in {'editor_calls', 'editor_tokens', 'env_steps', 'service_tokens'}:
        raise ValueError('unknown cost metric')
    if budget < 0 or not runs:
        raise ValueError('nonnegative budget and at least one run required')
    first = runs[0]
    for run in runs[1:]:
        for key in ('task_ids', 'scene_ids', 'seeds', 'comparison'):
            if run[key] != first[key]:
                raise ValueError(f'runs have different {key}')
    results = []
    for run in runs:
        # Only completed evaluations count; root evaluation may already exceed budget.
        eligible = [point for point in run['curve'] if point[metric] <= budget]
        score = eligible[-1]['best_score'] if eligible else None
        results.append({'method': run['method'], 'feedback': run['feedback'],
                        'best_score': score, 'used': eligible[-1][metric] if eligible else None,
                        'recorded_total': run['curve'][-1][metric],
                        'run_reached_budget': run['curve'][-1][metric] >= budget})
    return {'metric': metric, 'budget': budget, 'runs': results}


def paired_replicates(runs, metric, budget, baseline_method, baseline_feedback,
                      target_method, target_feedback, bootstrap_samples=10000):
    """Paired independent-search-seed deltas and a percentile bootstrap interval."""
    # Replicates share one frozen task/protocol/model setup across all seeds.
    compare_at_budget(runs, metric, budget)
    keys = ((baseline_method, baseline_feedback), (target_method, target_feedback))
    groups = {key: {} for key in keys}
    if keys[0] == keys[1]:
        raise ValueError('baseline and target must differ')
    for run in runs:
        key = (run['method'], run['feedback'])
        if key not in groups:
            raise ValueError(f'unexpected method in paired comparison: {key}')
        seed = run['rng_seed']
        if seed in groups[key]:
            raise ValueError(f'duplicate run for {key}, seed {seed}')
        groups[key][seed] = run
    seeds = set(groups[keys[0]])
    if len(seeds) < 2 or seeds != set(groups[keys[1]]):
        raise ValueError('at least two matching independent search seeds are required')
    if bootstrap_samples < 2:
        raise ValueError('at least two bootstrap samples are required')
    diffs = []
    for seed in sorted(seeds):
        a, b = groups[keys[0]][seed], groups[keys[1]][seed]
        comparison = compare_at_budget([a, b], metric, budget)['runs']
        if any(not item['run_reached_budget'] or item['best_score'] is None for item in comparison):
            raise ValueError(f'run for seed {seed} does not reach the requested budget')
        diffs.append({'rng_seed': seed, 'baseline': comparison[0]['best_score'],
                      'target': comparison[1]['best_score'],
                      'difference': comparison[1]['best_score'] - comparison[0]['best_score']})
    values = [item['difference'] for item in diffs]
    rng = random.Random(0)
    estimates = sorted(mean(rng.choices(values, k=len(values))) for _ in range(bootstrap_samples))
    lo = int(0.025 * (bootstrap_samples - 1))
    hi = int(0.975 * (bootstrap_samples - 1))
    return {'metric': metric, 'budget': budget,
            'baseline': {'method': baseline_method, 'feedback': baseline_feedback},
            'target': {'method': target_method, 'feedback': target_feedback},
            'paired_runs': diffs, 'mean_difference': mean(values),
            'bootstrap_percentile_95_ci': [estimates[lo], estimates[hi]],
            'bootstrap_samples': bootstrap_samples}

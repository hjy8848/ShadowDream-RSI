"""Select and test policies using task sets disjoint from the search."""
from dataclasses import asdict
from pathlib import Path
import json
from statistics import mean

from .host import run_episode


def task_identity(tasks):
    if not tasks:
        raise ValueError('evaluation task set must not be empty')
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate task IDs in evaluation set')
    return {'task_ids': sorted(ids), 'scene_ids': sorted({t.source_scene_id for t in tasks})}


def check_disjoint(a, b):
    if 'scene_ids' not in a or 'scene_ids' not in b:
        raise ValueError('task scene identities are required to enforce disjoint splits')
    for field in ('task_ids', 'scene_ids'):
        common = set(a[field]) & set(b[field])
        if common:
            raise ValueError(f'overlapping {field}: {sorted(common)}')


def evaluate(config, model, service, tasks, seeds, action_mode='policy'):
    if not seeds:
        raise ValueError('evaluation seeds must not be empty')
    episodes = [run_episode(config, service, model, task, seed, action_mode=action_mode)
                for task in tasks for seed in seeds]
    return {'score': mean(float(e.success) for e in episodes),
            'valid': not any(getattr(e, 'candidate_error', None) for e in episodes),
            'episodes': [asdict(e) for e in episodes]}


def select(config, model, snapshot, run_dir, tasks, seeds, incumbent=None):
    """Re-evaluate every archive policy on an independent selection set.

    incumbent, when present, is (config, model, snapshot, service). It is measured
    on the same selection set, allowing an old model snapshot to remain selected.
    """
    root = Path(run_dir)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    identity = task_identity(tasks)
    check_disjoint(manifest, identity)
    paths = sorted((root / 'nodes').glob('*.json'))
    if not paths:
        raise ValueError('archive contains no nodes')
    entries = []
    if incumbent is not None:
        old_config, old_model, old_snapshot, old_service = incumbent
        if old_config['run']['max_steps'] != config['run']['max_steps']:
            raise ValueError('incumbent and new model must use the same action budget')
        for key in ('temperature', 'top_p'):
            if old_config['model'][key] != config['model'][key]:
                raise ValueError('incumbent and new model must use the same sampling protocol')
        result = evaluate(old_config, old_model, old_service, tasks, seeds)
        entries.append({'node_id': None, 'snapshot': old_snapshot, 'service': old_service,
                        'incumbent': True, **result})
    for path in paths:
        node = json.loads(path.read_text(encoding='utf-8'))
        result = evaluate(config, model, node['service'], tasks, seeds)
        entries.append({'node_id': node['id'], 'snapshot': snapshot, 'service': node['service'],
                        'incumbent': False, **result})
    # Keep the incumbent on ties. For a fresh selection, prefer the root on ties.
    if incumbent is None:
        entries.sort(key=lambda e: e['node_id'] != 'root')
    eligible = [entry for entry in entries if entry['valid']]
    if not eligible:
        raise ValueError('all policies failed during selection evaluation')
    best = max(eligible, key=lambda e: e['score'])
    return {'search_split': {'task_ids': manifest['task_ids'], 'scene_ids': manifest['scene_ids']},
            'selection_split': identity, 'seeds': seeds, 'candidates': entries,
            'selected': {'node_id': best['node_id'], 'snapshot': best['snapshot'],
                         'service': best['service'], 'score': best['score'],
                         'incumbent': best['incumbent']}}


def test_selected(config, model, snapshot, selection, tasks, seeds, action_mode='policy'):
    identity = task_identity(tasks)
    check_disjoint(selection['search_split'], identity)
    check_disjoint(selection['selection_split'], identity)
    if snapshot != selection['selected']['snapshot']:
        raise ValueError('loaded model snapshot differs from the selected snapshot')
    result = evaluate(config, model, selection['selected']['service'], tasks, seeds, action_mode)
    return {'split': identity, 'seeds': seeds, 'snapshot': snapshot,
            'node_id': selection['selected']['node_id'], 'action_mode': action_mode, **result}

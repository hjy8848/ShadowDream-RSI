"""Run real Editor expansions against full tasks and paired checkpoints."""
from dataclasses import asdict
from hashlib import sha256
import ast
import json
import random
import time
from pathlib import Path
from statistics import mean

import experiment_rsi
from experiment_rsi.model import seed_everything

from .host import evaluate_paired, run_episode
from .budget import Budget, BudgetExhausted, difference
from .search import Archive, Node, Search


def policy_hash(source):
    return sha256(source.encode('utf-8')).hexdigest()


def code_fingerprint():
    """Identify the actual ShadowDream and Remedy Python sources used in a run."""
    digest = sha256()
    for label, root in (('shadowdream', Path(__file__).parent),
                        ('remedy', Path(experiment_rsi.__file__).parent)):
        for path in sorted(root.rglob('*.py')):
            name = f'{label}/{path.relative_to(root).as_posix()}'.encode()
            digest.update(len(name).to_bytes(8, 'big'))
            digest.update(name)
            digest.update(path.read_bytes())
    return digest.hexdigest()


def task_fingerprint(task):
    fields = dict(vars(task))
    path = fields.pop('task_game_file', None)
    if path is not None:
        digest = sha256()
        with Path(path).open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        fields['task_game_sha256'] = digest.hexdigest()
    return sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def editor_prompt(service, diagnostics):
    return [
        {'role': 'system', 'content': (
            'Improve this ALFWorld Python policy. Return only complete Python source '
            'defining step(view, state); do not wrap it in JSON. '
            'Return {kind: action, action: a legal command, state: object} or '
            '{kind: model, messages: chat messages, state: object}. '
            'view provides objective, observation, history, admissible_commands, '
            'knowledge and remaining_steps. Only standard library is available. '
            'Do not include markdown or change knowledge.')},
        {'role': 'user', 'content': json.dumps({'policy_source': service['policy_source'],
                                                'diagnostics': diagnostics}, ensure_ascii=False)}]


def parse_edit(text, knowledge):
    source = text.strip()
    if source.startswith('```') and source.endswith('```'):
        lines = source.splitlines()
        if len(lines) < 3 or lines[0].strip() not in {'```', '```python'}:
            raise ValueError('invalid Python code fence')
        source = '\n'.join(lines[1:-1])
    elif source.startswith('{'):
        value = json.loads(source)
        if not isinstance(value, dict) or set(value) != {'policy_source'} or not isinstance(value['policy_source'], str):
            raise ValueError('Editor JSON must contain only policy_source text')
        source = value['policy_source']
    tree = ast.parse(source)
    if not any(isinstance(node, ast.FunctionDef) and node.name == 'step' for node in tree.body):
        raise ValueError('policy must define a synchronous step function')
    return {'policy_source': source, 'knowledge': knowledge}


def candidate_error_category(message):
    message = message.lower()
    if 'exceeded' in message or 'timed out' in message or 'timeout' in message:
        return 'timeout'
    if 'not admissible' in message or 'no admissible' in message:
        return 'illegal_action'
    return 'runtime'


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def _full(config, service, model, tasks, seeds, budget=None):
    return [run_episode(config, service, model, task, seed, budget=budget)
            for task in tasks for seed in seeds]


def _episode_cost(episodes):
    return {'env_steps': sum(e.prefix_steps + len(e.trace) for e in episodes),
            'model_calls': sum(e.model_calls for e in episodes),
            'model_tokens': sum(e.model_tokens for e in episodes)}


def _metrics(episodes):
    return mean(float(e.success) for e in episodes), mean(e.progress for e in episodes)



def select_group_parent(method, archive, search, progress, rng, services, diagnostics):
    if method == 'greedy':
        parent = max(archive.nodes.values(), key=lambda n: n.full_score)
    elif method == 'random':
        parent = rng.choice(list(archive.nodes.values()))
    elif method == 'progress_ucb':
        parent = search.select(priors=progress)
    else:
        old_beta = search.beta
        if method == 'ucb':
            search.beta = 0.0
        parent = search.select()
        search.beta = old_beta
    return parent, editor_prompt(services[parent.id], diagnostics[parent.id])

def run_search(config, editor, service_model, root_service, tasks, checkpoints, seeds,
               out_dir, attempts, method='shadow_ucb', feedback='improvement', rng_seed=0,
               service_snapshot='unspecified', editor_snapshot='unspecified'):
    """Run a fixed number of Editor calls. Each valid candidate receives both evaluations.

    Every invocation uses the same evaluation protocol and frozen model objects. The
    supplied task set is a search set; selection/test belong to a separate invocation.
    """
    if not tasks or not seeds or attempts < 0:
        raise ValueError('nonempty tasks/seeds and nonnegative attempt budget are required')
    if len({t.task_id for t in tasks}) != len(tasks) or len(set(seeds)) != len(seeds):
        raise ValueError('task IDs and seeds must not repeat')
    if set(checkpoints) != {t.task_id for t in tasks}:
        raise ValueError('checkpoint map must match the search task IDs')
    for task in tasks:
        values = checkpoints[task.task_id]
        if (not values or len(set(values)) != len(values)
            or any(not isinstance(k, int) or isinstance(k, bool) or k <= 0
                   or k >= len(task.witness_actions) or k >= config['run']['max_steps'] for k in values)):
            raise ValueError(f'invalid checkpoints for task {task.task_id}')
    if method not in {'greedy', 'random', 'ucb', 'progress_ucb', 'shadow_ucb'}:
        raise ValueError('unknown search method')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('output already contains a run; choose a new directory')
    manifest = {'method': method, 'feedback': feedback, 'attempt_budget': attempts,
                'service_snapshot': service_snapshot, 'editor_snapshot': editor_snapshot,
                'code_fingerprint': code_fingerprint(), 'task_ids': [t.task_id for t in tasks],
                'scene_ids': sorted({t.source_scene_id for t in tasks}),
                'task_fingerprints': {t.task_id: task_fingerprint(t) for t in tasks},
                'root_service_sha256': sha256(json.dumps(root_service, sort_keys=True,
                    ensure_ascii=False).encode()).hexdigest(),
                'checkpoints': checkpoints, 'seeds': seeds, 'rng_seed': rng_seed,
                'config': config}
    _write_json(out / 'manifest.json', manifest)
    started = time.monotonic()
    limit = config['search'].get('env_step_budget')
    budget = Budget(limit) if limit is not None else None
    try:
        root_results = _full(config, root_service, service_model, tasks, seeds, budget)
    except BudgetExhausted:
        _write_json(out / 'budget_exhausted.json', {'phase': 'root', 'cost': budget.snapshot()})
        raise
    except Exception as error:
        _write_json(out / 'failure.json', {'phase': 'root', 'type': type(error).__name__,
                    'error': str(error), 'cost': budget.snapshot() if budget is not None else None})
        raise
    if any(e.candidate_error for e in root_results):
        _write_json(out / 'failure.json', {'phase': 'root', 'type': 'candidate',
                    'episodes': [asdict(e) for e in root_results]})
        raise ValueError('root policy failed during the full-task evaluation')
    root_score, root_progress = _metrics(root_results)
    archive = Archive()
    archive.add_root(Node('root', None, root_score))
    search = Search(archive, feedback, c=config['search']['c'], beta=config['search']['beta'])
    services = {'root': root_service}
    progress = {'root': root_progress}
    root_full = [asdict(x) for x in root_results]
    diagnostics = {'root': {'full': root_full, 'shadow': None}}
    _write_json(out / 'nodes' / 'root.json', {'id': 'root', 'parent_id': None,
        'policy_hash': policy_hash(root_service['policy_source']), 'service': root_service,
        'service_snapshot': service_snapshot, 'full': root_full,
        'full_score': root_score, 'progress': root_progress, 'cost': _episode_cost(root_results)})
    rng = random.Random(rng_seed)
    if hasattr(editor, 'model'):
        # VeRL recomputes untruncated token probabilities for Editor updates.
        editor.model.generation_config.repetition_penalty = 1.0
    group_size = config['editor'].get('group_size', 1)
    if group_size < 1:
        raise ValueError('editor group_size must be positive')
    for attempt in range(attempts):
        if budget is not None and budget.env_steps >= budget.limit:
            break
        if attempt % group_size == 0:
            parent, messages = select_group_parent(method, archive, search, progress, rng, services, diagnostics)
        # Environment evaluation resets global RNGs; reseed each Editor draw explicitly.
        seed_everything(rng_seed + attempt, getattr(editor, 'torch', None))
        try:
            generation = editor.generate(messages, max_new_tokens=config['editor']['max_new_tokens'],
                                         temperature=config['editor']['temperature'],
                                         top_p=config['editor']['top_p'])
        except Exception as error:
            _write_json(out / 'failure.json', {'phase': 'editor', 'attempt': attempt,
                        'parent_id': parent.id, 'type': type(error).__name__, 'error': str(error),
                        'cost': budget.snapshot() if budget is not None else None})
            raise
        if 'sampling_parameters' in generation:
            generation['sampling_parameters']['repetition_penalty'] = 1.0
        event = {'attempt': attempt, 'group_id': f'g{attempt // group_size:06}',
                 'parent_id': parent.id, 'prompt_messages': messages, 'generation': generation,
                 'editor_calls': 1, 'editor_tokens': (len(generation.get('prompt_ids', [])) +
                                                      len(generation.get('generated_ids', [])))}
        try:
            service = parse_edit(generation['generated_text'], services[parent.id]['knowledge'])
        except (ValueError, SyntaxError) as error:
            event.update({'status': 'invalid', 'error': str(error),
                          'error_category': 'syntax' if isinstance(error, SyntaxError) else 'schema'})
            _write_json(out / 'events' / f'{attempt:06}.json', event)
            continue
        phase_start = budget.snapshot() if budget is not None else None
        try:
            full = _full(config, service, service_model, tasks, seeds, budget)
        except BudgetExhausted:
            event.update({'status': 'budget_exhausted',
                          'full_cost': difference(budget.snapshot(), phase_start)})
            _write_json(out / 'events' / f'{attempt:06}.json', event)
            break
        except Exception as error:
            _write_json(out / 'failure.json', {'phase': 'full', 'attempt': attempt,
                        'type': type(error).__name__, 'error': str(error),
                        'cost': budget.snapshot() if budget is not None else None})
            raise
        event['full_cost'] = _episode_cost(full)

        # Runtime errors cannot become successful candidate nodes; preserve their evidence.
        if any(e.candidate_error for e in full):
            details = next(e.candidate_error for e in full if e.candidate_error)
            event.update({'status': 'invalid', 'error': details,
                          'error_category': candidate_error_category(details), 'full': [asdict(e) for e in full]})
            _write_json(out / 'events' / f'{attempt:06}.json', event)
            continue
        phase_start = budget.snapshot() if budget is not None else None
        try:
            if budget is None:
                shadow, episodes = evaluate_paired(config, services[parent.id], service,
                                                   service_model, tasks, checkpoints, seeds)
            else:
                shadow, episodes = evaluate_paired(config, services[parent.id], service,
                                                   service_model, tasks, checkpoints, seeds, budget)
        except BudgetExhausted:
            event.update({'status': 'budget_exhausted',
                          'shadow_cost': difference(budget.snapshot(), phase_start)})
            _write_json(out / 'events' / f'{attempt:06}.json', event)
            break
        except Exception as error:
            _write_json(out / 'failure.json', {'phase': 'shadow', 'attempt': attempt,
                        'type': type(error).__name__, 'error': str(error),
                        'cost': budget.snapshot() if budget is not None else None})
            raise
        event['shadow_cost'] = _episode_cost(episodes['parent'] + episodes['child'])
        if any(e.candidate_error for e in episodes['parent']):
            raise ValueError('parent policy failed during checkpoint evaluation')
        if any(e.candidate_error for e in episodes['child']):
            details = next(e.candidate_error for e in episodes['child'] if e.candidate_error)
            event.update({'status': 'invalid', 'error': details,
                          'error_category': candidate_error_category(details), 'shadow_episodes': {k: [asdict(e) for e in v] for k, v in episodes.items()}})
            _write_json(out / 'events' / f'{attempt:06}.json', event)
            continue
        full_score, progress_score = _metrics(full)
        node_id = f'n{attempt:06}'
        node = Node(node_id, parent.id, full_score, shadow.gain)
        feedback_event = search.add_child(node)
        services[node_id], progress[node_id] = service, progress_score
        child_full = [asdict(e) for e in full]
        diagnostics[node_id] = {'full': child_full, 'shadow': asdict(shadow)}
        _write_json(out / 'nodes' / f'{node_id}.json', {
            'id': node_id, 'parent_id': parent.id, 'policy_hash': policy_hash(service['policy_source']),
            'service': service, 'service_snapshot': service_snapshot, 'full': child_full,
            'full_score': full_score, 'progress': progress_score, 'shadow': asdict(shadow),
            'shadow_episodes': {k: [asdict(e) for e in v] for k, v in episodes.items()}})
        event.update({'status': 'accepted', 'node_id': node_id, 'feedback': feedback_event})
        _write_json(out / 'events' / f'{attempt:06}.json', event)
    _write_json(out / 'archive.json', {'nodes': [asdict(n) for n in archive.nodes.values()],
                                      'best_score': archive.best_score,
                                      'budget': budget.snapshot() if budget is not None else None,
                                      'wall_seconds': time.monotonic() - started})
    return archive

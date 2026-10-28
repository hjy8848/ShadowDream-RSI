"""Command-line entry points for actual ALFWorld evaluations and search."""
import argparse
import json
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

from .model import ModelSession
from .policy import load_service
from .tasks import read_tasks
from .host import evaluate_paired
from .runner import run_search
from .snapshot import snapshot_id
from .action_model import ActionModel


def _json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    alfworld = value.get('alfworld')
    if isinstance(alfworld, dict):
        if alfworld.get('config_path'):
            config_path = Path(alfworld['config_path']).expanduser()
            if not config_path.is_absolute():
                alfworld['config_path'] = str((Path(path).resolve().parent / config_path).resolve())
        else:
            alfworld['config_path'] = str(files('shadowdream').joinpath('alfworld_text.yaml'))
    return value


def _tasks(path):
    return read_tasks(path)


def main():
    parser = argparse.ArgumentParser(prog='shadowdream')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('shadow', 'search'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True, help='JSON run configuration')
        p.add_argument('--tasks', required=True, help='ShadowDream task JSONL')
        p.add_argument('--checkpoints', required=True, help='JSON mapping task ID to prefix lengths')
        p.add_argument('--parent', required=True, help='service JSON with policy_source and knowledge')
        p.add_argument('--out', required=True)
        if name == 'shadow':
            p.add_argument('--child', required=True)
        else:
            p.add_argument('--attempts', type=int, required=True)
            p.add_argument('--method', choices=['greedy', 'random', 'ucb', 'progress_ucb', 'shadow_ucb'], default='shadow_ucb')
            p.add_argument('--feedback', choices=['absolute', 'improvement'], default='improvement')
    for name in ('select', 'test'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--tasks', required=True)
        p.add_argument('--out', required=True)
        if name == 'select':
            p.add_argument('--run-dir', required=True)
            p.add_argument('--incumbent-config')
            p.add_argument('--incumbent-service')
        else:
            p.add_argument('--selection', required=True)
            p.add_argument('--action-mode', choices=['policy', 'direct'], default='policy')
    p = sub.add_parser('cycle')
    p.add_argument('--config')
    p.add_argument('--parent')
    p.add_argument('--state', help='previous cycle state.json')
    p.add_argument('--tasks', required=True)
    p.add_argument('--selection-tasks', required=True)
    p.add_argument('--checkpoints', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--attempts', type=int, required=True)
    p.add_argument('--method', choices=['greedy', 'random', 'ucb', 'progress_ucb', 'shadow_ucb'], default='shadow_ucb')
    p.add_argument('--feedback', choices=['absolute', 'improvement'], default='improvement')
    p.add_argument('--service-update', choices=['none', 'sft', 'grpo', 'sft-grpo'], default='none')
    p.add_argument('--train-editor', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('report')
    p.add_argument('--run-dir', action='append', required=True)
    p.add_argument('--metric', choices=['editor_calls', 'editor_tokens', 'env_steps', 'service_tokens'], required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('paired-report')
    p.add_argument('--run-dir', action='append', required=True)
    p.add_argument('--metric', choices=['editor_calls', 'editor_tokens', 'env_steps', 'service_tokens'], required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--baseline-method', required=True)
    p.add_argument('--baseline-feedback', choices=['absolute', 'improvement'], required=True)
    p.add_argument('--target-method', required=True)
    p.add_argument('--target-feedback', choices=['absolute', 'improvement'], required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('service-data')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--config', help='needed when using --token-budget')
    p.add_argument('--token-budget', type=int)
    p = sub.add_parser('witness-data')
    p.add_argument('--config', required=True)
    p.add_argument('--tasks', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--token-budget', type=int)
    p = sub.add_parser('service-sft')
    p.add_argument('--config', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--adapter')
    p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('service-grpo')
    p.add_argument('--config', required=True)
    p.add_argument('--tasks', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--adapter')
    p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('editor-grpo')
    p.add_argument('--config', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--adapter')
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--shadow-weight', type=float, default=1.0)
    p.add_argument('--regression-weight', type=float, default=1.0)
    p.add_argument('--invalid-penalty', type=float, default=1.0)
    args = parser.parse_args()
    if args.command == 'cycle':
        from .cycle import run_cycle
        train_tasks, selection_tasks = _tasks(args.tasks), _tasks(args.selection_tasks)
        if args.state:
            if args.config or args.parent:
                parser.error('--state cannot be combined with --config or --parent')
            from .validation import check_disjoint, task_identity
            previous = _json(args.state)
            check_disjoint(previous['selection_split'], task_identity(train_tasks))
            check_disjoint(previous['search_split'], task_identity(selection_tasks))
            config, parent, expected = previous['config'], previous['service'], previous['snapshot']
        else:
            if not args.config or not args.parent:
                parser.error('provide --state or both --config and --parent')
            config, parent, expected = _json(args.config), load_service(args.parent), None
        run_cycle(config, parent, train_tasks,
                  selection_tasks, _json(args.checkpoints), args.out,
                  args.attempts, args.method, args.feedback, args.service_update,
                  args.train_editor, args.device, expected)
        return
    if args.command == 'paired-report':
        from .report import load_run, paired_replicates
        result = paired_replicates([load_run(path) for path in args.run_dir], args.metric, args.budget,
            args.baseline_method, args.baseline_feedback, args.target_method, args.target_feedback)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        return
    if args.command == 'report':
        from .report import compare_at_budget, load_run
        result = compare_at_budget([load_run(path) for path in args.run_dir], args.metric, args.budget)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        return
    if args.command in {'select', 'test'}:
        from .validation import select, test_selected
        config = _json(args.config)
        model = ActionModel(ModelSession({'model': config['model']}, adapter_path=config.get('service_adapter')))
        snapshot = snapshot_id(model, config)
        tasks = _tasks(args.tasks)
        if args.command == 'select':
            if bool(args.incumbent_config) != bool(args.incumbent_service):
                parser.error('--incumbent-config and --incumbent-service must be supplied together')
            incumbent = None
            if args.incumbent_config:
                old_config = _json(args.incumbent_config)
                old_model = ActionModel(ModelSession({'model': old_config['model']}, adapter_path=old_config.get('service_adapter')))
                incumbent = (old_config, old_model,
                             snapshot_id(old_model, old_config),
                             load_service(args.incumbent_service))
            result = select(config, model, snapshot, args.run_dir, tasks, config['seeds'], incumbent)
        else:
            result = test_selected(config, model, snapshot, _json(args.selection), tasks,
                                   config['seeds'], args.action_mode)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        return
    if args.command == 'service-grpo':
        from .service_grpo import train
        train(_json(args.config), _tasks(args.tasks), args.out, args.adapter, args.device)
        return
    if args.command == 'editor-grpo':
        from .editor_learning import editor_samples, train_editor
        config = _json(args.config)
        config['model'] = dict(config['editor_model'], temperature=config['editor']['temperature'],
                               top_p=config['editor']['top_p'], max_new_tokens=config['editor']['max_new_tokens'])
        samples = editor_samples(args.run_dir, args.alpha, args.shadow_weight,
                                 args.regression_weight, args.invalid_penalty)
        train_editor(config, samples, args.out, args.adapter)
        return
    if args.command in {'service-data', 'witness-data', 'service-sft'}:
        from .service_learning import examples_from_run, witness_examples, write_examples, train_sft, cap_with_config
        if args.command == 'service-sft':
            train_sft(_json(args.config), args.dataset, args.out, args.adapter, args.device)
        else:
            if args.token_budget is not None and not args.config:
                parser.error('--config is required with --token-budget')
            config = _json(args.config) if args.config else None
            examples = (examples_from_run(args.run_dir) if args.command == 'service-data'
                        else witness_examples(config, _tasks(args.tasks)))
            if args.token_budget is not None:
                examples, used = cap_with_config(examples, config, args.token_budget)
                print(f'Selected {len(examples)} complete examples using {used} input tokens')
            write_examples(args.out, examples)
        return
    config, tasks, checkpoints = _json(args.config), _tasks(args.tasks), _json(args.checkpoints)
    parent = load_service(args.parent)
    service_config = {'model': config['model']}
    service_model = ActionModel(ModelSession(service_config, adapter_path=config.get('service_adapter')))
    if args.command == 'shadow':
        result, episodes = evaluate_paired(config, parent, load_service(args.child), service_model,
                                           tasks, checkpoints, config['seeds'])
        if result is None:
            raise ValueError('candidate execution failed during checkpoint evaluation')
        output = {'shadow': asdict(result),
                  'episodes': {k: [asdict(e) for e in v] for k, v in episodes.items()}}
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    else:
        editor_config = {'model': config['editor_model']}
        editor = ModelSession(editor_config, adapter_path=config.get('editor_adapter'))
        snapshot = snapshot_id(service_model, config)
        run_search(config, editor, service_model, parent, tasks, checkpoints, config['seeds'],
                   args.out, args.attempts, args.method, args.feedback,
                   config['run']['seed'], snapshot,
                   snapshot_id(editor, {'model': config['editor_model'],
                                        'service_adapter': config.get('editor_adapter')}))

if __name__ == '__main__':
    main()

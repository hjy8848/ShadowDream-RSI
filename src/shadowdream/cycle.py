"""One search/training cycle with fresh independent selection."""
import copy
import json
from pathlib import Path

from experiment_rsi.model import ModelSession
from .action_model import ActionModel
from .editor_learning import editor_samples, train_editor
from .runner import run_search
from .service_grpo import train as train_service_grpo
from .service_learning import cap_with_config, examples_from_run, train_sft, write_examples
from .snapshot import snapshot_id
from .validation import check_disjoint, select, task_identity


def run_cycle(config, root_service, train_tasks, selection_tasks, checkpoints,
              output_dir, attempts, method='shadow_ucb', feedback='improvement',
              service_update='none', train_editor_update=False, device='cuda:0',
              expected_snapshot=None):
    if service_update not in {'none', 'sft', 'grpo', 'sft-grpo'}:
        raise ValueError('unknown Service update')
    check_disjoint(task_identity(train_tasks), task_identity(selection_tasks))
    if service_update in {'grpo', 'sft-grpo'} and (
        config['cycle']['grpo_temperature'] <= 0 or len(set(config['seeds'])) < 2
    ):
        raise ValueError('online GRPO needs positive temperature and at least two distinct seeds')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    current = copy.deepcopy(config)
    service_model = ActionModel(ModelSession({'model': current['model']}, adapter_path=current.get('service_adapter')))
    old_snapshot = snapshot_id(service_model, current)
    if expected_snapshot is not None and old_snapshot != expected_snapshot:
        raise ValueError('cycle state Service snapshot differs from loaded model')
    editor = ModelSession({'model': current['editor_model']}, adapter_path=current.get('editor_adapter'))
    editor_snapshot = snapshot_id(editor, {'model': current['editor_model'],
                                          'service_adapter': current.get('editor_adapter')})
    run_search(current, editor, service_model, root_service, train_tasks, checkpoints,
               current['seeds'], output / 'search', attempts, method, feedback,
               current['run']['seed'], old_snapshot, editor_snapshot)
    del editor
    if train_editor_update:
        editor_config = copy.deepcopy(current)
        editor_config['model'] = dict(current['editor_model'],
            temperature=current['editor']['temperature'], top_p=current['editor']['top_p'],
            max_new_tokens=current['editor']['max_new_tokens'])
        samples = editor_samples(output / 'search', **current.get('editor_reward', {}))
        result = train_editor(editor_config, samples, output / 'editor_update', current.get('editor_adapter'))
        current['editor_adapter'] = result['adapter']
    updated = copy.deepcopy(current)
    if service_update in {'sft', 'sft-grpo'}:
        data_path = output / 'service_data.jsonl'
        examples = examples_from_run(output / 'search')
        token_budget = current['training'].get('sft_token_budget')
        if token_budget is not None:
            examples, _ = cap_with_config(examples, current, token_budget)
        write_examples(data_path, examples)
        result = train_sft(current, data_path, output / 'service_sft', current.get('service_adapter'), device)
        updated['service_adapter'] = result['adapter']
    if service_update in {'grpo', 'sft-grpo'}:
        grpo_config = copy.deepcopy(updated)
        grpo_config['model']['temperature'] = config['cycle']['grpo_temperature']
        grpo_config['model']['top_p'] = 1.0
        train_service_grpo(grpo_config, train_tasks, output / 'service_grpo',
                           updated.get('service_adapter'), device)
        updated['service_adapter'] = str((output / 'service_grpo' / 'adapter').resolve())
    if service_update == 'none':
        updated_model, new_snapshot = service_model, old_snapshot
        incumbent = None
    else:
        updated_model = ActionModel(ModelSession({'model': updated['model']}, adapter_path=updated['service_adapter']))
        new_snapshot = snapshot_id(updated_model, updated)
        incumbent = (current, service_model, old_snapshot, root_service)
    selected = select(updated, updated_model, new_snapshot, output / 'search',
                      selection_tasks, current['seeds'], incumbent)
    (output / 'selection.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2) + '\n')
    selected_config = copy.deepcopy(current if selected['selected']['incumbent'] else updated)
    # Editor can advance independently when the Service update is rejected.
    if train_editor_update:
        selected_config['editor_adapter'] = current['editor_adapter']
    state = {'config': selected_config, 'service': selected['selected']['service'],
             'snapshot': selected['selected']['snapshot'], 'selection': selected['selected'],
             'search_split': selected['search_split'], 'selection_split': selected['selection_split']}
    (output / 'state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    return state

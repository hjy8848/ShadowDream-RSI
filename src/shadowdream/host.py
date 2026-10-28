"""ALFWorld execution against Remedy's environment and isolated policy worker.

Only observation, objective, public action history and admissible commands reach
candidate policies. Witnesses and facts remain on the host.
"""
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from experiment_rsi.alfworld import AlfworldEpisode
from experiment_rsi.evaluation import direct_action_messages, replace_native_objective, resolve_action, task_progress
from experiment_rsi.model import seed_everything
from experiment_rsi.policy import make_policy_view
from experiment_rsi.synthesis import goal_reached
from experiment_rsi.worker import CandidateExecutionError, PolicyWorker

from .shadow import CheckpointOutcome, compare


@dataclass
class Episode:
    task_id: str
    checkpoint: int
    seed: int
    success: bool = False
    progress: float = 0.0
    prefix_steps: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    candidate_error: str | None = None
    checkpoint_protocol: str = 'cold-start'
    model_calls: int = 0
    model_tokens: int = 0


def run_episode(config, service, model, task, seed, checkpoint=0, action_mode='policy', budget=None):
    """Run from the initial state or a replayed, nonterminal witness prefix.

    max_steps counts both prefix and continuation actions. Each invocation gets
    a fresh environment and worker. Cold-start uses a fixed history window;
    history rehydration provides the full public history to the policy.
    """
    if action_mode not in {'policy', 'direct'}:
        raise ValueError('action_mode must be policy or direct')
    if action_mode == 'direct' and model is None:
        raise ValueError('direct mode requires a Service model')
    protocol = config.get('shadow', {}).get('protocol', 'cold-start')
    if protocol not in {'cold-start', 'history'}:
        raise ValueError('shadow protocol must be cold-start or history')
    witness = task.witness_actions
    max_steps = config['run']['max_steps']
    if checkpoint < 0 or checkpoint >= len(witness) or checkpoint >= max_steps:
        raise ValueError('checkpoint must leave a witness suffix and action budget')
    seed_everything(seed, getattr(model, 'torch', None))
    result = Episode(task_id=task.task_id, checkpoint=checkpoint, seed=seed,
                     checkpoint_protocol=protocol)
    history = []
    with AlfworldEpisode(config['alfworld']['config_path'], task.task_game_file,
                         task.source_split) as env:
        view = env.reset()
        initial_observation = replace_native_objective(view['observation'], task.instruction)
        if goal_reached(task.goal_atoms, view['facts']):
            raise ValueError('task is already solved at reset')
        for action in witness[:checkpoint]:
            if action not in view['admissible_commands']:
                raise ValueError(f'witness action is not admissible at prefix step {len(history)}: {action}')
            if budget is not None:
                budget.charge_step()
            view = env.step(action)
            history.append({'action': action, 'observation': replace_native_objective(
                view['observation'], task.instruction)})
            if view['done'] or view['won'] or goal_reached(task.goal_atoms, view['facts']):
                raise ValueError('witness prefix reached a terminal or solved state')
        result.prefix_steps = len(history)
        _, result.progress = task_progress(task, view['facts'])
        worker_context = (PolicyWorker(service, config['worker'], seed)
                          if action_mode == 'policy' else nullcontext(None))
        with worker_context as worker:
            state = {}
            for index in range(checkpoint, max_steps):
                if budget is not None:
                    budget.require_step()
                observation = replace_native_objective(view['observation'], task.instruction)
                policy_history = history[:] if protocol == 'history' else history[-5:]
                public_view = make_policy_view(observation, task.instruction, policy_history,
                                               view['admissible_commands'],
                                               service['knowledge'] if action_mode == 'policy' else [],
                                               max_steps - index)
                if protocol == 'history':
                    public_view['initial_observation'] = initial_observation
                try:
                    if action_mode == 'direct':
                        directive = {'kind': 'model', 'messages': direct_action_messages(
                            task.instruction, observation, history, view['admissible_commands'],
                            max_steps - index)}
                        state = {}
                    else:
                        directive, state = worker.step(public_view, state)
                    choices = state.get('action_choices', view['admissible_commands'])
                    if not isinstance(choices, list) or not all(isinstance(x, str) for x in choices):
                        raise CandidateExecutionError('action_choices must be a list of strings')
                    generation = None
                    if directive['kind'] == 'model':
                        if model is None:
                            raise ValueError('model directive requires a Service model')
                        allowed = ([str(i + 1) for i, c in enumerate(choices)
                                    if c in view['admissible_commands']]
                                   if 'action_choices' in state else view['admissible_commands'])
                        if not allowed:
                            raise CandidateExecutionError('policy has no admissible model action choices')
                        generation = model.generate(directive['messages'],
                            max_new_tokens=config['model']['max_new_tokens'],
                            temperature=config['model']['temperature'],
                            top_p=config['model']['top_p'], allowed_texts=allowed,
                            **({'enable_thinking': False} if action_mode == 'direct' else {}))
                        result.model_calls += 1
                        result.model_tokens += (len(generation.get('prompt_ids', [])) +
                                                len(generation.get('generated_ids', [])))
                        if budget is not None:
                            budget.charge_generation(generation)
                        selection = generation['generated_text'].strip()
                    else:
                        selection = directive['action'].strip()
                    action = resolve_action(selection, view['admissible_commands'], choices)
                    if action not in view['admissible_commands']:
                        raise CandidateExecutionError(f'action {selection!r} is not admissible')
                except CandidateExecutionError as error:
                    result.candidate_error = str(error)
                    break
                before = public_view
                if budget is not None:
                    budget.charge_step()
                view = env.step(action)
                _, result.progress = task_progress(task, view['facts'])
                result.success = bool(view['won'] and goal_reached(task.goal_atoms, view['facts']))
                result.trace.append({'view': before, 'action': action,
                                     'observation': view['observation'],
                                     'generation': generation, 'progress': result.progress,
                                     'done': view['done'], 'success': result.success})
                history.append({'action': action, 'observation': replace_native_objective(
                    view['observation'], task.instruction)})
                if result.success or view['done']:
                    break
    return result


def evaluate_paired(config, parent, child, model, tasks, checkpoints, seeds, budget=None):
    """Evaluate every specified task/checkpoint/seed on fresh parent and child runs."""
    episodes = {'parent': [], 'child': []}
    for task in tasks:
        for checkpoint in checkpoints[task.task_id]:
            if checkpoint == 0:
                raise ValueError('initial-state evaluations are not hidden capability checkpoints')
            for seed in seeds:
                for name, service in (('parent', parent), ('child', child)):
                    if budget is None:
                        episode = run_episode(config, service, model, task, seed, checkpoint)
                    else:
                        episode = run_episode(config, service, model, task, seed, checkpoint, budget=budget)
                    episodes[name].append(episode)
    if any(e.candidate_error for group in episodes.values() for e in group):
        return None, episodes
    outcomes = {name: [CheckpointOutcome(x.task_id, x.checkpoint, x.seed, x.success)
                       for x in values] for name, values in episodes.items()}
    return compare(outcomes['parent'], outcomes['child']), episodes

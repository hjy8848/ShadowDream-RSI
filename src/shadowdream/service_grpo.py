"""On-policy direct-action GRPO with the same token constraints as collection.

This uses the project's Transformers/PEFT actor. It does not use VeRL's default
replay path, whose unconstrained actor log-probability
recomputation cannot represent the legal-action trie used for collection.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path

from .host import run_episode
from .budget import Budget, BudgetExhausted


def action_token_masks(tokenizer, actions, response):
    """Return legal next-token IDs for every token of a completed action."""
    eos = tokenizer.eos_token_id
    if eos is None or not actions or not response:
        raise ValueError('legal actions, response tokens and EOS are required')
    choices = []
    for action in actions:
        tokens = tokenizer.encode(action, add_special_tokens=False)
        if not tokens or tokenizer.decode(tokens, skip_special_tokens=True) != action:
            raise ValueError(f'action does not round-trip through tokenizer: {action!r}')
        choices.append(tokens + [eos])
    masks, prefix = [], []
    for target in response:
        allowed = sorted({choice[len(prefix)] for choice in choices
                          if len(choice) > len(prefix) and choice[:len(prefix)] == prefix})
        if target not in allowed:
            raise ValueError('rollout token violates its legal-action trie')
        masks.append(allowed)
        prefix.append(target)
    if prefix not in choices:
        raise ValueError('rollout did not finish an allowed action including EOS')
    return masks


def constrained_logprobs(session, step, temperature):
    """Recompute each sampled token log probability under the legal-action trie."""
    import torch

    generation = step['generation']
    prompt_ids = generation['prompt_ids']
    response = generation['generated_ids']
    if not prompt_ids or not response:
        raise ValueError('rollout has empty prompt or response')
    masks = action_token_masks(session.tokenizer, step['view']['admissible_commands'], response)
    device = next(session.model.parameters()).device
    tokens = torch.tensor([prompt_ids + response], dtype=torch.long, device=device)
    logits = session.model(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits[0]
    logprobs = []
    for index, (target, allowed) in enumerate(zip(response, masks)):
        token_logits = logits[len(prompt_ids) - 1 + index, allowed].float() / temperature
        logprobs.append(torch.log_softmax(token_logits, dim=-1)[allowed.index(target)])
    return torch.stack(logprobs)


def masked_replay(groups, tokenizer, snapshot, progress_weight, invalid_penalty):
    """Materialize on-policy trajectories with per-token legal sets for trainer integration."""
    replay = []
    for group in groups:
        episodes = []
        for episode in group:
            steps = []
            for step in episode.trace:
                generation = step['generation']
                response = generation['generated_ids']
                if len(response) != len(generation['behavior_logprobs']):
                    raise ValueError('behavior token and logprob counts differ')
                steps.append({'prompt_ids': generation['prompt_ids'], 'response_ids': response,
                    'behavior_logprobs': generation['behavior_logprobs'],
                    'allowed_token_ids': action_token_masks(tokenizer,
                        step['view']['admissible_commands'], response),
                    'model_revision': generation['model_revision'],
                    'adapter_version': generation['adapter_version'],
                    'sampling_parameters': generation['sampling_parameters']})
            episodes.append({'task_id': episode.task_id, 'seed': episode.seed,
                             'reward': reward_components(episode, progress_weight, invalid_penalty),
                             'steps': steps})
        replay.append({'task_id': group[0].task_id, 'service_snapshot': snapshot,
                       'episodes': episodes})
    return replay


def collect(config, session, tasks, seeds, budget=None):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('GRPO requires at least two distinct rollout seeds per task')
    if config['model']['temperature'] <= 0 or config['model']['top_p'] != 1.0:
        raise ValueError('Service GRPO requires temperature > 0 and top_p = 1')
    groups = []
    for task in tasks:
        group = []
        try:
            for seed in seeds:
                group.append(run_episode(config, None, session, task, seed,
                                         action_mode='direct', budget=budget))
        except BudgetExhausted:
            # A partial group has no valid within-task relative advantages.
            return groups, True
        groups.append(group)
    return groups, False


def reward_components(episode, progress_weight, invalid_penalty):
    success = float(episode.success)
    progress = progress_weight * episode.progress
    invalid = invalid_penalty * bool(episode.candidate_error)
    return {'success': success, 'progress': progress, 'invalid_penalty': invalid,
            'total': success + progress - invalid}


def update(session, groups, temperature, learning_rate, clip=0.2, tolerance=1e-3,
           progress_weight=0.0, invalid_penalty=1.0):
    """One on-policy update. Verify behavior logprobs before computing gradients."""
    import torch

    if not groups or any(len(group) < 2 for group in groups):
        raise ValueError('GRPO requires nonempty groups with at least two rollouts')
    trainable = [parameter for parameter in session.model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError('Service model has no trainable parameters')
    session.model.eval()
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    optimizer.zero_grad()
    active_groups = [group for group in groups if any(episode.trace for episode in group)]
    if not active_groups:
        raise ValueError('rollouts have no sampled model actions')
    total_loss = 0.0
    maximum_error = 0.0
    for group in active_groups:
        rewards = torch.tensor([reward_components(e, progress_weight, invalid_penalty)['total']
                                for e in group], dtype=torch.float32)
        advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
        for episode, advantage in zip(group, advantages.tolist()):
            for step in episode.trace:
                with torch.set_grad_enabled(advantage != 0.0):
                    current = constrained_logprobs(session, step, temperature)
                behavior = torch.tensor(step['generation']['behavior_logprobs'],
                                        dtype=current.dtype, device=current.device)
                if current.shape != behavior.shape:
                    raise ValueError('behavior token and logprob counts differ')
                error = (current.detach() - behavior).abs().max().item()
                if not math.isfinite(error):
                    raise ValueError('non-finite behavior or recomputed logprobs')
                maximum_error = max(maximum_error, error)
                if error > tolerance:
                    raise ValueError(f'behavior and recomputed constrained logprobs differ by {error:.6g}')
                ratio = torch.exp(current - behavior)
                clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
                loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
                loss = loss / (len(episode.trace) * len(group) * len(active_groups))
                total_loss += float(loss.detach())
                # Accumulate per-action gradients; the optimizer steps only after every
                # behavior probability has passed validation.
                if advantage != 0.0:
                    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
    if not math.isfinite(gradient_norm):
        raise ValueError('non-finite Service gradient; update was not applied')
    optimizer.step()
    return {'loss': total_loss, 'gradient_norm': gradient_norm,
            'max_behavior_logprob_error': maximum_error, 'behavior_logprob_tolerance': tolerance,
            'rollouts': sum(len(group) for group in groups),
            'env_steps': sum(len(e.trace) for group in groups for e in group),
            'successes': sum(e.success for group in groups for e in group)}



def train(config, tasks, output_dir, parent_adapter=None, device='cuda:0'):
    import torch
    from .model import ModelSession
    from .environment import seed_everything

    seed_everything(config['run']['seed'], torch)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    from .action_model import ActionModel
    from .snapshot import snapshot_id
    session = ActionModel(ModelSession(config, adapter_path=parent_adapter, trainable=True, device=device))
    parent_config = dict(config, service_adapter=parent_adapter)
    before_snapshot = snapshot_id(session, parent_config)
    limit = config['training'].get('service_env_step_budget')
    budget = Budget(limit) if limit is not None else None
    groups, exhausted = collect(config, session, tasks, config['seeds'], budget)
    if budget is not None:
        (output / 'budget.json').write_text(json.dumps(
            {'limit': limit, **budget.snapshot(), 'discarded_partial_group': exhausted}, indent=2) + '\n')
    progress_weight = config['training'].get('service_progress_weight', 0.0)
    invalid_penalty = config['training'].get('service_invalid_penalty', 1.0)
    (output / 'rollouts.json').write_text(json.dumps(
        [[dict(asdict(e), reward=reward_components(e, progress_weight, invalid_penalty))
          for e in group] for group in groups], ensure_ascii=False, indent=2) + '\n')
    replay = masked_replay(groups, session.tokenizer, before_snapshot, progress_weight, invalid_penalty)
    (output / 'masked_replay.json').write_text(json.dumps(replay, ensure_ascii=False) + '\n')
    metrics = update(session, groups, config['model']['temperature'], config['training']['learning_rate'],
                     progress_weight=config['training'].get('service_progress_weight', 0.0),
                     invalid_penalty=config['training'].get('service_invalid_penalty', 1.0),
                     tolerance=config['training'].get('service_logprob_tolerance', 1e-3))
    session.model.save_pretrained(output / 'adapter')
    session.tokenizer.save_pretrained(output / 'adapter')
    after_snapshot = snapshot_id(session, dict(config, service_adapter=str(output / 'adapter')))
    metrics.update({'before_snapshot': before_snapshot, 'after_snapshot': after_snapshot,
                    'model_revision': session.model_revision, 'parent_adapter': parent_adapter,
                    'seeds': config['seeds'], 'task_ids': [task.task_id for task in tasks],
                    'progress_weight': config['training'].get('service_progress_weight', 0.0),
                    'invalid_penalty': config['training'].get('service_invalid_penalty', 1.0),
                    'budget_exhausted': exhausted,
                    'charged_env_steps': budget.env_steps if budget is not None else metrics['env_steps']})
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics

"""Verified behavior cloning data and Service LoRA SFT."""
import json
import random
from hashlib import sha256
from pathlib import Path

from experiment_rsi.evaluation import direct_action_messages
from experiment_rsi.solver_sft import collect_solver_sft_examples


def behavior_examples(episodes):
    """Create direct-action examples only from host-verified successful episodes."""
    examples = []
    for episode in episodes:
        if not episode['success'] or episode.get('candidate_error'):
            continue
        for index, step in enumerate(episode['trace']):
            view = step['view']
            action = step['action']
            if action not in view['admissible_commands']:
                raise ValueError('recorded action was not admissible')
            examples.append({'task_id': episode['task_id'],
                'step': episode['checkpoint'] + index,
                'checkpoint': episode['checkpoint'],
                'seed': episode['seed'],
                'messages': direct_action_messages(view['objective'], view['observation'],
                    view['history'], view['admissible_commands'], view['remaining_steps']),
                'target_action': action})
    return examples


def examples_from_run(run_dir):
    """Read successful rollouts with node, policy, snapshot and checkpoint provenance."""
    examples = []
    for path in sorted((Path(run_dir) / 'nodes').glob('*.json')):
        node = json.loads(path.read_text(encoding='utf-8'))
        for source, episodes in (('full', node['full']),
                                 ('checkpoint', node.get('shadow_episodes', {}).get('child', []))):
            for example in behavior_examples(episodes):
                example.update({'source_node_id': node['id'], 'source_kind': source,
                                'policy_hash': node['policy_hash'],
                                'service_snapshot': node['service_snapshot']})
                examples.append(example)
    return examples


def write_examples(path, examples):
    if not examples:
        raise ValueError('no successful behavior examples')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        for item in examples:
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')


def witness_examples(config, tasks):
    examples = collect_solver_sft_examples(config, tasks)
    for item in examples:
        item['source_kind'] = 'witness'
    return examples


def action_tokens(tokenizer, example):
    """Build the exact inference prompt and mask it in the training target."""
    prompt = tokenizer.apply_chat_template(example['messages'], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    complete = tokenizer.apply_chat_template(example['messages'] + [
        {'role': 'assistant', 'content': example['target_action']}],
        tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False)
    if complete[:len(prompt)] != prompt or len(complete) <= len(prompt):
        raise ValueError('SFT target does not extend the inference prompt')
    return complete, [-100] * len(prompt) + complete[len(prompt):]


def cap_examples(examples, tokenizer, token_budget, seed):
    """Select complete examples within a reproducible input-token budget."""
    if token_budget <= 0:
        raise ValueError('SFT token budget must be positive')
    order = list(examples)
    random.Random(seed).shuffle(order)
    selected, used = [], 0
    for example in order:
        tokens, _ = action_tokens(tokenizer, example)
        if used + len(tokens) <= token_budget:
            selected.append(example)
            used += len(tokens)
    if not selected:
        raise ValueError('token budget cannot accommodate any complete SFT example')
    return selected, used


def cap_with_config(examples, config, token_budget):
    from transformers import AutoTokenizer
    model = config['model']
    revision = None if model['revision'] == 'local' else model['revision']
    tokenizer = AutoTokenizer.from_pretrained(model['path'], revision=revision)
    return cap_examples(examples, tokenizer, token_budget, config['run']['seed'])


def train_sft(config, dataset_path, output_dir, parent_adapter=None, device='cuda:0'):
    """Train direct-action examples using the same chat template as inference."""
    import torch
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainingArguments, set_seed
    from experiment_rsi.model import ModelSession
    from .snapshot import snapshot_id

    raw_dataset = Path(dataset_path).read_bytes()
    dataset_hash = sha256(raw_dataset).hexdigest()
    examples = [json.loads(line) for line in raw_dataset.decode('utf-8').splitlines() if line.strip()]
    if not examples:
        raise ValueError('Service SFT dataset is empty')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if device.startswith('cuda'):
        torch.cuda.set_device(torch.device(device))
    set_seed(config['run']['seed'])
    session = ModelSession(config, adapter_path=parent_adapter, trainable=True, device=device)
    before_snapshot = snapshot_id(session, dict(config, service_adapter=parent_adapter))
    before_weights = {name: value.detach().cpu().clone() for name, value in session.model.named_parameters()
                      if value.requires_grad}
    if not before_weights:
        raise ValueError('Service SFT model has no trainable parameters')
    tokenizer = session.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError('Service SFT requires a padding or EOS token')

    encoded = [action_tokens(tokenizer, item) for item in examples]
    input_tokens = sum(len(tokens) for tokens, _ in encoded)
    target_tokens = sum(sum(value != -100 for value in labels) for _, labels in encoded)

    class ActionDataset(Dataset):
        def __len__(self):
            return len(encoded)

        def __getitem__(self, index):
            complete, labels = encoded[index]
            return {'input_ids': torch.tensor(complete, dtype=torch.long),
                    'labels': torch.tensor(labels, dtype=torch.long)}

    def collate(rows):
        width = max(len(row['input_ids']) for row in rows)
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            length = len(row['input_ids'])
            ids[index, :length], labels[index, :length] = row['input_ids'], row['labels']
            mask[index, :length] = 1
        return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}

    dtype = config['model']['dtype']
    arguments = TrainingArguments(output_dir=str(output),
        learning_rate=config['training']['learning_rate'],
        num_train_epochs=config['training'].get('sft_epochs', 1),
        per_device_train_batch_size=config['training'].get('sft_batch_size', 1),
        gradient_accumulation_steps=config['training'].get('sft_gradient_accumulation_steps', 4),
        logging_steps=10, save_strategy='no', report_to=[],
        bf16=dtype == 'bfloat16', fp16=dtype == 'float16', remove_unused_columns=False)
    trainer = Trainer(model=session.model, args=arguments, train_dataset=ActionDataset(),
                      data_collator=collate, processing_class=tokenizer)
    result = trainer.train()
    change = sum(float((value.detach().cpu().float() - before_weights[name].float()).abs().sum())
                 for name, value in session.model.named_parameters() if value.requires_grad)
    adapter = output / 'adapter'
    session.model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    after_snapshot = snapshot_id(session, dict(config, service_adapter=str(adapter)))
    metrics = {'train_metrics': result.metrics, 'log_history': trainer.state.log_history,
               'trainable_parameter_l1_change': change, 'parameter_update': change > 0,
               'training_examples': len(examples), 'input_tokens_per_epoch': input_tokens,
               'target_tokens_per_epoch': target_tokens, 'dataset_sha256': dataset_hash,
               'parent_adapter': parent_adapter,
               'before_snapshot': before_snapshot, 'after_snapshot': after_snapshot,
               'adapter': str(adapter.resolve())}
    (output / 'training_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')
    return metrics

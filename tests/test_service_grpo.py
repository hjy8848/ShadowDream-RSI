import unittest
from types import SimpleNamespace
import torch

from shadowdream.service_grpo import constrained_logprobs, update


class Tokenizer:
    eos_token_id = 3
    def encode(self, action, add_special_tokens=False): return {'a': [1], 'b': [2]}[action]
    def decode(self, ids, skip_special_tokens=True): return {1: 'a', 2: 'b'}[ids[0]]


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0., 0.3, -0.2, 0.]))
    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.logits.expand(input_ids.shape[0], input_ids.shape[1], -1))


class ServiceGRPOTest(unittest.TestCase):
    def setUp(self):
        self.session = SimpleNamespace(tokenizer=Tokenizer(), model=TinyPolicy())

    def step(self, action):
        step = {'view': {'admissible_commands': ['a', 'b']},
                'generation': {'prompt_ids': [0], 'generated_ids': [action, 3]}}
        step['generation']['behavior_logprobs'] = constrained_logprobs(self.session, step, 1.0).detach().tolist()
        return step

    def test_on_policy_update_uses_constrained_behavior_and_moves_weights(self):
        groups = [[SimpleNamespace(success=True, progress=0.0, candidate_error=None, trace=[self.step(1)]),
                   SimpleNamespace(success=False, progress=0.0, candidate_error=None, trace=[self.step(2)])]]
        before = self.session.model.logits.detach().clone()
        metrics = update(self.session, groups, 1.0, 0.01)
        self.assertGreater(metrics['gradient_norm'], 0)
        self.assertFalse(torch.equal(before, self.session.model.logits.detach()))
        self.assertLess(metrics['max_behavior_logprob_error'], 1e-6)

    def test_off_policy_logprob_is_rejected(self):
        step = self.step(1)
        step['generation']['behavior_logprobs'][0] += 0.2
        groups = [[SimpleNamespace(success=True, progress=0.0, candidate_error=None, trace=[step]),
                   SimpleNamespace(success=False, progress=0.0, candidate_error=None, trace=[self.step(2)])]]
        with self.assertRaisesRegex(ValueError, 'logprobs differ'):
            update(self.session, groups, 1.0, 0.01)

class RolloutBudgetTest(unittest.TestCase):
    def test_partial_group_is_not_used(self):
        from unittest.mock import patch
        from shadowdream.budget import Budget
        from shadowdream.service_grpo import collect
        config = {'model': {'temperature': 0.7, 'top_p': 1.0}}
        tasks = [SimpleNamespace(task_id='a'), SimpleNamespace(task_id='b')]
        def run(config, service, session, task, seed, action_mode, budget):
            budget.charge_step()
            return SimpleNamespace(success=False, progress=0.0, candidate_error=None, trace=[])
        with patch('shadowdream.service_grpo.run_episode', side_effect=run):
            budget = Budget(3)
            groups, exhausted = collect(config, None, tasks, [1, 2], budget)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)
        self.assertTrue(exhausted)
        self.assertEqual(budget.env_steps, 3)

class ReplayTest(unittest.TestCase):
    def test_records_complete_token_constraints(self):
        from shadowdream.service_grpo import action_token_masks
        masks = action_token_masks(Tokenizer(), ['a', 'b'], [1, 3])
        self.assertEqual(masks, [[1, 2], [3]])
        with self.assertRaisesRegex(ValueError, 'finish'):
            action_token_masks(Tokenizer(), ['a', 'b'], [1])
        with self.assertRaisesRegex(ValueError, 'violates'):
            action_token_masks(Tokenizer(), ['a', 'b'], [0, 3])

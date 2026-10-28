import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shadowdream.budget import Budget, BudgetExhausted
from shadowdream.host import Episode
from shadowdream.runner import run_search
from shadowdream.report import load_run


class Editor:
    def generate(self, messages, **kwargs):
        return {'generated_text': json.dumps({'policy_source': 'def step(view, state): pass'}),
                'prompt_ids': [1], 'generated_ids': [2]}


class BudgetTest(unittest.TestCase):
    def test_limit_stops_without_overcharging(self):
        budget = Budget(1)
        budget.charge_step()
        with self.assertRaises(BudgetExhausted):
            budget.charge_step()
        self.assertEqual(budget.env_steps, 1)

    @patch('shadowdream.runner.evaluate_paired')
    @patch('shadowdream.runner.run_episode')
    def test_incomplete_candidate_not_archived_and_cost_is_counted(self, episode, paired):
        def run(config, service, model, task, seed, budget):
            budget.charge_step()
            return Episode(task_id=task.task_id, checkpoint=0, seed=seed,
                           trace=[{'action': 'a'}])
        def shadow(config, parent, child, model, tasks, checkpoints, seeds, budget):
            budget.require_step()
        episode.side_effect, paired.side_effect = run, shadow
        task = SimpleNamespace(task_id='t', source_scene_id='s', witness_actions=['a', 'b'])
        config = {'editor': {'group_size': 1, 'max_new_tokens': 5, 'temperature': 1, 'top_p': 1},
                  'search': {'c': 1, 'beta': 1, 'env_step_budget': 2},
                  'model': {'temperature': 0, 'top_p': 1}, 'run': {'max_steps': 2}}
        with tempfile.TemporaryDirectory() as directory:
            archive = run_search(config, Editor(), None, {'policy_source': 'p', 'knowledge': []},
                                 [task], {'t': [1]}, [1], directory, 2)
            event = json.loads((Path(directory) / 'events/000000.json').read_text())
            curve = load_run(directory)['curve']
        self.assertEqual(len(archive.nodes), 1)
        self.assertEqual(event['status'], 'budget_exhausted')
        self.assertEqual(curve[-1]['env_steps'], 2)

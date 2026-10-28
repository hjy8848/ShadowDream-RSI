import unittest
from types import SimpleNamespace
from unittest.mock import patch

from shadowdream.host import run_episode


class Environment:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def reset(self): return {'observation': 'start', 'admissible_commands': ['advance'], 'facts': []}
    def step(self, action):
        if action == 'advance':
            return {'observation': 'middle', 'admissible_commands': ['finish'], 'facts': [], 'done': False, 'won': False}
        return {'observation': 'end', 'admissible_commands': [], 'facts': [{'predicate': 'done', 'arguments': []}], 'done': True, 'won': True}


class Worker:
    seen = None
    def __init__(self, *args): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def step(self, view, state):
        Worker.seen = view, state.copy()
        return {'kind': 'action', 'action': 'finish'}, state


class HostTest(unittest.TestCase):
    @patch('shadowdream.host.PolicyWorker', Worker)
    @patch('shadowdream.host.AlfworldEpisode', lambda *args: Environment())
    @patch('shadowdream.host.task_progress', lambda task, facts: (0, 0.0))
    def test_checkpoint_gives_only_public_history_and_cold_state(self):
        task = SimpleNamespace(task_id='t', task_game_file='x', source_split='train',
             goal_atoms=[{'predicate': 'done', 'arguments': []}], witness_actions=['advance', 'finish'], instruction='finish')
        config = {'alfworld': {'config_path': 'x'}, 'worker': {}, 'run': {'max_steps': 3}}
        result = run_episode(config, {'knowledge': [], 'policy_source': ''}, None, task, 1, 1)
        self.assertTrue(result.success)
        view, state = Worker.seen
        self.assertEqual(state, {})
        self.assertEqual(view['history'], [{'action': 'advance', 'observation': 'middle'}])
        self.assertNotIn('facts', view)
        self.assertNotIn('witness_actions', view)
        self.assertNotIn('finish', str(view['history']))
        self.assertEqual(result.prefix_steps, 1)

class DirectModel:
    def generate(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return {'generated_text': 'finish', 'generated_ids': [1]}


class DirectTest(unittest.TestCase):
    @patch('shadowdream.host.AlfworldEpisode', lambda *args: Environment())
    @patch('shadowdream.host.task_progress', lambda task, facts: (0, 0.0))
    def test_direct_mode_uses_shared_action_interface(self):
        task = SimpleNamespace(task_id='t', task_game_file='x', source_split='train',
             goal_atoms=[{'predicate': 'done', 'arguments': []}], witness_actions=['advance', 'finish'], instruction='finish')
        config = {'alfworld': {'config_path': 'x'}, 'run': {'max_steps': 3},
                  'model': {'max_new_tokens': 5, 'temperature': 0, 'top_p': 1}}
        model = DirectModel()
        result = run_episode(config, None, model, task, 1, 1, 'direct')
        self.assertTrue(result.success)
        self.assertEqual(model.kwargs['allowed_texts'], ['finish'])
        self.assertFalse(model.kwargs['enable_thinking'])
        self.assertNotIn('facts', str(model.messages))

class PairFailureTest(unittest.TestCase):
    def test_invalid_candidate_is_not_counted_as_a_shadow_outcome(self):
        from shadowdream.host import Episode, evaluate_paired
        parent, child = object(), object()
        def run(config, service, model, task, seed, checkpoint):
            return Episode(task_id=task.task_id, checkpoint=checkpoint, seed=seed,
                           candidate_error='bad policy' if service is child else None)
        task = SimpleNamespace(task_id='t')
        with patch('shadowdream.host.run_episode', side_effect=run):
            shadow, episodes = evaluate_paired({}, parent, child, None, [task], {'t': [1]}, [7])
        self.assertIsNone(shadow)
        self.assertEqual(episodes['child'][0].candidate_error, 'bad policy')

class LongEnvironment:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def reset(self): return {'observation': 'initial', 'admissible_commands': ['advance'], 'facts': []}
    def step(self, action):
        if action == 'advance':
            return {'observation': 'middle', 'admissible_commands': ['advance', 'finish'], 'facts': [], 'done': False, 'won': False}
        return {'observation': 'end', 'admissible_commands': [], 'facts': [{'predicate': 'done', 'arguments': []}], 'done': True, 'won': True}


class ProtocolTest(unittest.TestCase):
    @patch('shadowdream.host.PolicyWorker', Worker)
    @patch('shadowdream.host.AlfworldEpisode', lambda *args: LongEnvironment())
    @patch('shadowdream.host.task_progress', lambda task, facts: (0, 0.0))
    def test_history_protocol_provides_full_public_prefix_not_future(self):
        task = SimpleNamespace(task_id='t', task_game_file='x', source_split='train',
            goal_atoms=[{'predicate': 'done', 'arguments': []}],
            witness_actions=['advance'] * 7 + ['finish'], instruction='finish')
        config = {'alfworld': {'config_path': 'x'}, 'worker': {}, 'run': {'max_steps': 10},
                  'shadow': {'protocol': 'history'}}
        result = run_episode(config, {'knowledge': [], 'policy_source': ''}, None, task, 1, 7)
        view, state = Worker.seen
        self.assertTrue(result.success)
        self.assertEqual(result.checkpoint_protocol, 'history')
        self.assertEqual(len(view['history']), 7)
        self.assertEqual(view['initial_observation'], 'initial')
        self.assertEqual(state, {})
        self.assertNotIn('facts', view)
        self.assertNotIn('finish', str(view['history']))

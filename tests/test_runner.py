import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from shadowdream.host import Episode
from shadowdream.runner import run_search
from shadowdream.shadow import ShadowResult


class Editor:
    def __init__(self): self.messages = []
    def generate(self, messages, **kwargs):
        self.messages.append(messages)
        return {'generated_text': json.dumps({'policy_source': 'def step(view,state): pass'}),
                'prompt_ids': [1], 'generated_ids': [2]}


class RunnerTest(unittest.TestCase):
    @patch('shadowdream.runner.evaluate_paired')
    @patch('shadowdream.runner._full')
    def test_editor_receives_full_and_shadow_diagnostics(self, full, paired):
        root = Episode(task_id='t', checkpoint=0, seed=1, success=False)
        child = Episode(task_id='t', checkpoint=0, seed=1, success=True)
        full.side_effect = [[root], [child], [child]]
        parent_shadow = Episode(task_id='t', checkpoint=1, seed=1, success=False)
        child_shadow = Episode(task_id='t', checkpoint=1, seed=1, success=True)
        paired.return_value = (ShadowResult((('t', 1, 1.0),), 1.0, 0.0),
                               {'parent': [parent_shadow], 'child': [child_shadow]})
        config = {'editor': {'group_size': 1, 'max_new_tokens': 5, 'temperature': 1, 'top_p': 1},
                  'search': {'c': 1, 'beta': 1}, 'run': {'max_steps': 3}}
        task = SimpleNamespace(task_id='t', source_scene_id='s', witness_actions=['a', 'b'])
        editor = Editor()
        with tempfile.TemporaryDirectory() as root_dir:
            run_search(config, editor, None, {'policy_source': 'def step(view,state): pass', 'knowledge': []},
                       [task], {'t': [1]}, [1], root_dir, 2, method='greedy')
        diagnostics = json.loads(editor.messages[1][1]['content'])['diagnostics']
        self.assertEqual(diagnostics['shadow']['gain'], 1.0)
        self.assertTrue(diagnostics['full'][0]['success'])

    @patch('shadowdream.runner._full')
    def test_editor_infrastructure_failure_is_not_a_candidate_reward(self, full):
        from pathlib import Path
        full.return_value = [Episode(task_id='t', checkpoint=0, seed=1)]
        class FailedEditor:
            def generate(self, *args, **kwargs):
                raise RuntimeError('model unavailable')
        config = {'editor': {'group_size': 1, 'max_new_tokens': 5, 'temperature': 1, 'top_p': 1},
                  'search': {'c': 1, 'beta': 1}, 'run': {'max_steps': 3}}
        task = SimpleNamespace(task_id='t', source_scene_id='s', witness_actions=['a', 'b'])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'model unavailable'):
                run_search(config, FailedEditor(), None, {'policy_source': 'p', 'knowledge': []},
                           [task], {'t': [1]}, [1], directory, 1)
            failure = json.loads((Path(directory) / 'failure.json').read_text())
            self.assertEqual(failure['phase'], 'editor')
            self.assertFalse((Path(directory) / 'events').exists())

    def test_task_fingerprint_changes_with_game_or_witness(self):
        from pathlib import Path
        from shadowdream.runner import task_fingerprint
        with tempfile.TemporaryDirectory() as directory:
            game = Path(directory) / 'game'
            game.write_text('first')
            task = SimpleNamespace(task_id='t', task_game_file=str(game), witness_actions=['a'])
            before = task_fingerprint(task)
            game.write_text('second')
            self.assertNotEqual(before, task_fingerprint(task))
            before = task_fingerprint(task)
            task.witness_actions.append('b')
            self.assertNotEqual(before, task_fingerprint(task))

    def test_editor_accepts_python_and_existing_json(self):
        from shadowdream.runner import parse_edit
        source = 'def step(view, state):\n    return {"kind": "action", "action": "look", "state": state}'
        self.assertEqual(parse_edit(source, [])['policy_source'], source)
        self.assertEqual(parse_edit('```python\n' + source + '\n```', [])['policy_source'], source)
        self.assertEqual(parse_edit(json.dumps({'policy_source': source}), [])['policy_source'], source)
        with self.assertRaisesRegex(ValueError, 'synchronous'):
            parse_edit('async def step(view, state): pass', [])

    def test_candidate_failure_categories(self):
        from shadowdream.runner import candidate_error_category
        self.assertEqual(candidate_error_category("action 'x' is not admissible"), 'illegal_action')
        self.assertEqual(candidate_error_category('policy exceeded 10 seconds'), 'timeout')
        self.assertEqual(candidate_error_category('NameError: x'), 'runtime')

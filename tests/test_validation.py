import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from shadowdream.validation import select, test_selected, check_disjoint


@dataclass
class Episode:
    success: bool


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / 'nodes').mkdir()
        (root / 'manifest.json').write_text(json.dumps({'task_ids': ['train'], 'scene_ids': ['scene-a']}))
        for name, value in [('root', 'old-policy'), ('child', 'new-policy')]:
            (root / 'nodes' / (name + '.json')).write_text(json.dumps({'id': name, 'service': {'name': value}}))
        self.tasks = [SimpleNamespace(task_id='valid', source_scene_id='scene-b')]

    @patch('shadowdream.validation.run_episode')
    def test_selection_rechecks_and_preserves_incumbent_on_tie(self, run):
        run.side_effect = lambda config, service, model, task, seed, action_mode: Episode(
            success=service['name'] in {'old', 'new-policy'})
        cfg = {'run': {'max_steps': 5}, 'model': {'temperature': 0, 'top_p': 1}}
        incumbent = (cfg, None, 'old-model', {'name': 'old'})
        result = select(cfg, None, 'new-model', self.temp.name, self.tasks, [1], incumbent)
        self.assertEqual(result['selected']['snapshot'], 'old-model')
        self.assertTrue(result['selected']['incumbent'])
        self.assertEqual(run.call_count, 3)
        heldout = [SimpleNamespace(task_id='test', source_scene_id='scene-c')]
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            test_selected({}, None, 'new-model', result, heldout, [1])
        output = test_selected({}, None, 'old-model', result, heldout, [1], 'direct')
        self.assertEqual(output['action_mode'], 'direct')
        self.assertEqual(output['score'], 1.0)

    def test_prevents_scene_leakage_even_for_different_task_ids(self):
        with self.assertRaisesRegex(ValueError, 'scene_ids'):
            check_disjoint({'task_ids': ['one'], 'scene_ids': ['same']},
                           {'task_ids': ['two'], 'scene_ids': ['same']})

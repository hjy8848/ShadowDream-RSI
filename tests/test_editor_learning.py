import unittest
from types import SimpleNamespace
import torch

from shadowdream.editor_learning import verify_behavior


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor([0.0, 1.0, -1.0]))
    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.weights.expand(1, input_ids.shape[1], -1))


class EditorBehaviorTest(unittest.TestCase):
    def test_detects_off_policy_editor_logprobs(self):
        session = SimpleNamespace(model=Policy())
        expected = torch.log_softmax(session.model.weights / 0.7, dim=-1)[1].item()
        sample = {'prompt_ids': [0], 'generated_ids': [1],
                  'behavior_logprobs': [expected], 'sampling_parameters': {'temperature': 0.7}}
        self.assertLess(verify_behavior(session, [sample]), 1e-6)
        sample['behavior_logprobs'][0] += 0.2
        with self.assertRaisesRegex(ValueError, 'logprobs differ'):
            verify_behavior(session, [sample])

class EditorGroupTest(unittest.TestCase):
    def test_budget_cut_group_is_excluded(self):
        import json
        import tempfile
        from pathlib import Path
        from shadowdream.editor_learning import editor_samples
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'events').mkdir()
            (root / 'manifest.json').write_text(json.dumps({'config': {'editor': {'group_size': 2}}}))
            for index, status in enumerate(('invalid', 'budget_exhausted')):
                (root / 'events' / f'{index}.json').write_text(json.dumps({'group_id': 'g', 'status': status}))
            with self.assertRaisesRegex(ValueError, 'no complete'):
                editor_samples(root)

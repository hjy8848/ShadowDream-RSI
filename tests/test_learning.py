import unittest
from shadowdream.service_learning import behavior_examples


class LearningTest(unittest.TestCase):
    def test_only_verified_successful_legal_actions_become_examples(self):
        view = {'objective': 'put', 'observation': 'room', 'history': [],
                'admissible_commands': ['go'], 'remaining_steps': 4}
        episode = {'task_id': 't', 'checkpoint': 2, 'seed': 1, 'success': True,
                   'candidate_error': None, 'trace': [{'view': view, 'action': 'go'}]}
        failure = dict(episode, success=False)
        examples = behavior_examples([episode, failure])
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]['target_action'], 'go')
        self.assertEqual(examples[0]['step'], 2)
        self.assertNotIn('witness', str(examples))

class TokenizationTest(unittest.TestCase):
    def test_sft_masks_exact_inference_prompt(self):
        from shadowdream.service_learning import action_tokens
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                assert kwargs['return_dict'] is False
                assert kwargs['enable_thinking'] is False
                return [1, 2] if kwargs['add_generation_prompt'] else [1, 2, 3, 4]
        tokens, labels = action_tokens(Tokenizer(), {'messages': [], 'target_action': 'action'})
        self.assertEqual(tokens, [1, 2, 3, 4])
        self.assertEqual(labels, [-100, -100, 3, 4])

class ProvenanceTest(unittest.TestCase):
    def test_program_examples_retain_node_and_snapshot(self):
        import json
        import tempfile
        from pathlib import Path
        from shadowdream.service_learning import examples_from_run
        view = {'objective': 'put', 'observation': 'room', 'history': [],
                'admissible_commands': ['go'], 'remaining_steps': 4}
        episode = {'task_id': 't', 'checkpoint': 0, 'seed': 1, 'success': True,
                   'candidate_error': None, 'trace': [{'view': view, 'action': 'go'}]}
        node = {'id': 'root', 'policy_hash': 'p', 'service_snapshot': 's', 'full': [episode]}
        with tempfile.TemporaryDirectory() as directory:
            nodes = Path(directory) / 'nodes'
            nodes.mkdir()
            (nodes / 'root.json').write_text(json.dumps(node))
            example = examples_from_run(directory)[0]
        self.assertEqual((example['source_node_id'], example['policy_hash'], example['service_snapshot']),
                         ('root', 'p', 's'))
        self.assertEqual(example['source_kind'], 'full')

class TokenBudgetTest(unittest.TestCase):
    def test_caps_complete_examples_deterministically(self):
        from shadowdream.service_learning import cap_examples
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2] if kwargs['add_generation_prompt'] else [1, 2, 3]
        examples = [{'messages': [], 'target_action': str(i)} for i in range(4)]
        chosen, used = cap_examples(examples, Tokenizer(), 7, 42)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(used, 6)
        self.assertEqual(chosen, cap_examples(examples, Tokenizer(), 7, 42)[0])
        with self.assertRaisesRegex(ValueError, 'cannot accommodate'):
            cap_examples(examples, Tokenizer(), 2, 42)

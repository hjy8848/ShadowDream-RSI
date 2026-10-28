import asyncio
import json
import unittest

from shadowdream.service_verl_data import rows_from_masked_replay


class Tokenizer:
    def decode(self, tokens, skip_special_tokens=False): return 'prompt'


class ReplayTest(unittest.TestCase):
    def fixture(self):
        step = {'prompt_ids': [1], 'response_ids': [2, 3], 'behavior_logprobs': [-.4, 0.],
                'allowed_token_ids': [[2, 4], [3]]}
        episodes = [{'task_id': 't', 'seed': seed, 'reward': {'total': reward}, 'steps': [step]}
                    for seed, reward in ((1, 1.), (2, 0.))]
        return [{'task_id': 't', 'service_snapshot': 'snapshot', 'episodes': episodes}]

    def test_replay_preserves_masks_and_group(self):
        row = rows_from_masked_replay(self.fixture(), Tokenizer())[0]
        self.assertEqual(row['__rollout_n__'], 2)
        calls = json.loads(row['replay_trajectories'])
        self.assertEqual(calls[0][0]['allowed_token_ids'], [[2, 4], [3]])
        self.assertEqual(calls[0][0]['sample_weight'], 1.0)
        from shadowdream.verl_service_agent import ServiceReplayAgentLoop
        loop = ServiceReplayAgentLoop.__new__(ServiceReplayAgentLoop)
        output = asyncio.run(loop.run({}, replay_trajectories=row['replay_trajectories'], session_id='0'))
        self.assertEqual(output[0].extra_fields['allowed_token_ids'], [[2, 4], [3]])

    def test_invalid_masks_are_rejected(self):
        data = self.fixture()
        data[0]['episodes'][0]['steps'][0]['allowed_token_ids'] = [[4], [3]]
        with self.assertRaisesRegex(ValueError, 'outside'):
            rows_from_masked_replay(data, Tokenizer())

    def test_parquet_roundtrip_preserves_masks(self):
        import tempfile
        from pathlib import Path
        import pyarrow.parquet as pq
        from shadowdream.service_verl_data import export_parquet
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'replay.json'
            target = Path(directory) / 'replay.parquet'
            source.write_text(json.dumps(self.fixture()))
            self.assertEqual(export_parquet(source, target, Tokenizer()), 1)
            row = pq.read_table(target).to_pylist()[0]
            calls = json.loads(row['replay_trajectories'])
            self.assertEqual(calls[1][0]['allowed_token_ids'], [[2, 4], [3]])
            self.assertEqual(row['extra_info']['service_snapshot'], 'snapshot')

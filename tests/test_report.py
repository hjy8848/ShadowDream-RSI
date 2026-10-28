import unittest
from shadowdream.report import compare_at_budget


class ReportTest(unittest.TestCase):
    def test_budget_includes_only_completed_evaluations(self):
        run = {'method': 'shadow_ucb', 'feedback': 'improvement', 'task_ids': ['t'],
               'scene_ids': ['s'], 'seeds': [1], 'comparison': {}, 'curve': [
                   {'env_steps': 5, 'best_score': 0.1},
                   {'env_steps': 12, 'best_score': 0.5},
                   {'env_steps': 20, 'best_score': 0.8}]}
        result = compare_at_budget([run], 'env_steps', 15)['runs'][0]
        self.assertEqual(result['best_score'], 0.5)
        self.assertEqual(result['used'], 12)
        self.assertTrue(result['run_reached_budget'])
        self.assertIsNone(compare_at_budget([run], 'env_steps', 2)['runs'][0]['best_score'])

    def test_paired_replicates_require_matched_seeds_and_budget(self):
        from shadowdream.report import paired_replicates
        def run(method, seed, value):
            return {'method': method, 'feedback': 'absolute', 'rng_seed': seed,
                    'task_ids': ['t'], 'scene_ids': ['s'], 'seeds': [1], 'comparison': {},
                    'curve': [{'editor_calls': 2, 'best_score': value}]}
        runs = [run('ucb', 1, .2), run('shadow_ucb', 1, .3),
                run('ucb', 2, .4), run('shadow_ucb', 2, .6)]
        result = paired_replicates(runs, 'editor_calls', 2, 'ucb', 'absolute',
                                   'shadow_ucb', 'absolute', 100)
        self.assertAlmostEqual(result['mean_difference'], .15)
        self.assertEqual(len(result['paired_runs']), 2)
        with self.assertRaisesRegex(ValueError, 'matching'):
            paired_replicates(runs[:-1], 'editor_calls', 2, 'ucb', 'absolute',
                              'shadow_ucb', 'absolute', 100)

    def test_mismatched_protocol_is_rejected(self):
        first = {'method': 'ucb', 'feedback': 'absolute', 'task_ids': ['t'],
                 'scene_ids': ['s'], 'seeds': [1], 'comparison': {'shadow_protocol': 'cold-start'},
                 'curve': [{'env_steps': 10, 'best_score': 0.1}]}
        second = dict(first, method='shadow_ucb', comparison={'shadow_protocol': 'history'})
        with self.assertRaisesRegex(ValueError, 'comparison'):
            compare_at_budget([first, second], 'env_steps', 10)

    def test_replicates_cannot_mix_task_sets(self):
        from shadowdream.report import paired_replicates
        def run(method, seed, task):
            return {'method': method, 'feedback': 'absolute', 'rng_seed': seed,
                    'task_ids': [task], 'scene_ids': ['s'], 'seeds': [1], 'comparison': {},
                    'curve': [{'editor_calls': 2, 'best_score': 0.2}]}
        runs = [run('ucb', 1, 't'), run('shadow', 1, 't'),
                run('ucb', 2, 'different'), run('shadow', 2, 'different')]
        with self.assertRaisesRegex(ValueError, 'task_ids'):
            paired_replicates(runs, 'editor_calls', 2, 'ucb', 'absolute', 'shadow', 'absolute', 100)

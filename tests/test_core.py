import unittest
from shadowdream.shadow import CheckpointOutcome, compare
from shadowdream.search import Archive, Node, Search

class ShadowTest(unittest.TestCase):
    def test_gain_and_regression(self):
        parent = [CheckpointOutcome('t', 1, 1, False), CheckpointOutcome('t', 2, 1, True)]
        child = [CheckpointOutcome('t', 1, 1, True), CheckpointOutcome('t', 2, 1, False)]
        result = compare(parent, child)
        self.assertEqual(result.vector, (('t', 1, 1.0), ('t', 2, -1.0)))
        self.assertEqual((result.gain, result.regression), (0.5, 0.5))

    def test_requires_matched_seed(self):
        with self.assertRaises(ValueError):
            compare([CheckpointOutcome('t', 1, 1, False)], [CheckpointOutcome('t', 1, 2, True)])

class SearchTest(unittest.TestCase):
    def test_distinct_feedback(self):
        for mode, expected in [('absolute', 0.8), ('improvement', 0.1)]:
            archive = Archive()
            root = Node('root', None, 0.7)
            archive.add_root(root)
            search = Search(archive, mode)
            event = search.add_child(Node('child', 'root', 0.8, shadow_gain=0.2))
            self.assertAlmostEqual(root.feedback_sum, expected)
            self.assertAlmostEqual(event['previous_best'], 0.7)
            self.assertAlmostEqual(event['improvement'], 0.1)
            event = search.add_child(Node('next', 'root', 0.75))
            self.assertEqual(event['improvement'], 0.0)
            self.assertEqual(archive.best_score, 0.8)

    def test_unvisited_and_ancestor_backup(self):
        archive = Archive()
        root = Node('root', None, 0.3)
        archive.add_root(root)
        search = Search(archive, 'absolute')
        self.assertIs(search.select(), root)
        child = Node('child', 'root', 0.4)
        search.add_child(child)
        self.assertIs(search.select(), root)  # real feedback can outweigh a new node
        search.add_child(Node('grandchild', 'child', 0.5))
        self.assertEqual((root.visits, child.visits), (2, 1))
        self.assertAlmostEqual(root.feedback_sum, 0.9)

    def test_alternative_progress_prior_does_not_overwrite_shadow(self):
        archive = Archive()
        root = Node('root', None, 0.2, shadow_gain=0.8, visits=1)
        child = Node('child', 'root', 0.2, shadow_gain=0.1, visits=1)
        archive.add_root(root)
        archive.nodes[child.id] = child
        search = Search(archive, 'absolute', c=0, beta=1)
        self.assertIs(search.select(), root)
        self.assertIs(search.select(priors={'root': 0.0, 'child': 0.9}), child)
        self.assertEqual((root.shadow_gain, child.shadow_gain), (0.8, 0.1))

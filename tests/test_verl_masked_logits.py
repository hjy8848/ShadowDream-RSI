import unittest
import torch
from shadowdream.verl_masked_logits import masked_jagged_logprobs


class MaskedLogprobTest(unittest.TestCase):
    def test_masked_actor_logprobs_and_gradients(self):
        logits = torch.tensor([[[0., 0., 0., 0., 0.], [0., 0., 1., 0., -1.],
                                [0., 0., 0., 2., 0.], [0., 0., 0., 0., 0.]]],
                              requires_grad=True)
        offsets = torch.tensor([0, 4])
        ids = torch.nested.nested_tensor_from_jagged(torch.tensor([0, 1, 2, 3]), offsets)
        original = torch.nested.nested_tensor_from_jagged(torch.zeros(4), offsets)
        result = masked_jagged_logprobs(logits, ids, original, [[[2, 4], [3]]], [1.])
        expected = torch.log_softmax(torch.tensor([1., -1.]), dim=0)[0]
        self.assertTrue(torch.allclose(result.values()[1], expected))
        self.assertEqual(result.values()[2].item(), 0.)
        result.values()[1].backward()
        self.assertGreater(logits.grad[0, 1, 2].item(), 0)
        self.assertLess(logits.grad[0, 1, 4].item(), 0)

    def test_illegal_sample_is_rejected(self):
        offsets = torch.tensor([0, 2])
        ids = torch.nested.nested_tensor_from_jagged(torch.tensor([0, 1]), offsets)
        logps = torch.nested.nested_tensor_from_jagged(torch.zeros(2), offsets)
        with self.assertRaisesRegex(ValueError, 'legal token'):
            masked_jagged_logprobs(torch.zeros(1, 2, 3), ids, logps, [[[2]]], [1.])

    def test_engine_hook_uses_recorded_masks(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from shadowdream.verl_service_hook import prepare_masked
        offsets = torch.tensor([0, 4])
        ids = torch.nested.nested_tensor_from_jagged(torch.tensor([0, 1, 2, 3]), offsets)
        logps = torch.nested.nested_tensor_from_jagged(torch.zeros(4), offsets)
        logits = torch.randn(1, 4, 5, requires_grad=True)
        class Field:
            def tolist(self): return [{'allowed_token_ids': [[2, 4], [3]]}]
        batch = {'input_ids': ids, 'extra_fields': Field()}
        def options(data, key, default):
            return {'use_remove_padding': False, 'use_fused_kernels': False,
                    'calculate_entropy': False}.get(key, default)
        with patch('verl.utils.tensordict_utils.get_non_tensor_data', side_effect=options):
            result = prepare_masked(lambda *args: {'log_probs': logps}, None,
                SimpleNamespace(logits=logits), {'temperature': torch.tensor([1.0])}, batch, None)
        result['log_probs'].values()[1].backward()
        self.assertNotEqual(logits.grad[0, 1, 2].item(), 0.)

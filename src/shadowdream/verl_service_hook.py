"""VeRL FSDP hook for Service replay with legal-token constrained logprobs.

Loaded only for the Service actor through VeRL's `model.external_lib`. This
requires the non-fused, non-remove-padding FSDP path and complete replay masks.
"""
from .verl_masked_logits import masked_jagged_logprobs


def prepare_masked(original, engine, output, output_args, micro_batch, logits_processor_func):
    import torch
    from verl.utils import tensordict_utils as tu

    if tu.get_non_tensor_data(data=micro_batch, key='use_remove_padding', default=True):
        raise ValueError('Service replay requires use_remove_padding=False')
    if tu.get_non_tensor_data(data=micro_batch, key='use_fused_kernels', default=False):
        raise ValueError('Service replay requires non-fused actor logits')
    if tu.get_non_tensor_data(data=micro_batch, key='calculate_entropy', default=False):
        raise ValueError('masked Service entropy is not implemented')
    fields = micro_batch['extra_fields'].tolist()
    masks = [field['allowed_token_ids'] for field in fields]
    result = original(engine, output, output_args, micro_batch, logits_processor_func)
    temperatures = output_args['temperature']
    if torch.is_tensor(temperatures):
        temperatures = temperatures.detach().cpu().tolist()
    logits = output.logits
    from torch.distributed.tensor import DTensor
    if isinstance(logits, DTensor):
        logits = logits.full_tensor()
    result['log_probs'] = masked_jagged_logprobs(logits, micro_batch['input_ids'],
                                                 result['log_probs'], masks, temperatures)
    return result


def install():
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    original = FSDPEngineWithLMHead.prepare_model_outputs
    if getattr(original, '_shadowdream_masked', False):
        return
    def masked(engine, output, output_args, micro_batch, logits_processor_func):
        return prepare_masked(original, engine, output, output_args, micro_batch, logits_processor_func)
    masked._shadowdream_masked = True
    FSDPEngineWithLMHead.prepare_model_outputs = masked


install()

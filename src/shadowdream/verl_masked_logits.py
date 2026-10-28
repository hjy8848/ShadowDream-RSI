"""Mask-aware log-probability computation for VeRL's non-fused FSDP outputs.

The caller must provide the per-response masks from the matching rollout; a
stock VeRL actor does not invoke this function automatically.
"""


def masked_jagged_logprobs(logits, input_ids, default_logprobs, masks, temperatures):
    """Replace response logprobs while retaining VeRL's jagged sequence layout.

    `logits` is [batch, padded_sequence, vocabulary]. `input_ids` and
    `default_logprobs` are aligned jagged tensors containing prompt+response.
    Each mask list covers one response token, including EOS. Temperatures are
    one positive scalar per trajectory call.
    """
    import torch

    offsets = input_ids.offsets()
    if not torch.equal(offsets, default_logprobs.offsets()):
        raise ValueError('input and logprob sequence boundaries differ')
    if len(masks) != len(offsets) - 1 or len(temperatures) != len(masks):
        raise ValueError('one mask set and temperature per sequence are required')
    token_values = input_ids.values()
    values = default_logprobs.values().clone()
    for row, (sets, temperature) in enumerate(zip(masks, temperatures)):
        begin, end = int(offsets[row]), int(offsets[row + 1])
        prompt_length = end - begin - len(sets)
        if prompt_length < 1 or temperature <= 0:
            raise ValueError('nonempty prompt and positive temperature are required')
        for step, allowed in enumerate(sets):
            position = prompt_length + step
            token = int(token_values[begin + position])
            if not allowed or token not in allowed or len(set(allowed)) != len(allowed):
                raise ValueError('response token does not match the legal token set')
            scores = logits[row, position - 1, allowed].float() / temperature
            values[begin + position - 1] = torch.log_softmax(scores, dim=-1)[allowed.index(token)]
    return torch.nested.nested_tensor_from_jagged(values, offsets)

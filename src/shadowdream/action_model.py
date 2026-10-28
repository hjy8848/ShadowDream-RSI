"""Legal-action generation shared by Service rollout and training."""


class ActionModel:
    def __init__(self, session):
        self.session = session

    def __getattr__(self, name):
        return getattr(self.session, name)

    def generate(self, messages, max_new_tokens, temperature, top_p, allowed_texts=None,
                 enable_thinking=None):
        if allowed_texts is None:
            return self.session.generate(messages, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, enable_thinking=enable_thinking)
        import torch

        tokenizer = self.session.tokenizer
        if not allowed_texts or tokenizer.eos_token_id is None:
            raise ValueError('constrained generation requires legal actions and an EOS token')
        if not 0 < top_p <= 1:
            raise ValueError('top_p must be in (0, 1]')
        choices = []
        for action in allowed_texts:
            ids = tokenizer.encode(action, add_special_tokens=False)
            if not ids or tokenizer.decode(ids, skip_special_tokens=True) != action:
                raise ValueError(f'action does not round-trip through tokenizer: {action!r}')
            choices.append(ids + [tokenizer.eos_token_id])
        prompt = self.session.encode_messages(messages, enable_thinking=enable_thinking)
        self.session.model.eval()
        selected, logprobs = [], []
        with torch.inference_mode():
            while True:
                allowed = sorted({ids[len(selected)] for ids in choices
                                  if len(ids) > len(selected) and ids[:len(selected)] == selected})
                tokens = torch.tensor([selected], dtype=torch.long, device=prompt.device)
                inputs = torch.cat((prompt, tokens), dim=1)
                logits = self.session.model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits[0, -1, allowed].float()
                if temperature > 0:
                    logits = logits / temperature
                    if top_p < 1:
                        ordered, indices = torch.sort(logits, descending=True)
                        cutoff = torch.cumsum(torch.softmax(ordered, dim=-1), dim=-1) - torch.softmax(ordered, dim=-1) >= top_p
                        ordered = ordered.masked_fill(cutoff, float('-inf'))
                        logits = torch.full_like(logits, float('-inf')).scatter(0, indices, ordered)
                    probs = torch.softmax(logits, dim=-1)
                    index = int(torch.multinomial(probs, 1).item())
                    logprob = float(torch.log_softmax(logits, dim=-1)[index].item())
                else:
                    index = int(logits.argmax().item())
                    logprob = 0.0
                token = allowed[index]
                selected.append(token)
                logprobs.append(logprob)
                if token == tokenizer.eos_token_id:
                    break
        return {'prompt_ids': prompt[0].tolist(), 'generated_ids': selected,
                'generated_text': tokenizer.decode(selected, skip_special_tokens=True),
                'behavior_logprobs': logprobs, 'model_revision': self.session.model_revision,
                'adapter_version': self.session.adapter_version,
                'sampling_parameters': {'max_new_tokens': max_new_tokens,
                    'temperature': temperature, 'top_k': 0, 'top_p': top_p}}

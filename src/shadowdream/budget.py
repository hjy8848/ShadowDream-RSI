"""Count environment interactions and stop before exceeding a fixed limit."""
from dataclasses import dataclass


class BudgetExhausted(Exception):
    pass


@dataclass
class Budget:
    limit: int
    env_steps: int = 0
    model_calls: int = 0
    model_tokens: int = 0
    editor_calls: int = 0
    editor_tokens: int = 0
    service_calls: int = 0
    service_tokens: int = 0

    def __post_init__(self):
        if self.limit < 0:
            raise ValueError('environment step budget must be nonnegative')

    def require_step(self):
        if self.env_steps >= self.limit:
            raise BudgetExhausted('environment step budget exhausted')

    def charge_step(self):
        self.require_step()
        self.env_steps += 1

    def charge_generation(self, generation, role='service'):
        if role not in {'editor', 'service'}:
            raise ValueError('generation role must be editor or service')
        tokens = len(generation.get('prompt_ids', [])) + len(generation.get('generated_ids', []))
        self.model_calls += 1
        self.model_tokens += tokens
        if role == 'editor':
            self.editor_calls += 1
            self.editor_tokens += tokens
        else:
            self.service_calls += 1
            self.service_tokens += tokens

    def snapshot(self):
        return {'env_steps': self.env_steps, 'model_calls': self.model_calls,
                'model_tokens': self.model_tokens, 'editor_calls': self.editor_calls,
                'editor_tokens': self.editor_tokens, 'service_calls': self.service_calls,
                'service_tokens': self.service_tokens}


def difference(after, before):
    return {key: after[key] - before[key] for key in before}

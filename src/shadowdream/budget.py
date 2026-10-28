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

    def __post_init__(self):
        if self.limit < 0:
            raise ValueError('environment step budget must be nonnegative')

    def require_step(self):
        if self.env_steps >= self.limit:
            raise BudgetExhausted('environment step budget exhausted')

    def charge_step(self):
        self.require_step()
        self.env_steps += 1

    def charge_generation(self, generation):
        self.model_calls += 1
        self.model_tokens += len(generation.get('prompt_ids', [])) + len(generation.get('generated_ids', []))

    def snapshot(self):
        return {'env_steps': self.env_steps, 'model_calls': self.model_calls,
                'model_tokens': self.model_tokens}


def difference(after, before):
    return {key: after[key] - before[key] for key in before}

"""Aggregate paired outcomes produced by real checkpoint executions."""
from collections import defaultdict
from dataclasses import dataclass

@dataclass(frozen=True)
class CheckpointOutcome:
    task_id: str
    checkpoint: int
    seed: int
    success: bool

@dataclass(frozen=True)
class ShadowResult:
    vector: tuple[tuple[str, int, float], ...]
    gain: float
    regression: float

def compare(parent: list[CheckpointOutcome], child: list[CheckpointOutcome]) -> ShadowResult:
    """Compare matching outcomes; each task/checkpoint receives equal weight."""
    def index(outcomes):
        result = {}
        for outcome in outcomes:
            key = outcome.task_id, outcome.checkpoint, outcome.seed
            if key in result:
                raise ValueError(f"duplicate checkpoint outcome: {key}")
            result[key] = outcome.success
        return result
    before, after = index(parent), index(child)
    if not before or before.keys() != after.keys():
        raise ValueError("parent and child must have the same nonempty checkpoint/seed set")
    deltas = defaultdict(list)
    for (task, checkpoint, seed), success in before.items():
        deltas[(task, checkpoint)].append(int(after[(task, checkpoint, seed)]) - int(success))
    vector = tuple((task, checkpoint, sum(values) / len(values))
                   for (task, checkpoint), values in sorted(deltas.items()))
    return ShadowResult(vector, sum(max(0, x[2]) for x in vector) / len(vector),
                        sum(max(0, -x[2]) for x in vector) / len(vector))

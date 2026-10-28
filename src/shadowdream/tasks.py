"""Input records for executable ALFWorld tasks."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Task:
    task_id: str
    source_game_file: str
    task_game_file: str
    source_scene_id: str
    instruction: str
    goal_kind: str
    goal_atoms: list[dict[str, Any]]
    source_goal_text: str
    custom_goal_text: str
    witness_actions: list[str]
    source_split: str
    experiment_id: str = ""
    hypothesis_id: str = ""
    hypothesis: str = ""
    feedback_refs: list[str] = field(default_factory=list)
    diagnostic_target: dict[str, Any] = field(default_factory=dict)
    generation_mode: str = "exploration"
    experiment_spec: dict[str, Any] = field(default_factory=dict)


def read_tasks(path):
    return [
        Task(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

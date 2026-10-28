"""Task-facing ALFWorld utilities used by evaluation and training."""
import json
import re
import random


TASK_LINE = re.compile(r"your task is to[^\n]*", re.IGNORECASE)
ACTION_OPTION = re.compile(r"(\d+): (.+)")


def seed_everything(seed, torch=None):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def replace_native_objective(observation, objective):
    return TASK_LINE.sub("Your task is to {}".format(objective), observation)


def direct_action_messages(task, observation, recent_actions, allowed_commands, remaining_steps):
    return [
        {
            "role": "system",
            "content": (
                "You control ALFWorld. Choose one command from the complete list of currently allowed "
                "commands. Use the task, observation, recent actions, and remaining step count to "
                "decide. Output the chosen command exactly, with no explanation."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": task,
                    "observation": observation,
                    "recent_actions": recent_actions[-5:],
                    "allowed_commands": allowed_commands,
                    "remaining_steps": remaining_steps,
                },
                ensure_ascii=False,
            ),
        },
    ]


def fact_key(fact):
    return fact["predicate"], tuple(fact["arguments"])


def goal_reached(goal_atoms, facts):
    available = {fact_key(fact) for fact in facts}
    return all(fact_key(atom) in available for atom in goal_atoms)


def task_progress(task, facts):
    available = {fact_key(fact) for fact in facts}
    completed = sum(fact_key(atom) in available for atom in task.goal_atoms)
    held_objects = {
        arguments[1]
        for predicate, arguments in available
        if predicate == "holds" and len(arguments) == 2
    }
    placed_objects = {atom["arguments"][0] for atom in task.goal_atoms if fact_key(atom) in available}
    partial = sum(
        1 for atom in task.goal_atoms
        if atom["arguments"][0] in held_objects and atom["arguments"][0] not in placed_objects
    )
    return completed, (completed + 0.5 * partial) / len(task.goal_atoms)


def resolve_action(selection, admissible_commands, action_choices):
    if selection in admissible_commands:
        return selection
    if re.fullmatch(r"\d+", selection):
        index = int(selection) - 1
        if 0 <= index < len(action_choices) and action_choices[index] in admissible_commands:
            return action_choices[index]
    lines = [line.strip() for line in selection.splitlines() if line.strip()]
    match = ACTION_OPTION.fullmatch(lines[-1]) if lines else None
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(action_choices) and match.group(2) == action_choices[index]:
            if action_choices[index] in admissible_commands:
                return action_choices[index]
    return selection

"""Flat archive bandit using real-outcome feedback."""
from dataclasses import dataclass, field
from math import log, sqrt

@dataclass
class Node:
    id: str
    parent_id: str | None
    full_score: float
    shadow_gain: float = 0.0
    visits: int = 0
    feedback_sum: float = 0.0

@dataclass
class Archive:
    nodes: dict[str, Node] = field(default_factory=dict)
    best_score: float = float('-inf')

    def add_root(self, node: Node) -> None:
        if self.nodes or node.parent_id is not None:
            raise ValueError("root must be the first node and have no parent")
        self.nodes[node.id] = node
        self.best_score = node.full_score

@dataclass
class Search:
    archive: Archive
    feedback: str
    c: float = 1.0
    beta: float = 1.0
    expansions: int = 0

    def __post_init__(self):
        if self.feedback not in {"absolute", "improvement"}:
            raise ValueError("feedback must be absolute or improvement")

    def select(self, priors=None) -> Node:
        nodes = list(self.archive.nodes.values())
        if not nodes:
            raise ValueError("archive is empty")
        def score(node):
            return ((node.feedback_sum / node.visits if node.visits else 0.0)
                    + self.c * sqrt(log(2 + self.expansions) / (1 + node.visits))
                    + self.beta * (node.shadow_gain if priors is None else priors[node.id])
                    / (1 + node.visits))
        return max(nodes, key=score)

    def add_child(self, child: Node) -> dict[str, float]:
        """Add a real result and return both absolute and historical-best feedback."""
        if child.id in self.archive.nodes or child.parent_id not in self.archive.nodes:
            raise ValueError("child must have a unique id and an existing parent")
        previous_best = self.archive.best_score
        absolute = child.full_score
        improvement = max(0.0, absolute - previous_best)
        value = absolute if self.feedback == "absolute" else improvement
        self.archive.nodes[child.id] = child
        self.archive.best_score = max(previous_best, absolute)
        self.expansions += 1
        ancestor = self.archive.nodes[child.parent_id]
        while True:
            ancestor.visits += 1
            ancestor.feedback_sum += value
            if ancestor.parent_id is None:
                break
            ancestor = self.archive.nodes[ancestor.parent_id]
        return {"absolute": absolute, "previous_best": previous_best, "improvement": improvement}

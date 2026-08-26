from dataclasses import dataclass

from algorithms.dubins import dubins_length
from model import Obstacle, Robot


@dataclass(frozen=True)
class Node:
    id: str  # "S" for start, or the obstacle's id
    robot: Robot  # where the robot stands at this node


@dataclass
class Graph:
    nodes: list[Node]
    weights: dict[tuple[str, str], float]  # (node_id, node_id) -> dubins length

    @classmethod
    def build(cls, start: Robot, obstacles: list[Obstacle]) -> "Graph":
        nodes = [Node("S", start)]

        for obs in obstacles:
            nodes.append(Node(obs.id, obs.cm_viewing_position()))

        weights = {}
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1 :]:
                d = dubins_length(node_a.robot, node_b.robot)
                weights[(node_a.id, node_b.id)] = d
                weights[(node_b.id, node_a.id)] = d

        return cls(nodes=nodes, weights=weights)

    def edge_weight(self, id_a: str, id_b: str) -> float:
        return self.weights[(id_a, id_b)]

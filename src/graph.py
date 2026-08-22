import math
from dataclasses import dataclass

from algorithms.dubins import dubins_length
from model import Direction, GRID_LENGTH_CM, Obstacle, Robot, Pose


def grid_to_pose(x_coord: int, y_coord: int, facing: Direction) -> Pose:
    """Convert a grid cell + facing into a continuous (x_cm, y_cm, theta_rad) pose."""
    x_cm = x_coord * GRID_LENGTH_CM
    y_cm = y_coord * GRID_LENGTH_CM
    theta_rad = math.atan2(facing.value[1], facing.value[0])
    return Pose(x_cm, y_cm, theta_rad)


@dataclass(frozen=True)
class Node:
    id: str  # "S" for start, or the obstacle's id
    pose: Pose


@dataclass
class Graph:
    nodes: list[Node]
    weights: dict[tuple[str, str], float]  # (node_id, node_id) -> dubins length

    @classmethod
    def build(cls, robot: Robot, obstacles: list[Obstacle]) -> "Graph":
        nodes = [Node("S", grid_to_pose(*robot.position()))]

        for obs in obstacles:
            x_coord, y_coord, facing = obs.desired_robot_position()
            nodes.append(Node((obs.id), grid_to_pose(x_coord, y_coord, facing)))

        weights = {}
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1 :]:
                d = dubins_length(node_a.pose, node_b.pose)
                weights[(node_a.id, node_b.id)] = d
                weights[(node_b.id, node_a.id)] = d

        return cls(nodes=nodes, weights=weights)

    def edge_weight(self, id_a: str, id_b: str) -> float:
        return self.weights[(id_a, id_b)]

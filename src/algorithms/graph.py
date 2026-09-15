from dataclasses import dataclass

from algorithms.dubins import dubins_length
from collision import footprint_in_collision
from model import Obstacle, Robot


@dataclass(frozen=True)
class Node:
    id: str  # "S" for start, or the obstacle's id
    viewing_pose: Robot  # where the robot stands at this node


def _resolve_viewing_pose(obstacle: Obstacle, obstacle_footprints: list) -> Robot:
    """The obstacle's ideal viewing pose if the robot can stand there
    without overlapping another obstacle or leaving the arena, otherwise the
    first fallback pose that's clear (see Obstacle.candidate_viewing_poses).

    If every candidate is blocked, hands back the ideal pose anyway -
    hybrid_astar then refuses it and plan_mission skips the obstacle,
    exactly as it did before this resolution step existed. So this can only
    rescue an obstacle that used to be skipped, never lose one that was
    fine.
    """
    candidates = obstacle.candidate_viewing_poses()
    for pose in candidates:
        if not footprint_in_collision(pose, obstacle_footprints):
            return pose
    return candidates[0]


@dataclass
class Graph:
    nodes: list[Node]
    weights: dict[tuple[str, str], float]  # (node_id, node_id) -> dubins length

    @classmethod
    def build(cls, start: Robot, obstacles: list[Obstacle]) -> "Graph":
        # Inflated, matching planner.py's own footprints list - a viewing
        # pose this resolves as "clear" must be clear by the same margin
        # hybrid_astar itself enforces, or a pose _resolve_viewing_pose
        # accepts here could still be inside OBSTACLE_MARGIN_CM and get
        # rejected as "goal in collision" the moment hybrid_astar checks it.
        obstacle_footprints = [obs.inflated_footprint_corners_cm() for obs in obstacles]

        nodes = [Node("S", start)]
        for obs in obstacles:
            nodes.append(Node(obs.id, _resolve_viewing_pose(obs, obstacle_footprints)))

        weights = {}
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1 :]:
                d = dubins_length(node_a.viewing_pose, node_b.viewing_pose)
                weights[(node_a.id, node_b.id)] = d
                weights[(node_b.id, node_a.id)] = d

        return cls(nodes=nodes, weights=weights)

    def edge_weight(self, id_a: str, id_b: str) -> float:
        return self.weights[(id_a, id_b)]

from dataclasses import dataclass

import math

from algorithms.graph import Graph
from algorithms.hamiltonian import exhaustive_search
from algorithms.hybrid_astar import hybrid_astar
from model import Obstacle, Robot, MotionPrimitive
from typing import Literal, Optional, Union


class PlanningError(RuntimeError):
    """A leg had no collision-free path; the mission cannot be driven as ordered."""

    # We should probably implement something here in case no collision-free path is calculated
    # maybe recalculate with a smaller robot footprint + allow robot footprint to move slightly outside of the arena
    pass


@dataclass
class Command:
    direction: Literal["FORWARD", "REVERSE"]
    turn: Literal["STRAIGHT", "LEFT", "RIGHT"]
    distance_cm: int
    swept_angle_deg: Optional[int] = None

    @property
    def radius_mm(self) -> int:
        distance_mm = self.distance_cm * 10
        return round(distance_mm / math.radians(self.swept_angle_deg))


@dataclass
class Leg:
    from_id: Union[Literal["S"], int]  # "S" for the initial leg, else obstacle id
    to_id: int
    commands: list[Command]


def _combine_straights(straight_primitives: list[MotionPrimitive]) -> Command:
    total_distance = sum(p.distance for p in straight_primitives)
    direction = "FORWARD" if straight_primitives[0].direction == 1 else "REVERSE"
    return Command(direction=direction, turn="STRAIGHT", distance_cm=total_distance)


def _combine_arcs(turning_primitives: list[MotionPrimitive]) -> Command:
    total_distance = sum(p.distance for p in turning_primitives)
    total_dtheta = sum(p.dtheta for p in turning_primitives)
    direction = "FORWARD" if turning_primitives[0].direction == 1 else "REVERSE"
    turn = "LEFT" if total_dtheta > 0 else "RIGHT"
    return Command(
        direction=direction,
        turn=turn,
        distance_cm=total_distance,
        swept_angle_deg=round(abs(math.degrees(total_dtheta))),
    )


def _primitives_to_commands(primitives: list[MotionPrimitive]) -> list[Command]:
    """Fold consecutive same-name primitives into single Commands."""
    commands = []
    i = 0
    while i < len(primitives):
        name = primitives[i].name
        j = i
        while j < len(primitives) and primitives[j].name == name:
            j += 1
        run = primitives[i:j]
        if "straight" in name:
            commands.append(_combine_straights(run))
        else:
            commands.append(_combine_arcs(run))
        i = j
    return commands


def plan_mission(robot: Robot, obstacles: list[Obstacle]) -> list[Leg]:
    graph = Graph.build(robot, obstacles)
    order = exhaustive_search(graph)

    id_pose_map = {node.id: node.viewing_pose for node in graph.nodes}
    footprints = [obstacle.footprint_corners_cm() for obstacle in obstacles]

    legs = []
    for from_id, to_id in zip(order, order[1:]):
        result = hybrid_astar(id_pose_map[from_id], id_pose_map[to_id], footprints)
        if result is None:
            raise PlanningError(f"no collision-free path for leg {from_id} -> {to_id}")
        legs.append(
            Leg(
                from_id="S" if not legs else from_id,
                to_id=to_id,
                commands=_primitives_to_commands(result.primitives),
            )
        )

    return legs

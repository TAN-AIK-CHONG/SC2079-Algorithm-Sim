from dataclasses import dataclass

import math

from algorithms.graph import Graph
from algorithms.hamiltonian import exhaustive_search
from algorithms.hybrid_astar import hybrid_astar
from model import Obstacle, Robot, MotionPrimitive
from typing import Literal, Optional, Union


class PlanningError(RuntimeError):
    """Raised only when NOT A SINGLE obstacle in the mission is reachable
    from the start - i.e. there is nothing at all to drive. An individual
    obstacle that hybrid_astar can't reach from wherever the robot currently
    is does NOT raise this; it's skipped instead (see plan_mission)."""


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


@dataclass
class MissionPlan:
    legs: list[Leg]
    skipped_ids: list[int]  # obstacles hybrid_astar could not reach - NOT visited, not in legs


def _combine_straights(straight_primitives: list[MotionPrimitive]) -> Command:
    total_distance = sum(p.distance for p in straight_primitives)
    direction = "FORWARD" if straight_primitives[0].direction == 1 else "REVERSE"
    return Command(direction=direction, turn="STRAIGHT", distance_cm=total_distance)


def _combine_arcs(turning_primitives: list[MotionPrimitive]) -> Command:
    total_distance = sum(p.distance for p in turning_primitives)
    total_dtheta = sum(p.dtheta for p in turning_primitives)
    direction = "FORWARD" if turning_primitives[0].direction == 1 else "REVERSE"
    # Steering side, not dtheta's sign: reversing flips which sign of dtheta
    # a given steering side produces (see hybrid_astar.py's
    # _motion_primitives), so "left" only means the same physical side
    # consistently if read from the primitive's own name - every primitive
    # in this run shares one name (that's how _primitives_to_commands
    # grouped them), so the first one is representative of the whole run.
    turn = "LEFT" if "left" in turning_primitives[0].name else "RIGHT"
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


def _apply_command(start: Robot, command: Command, distance_cm: float) -> Robot:
    """Pose after driving `distance_cm` (up to the command's full
    distance_cm) along the command's straight line or true circular arc -
    the arc radius implied by distance_cm/swept_angle_deg, not
    hybrid_astar's per-step chord approximation."""
    direction = 1 if command.direction == "FORWARD" else -1

    if command.turn == "STRAIGHT":
        return Robot(
            start.x_cm + direction * distance_cm * math.cos(start.theta_rad),
            start.y_cm + direction * distance_cm * math.sin(start.theta_rad),
            start.theta_rad,
        )

    # Sign of the heading change: LEFT/RIGHT is the physical steering side
    # (see _combine_arcs above), which only maps to a signed dtheta once
    # direction is known - reversing flips it, matching hybrid_astar's own
    # motion primitive convention.
    side_sign = 1 if command.turn == "LEFT" else -1
    dtheta_total = side_sign * direction * math.radians(command.swept_angle_deg)
    curvature = dtheta_total / command.distance_cm  # signed, rad/cm

    theta = start.theta_rad + curvature * distance_cm
    x = start.x_cm + (direction / curvature) * (math.sin(theta) - math.sin(start.theta_rad))
    y = start.y_cm - (direction / curvature) * (math.cos(theta) - math.cos(start.theta_rad))
    return Robot(x, y, theta)


def _commands_end_pose(start: Robot, commands: list[Command]) -> Robot:
    """Where the robot will actually be after driving every Command in
    order - the pose the *next* leg should plan from, since it is not
    generally the exact viewing pose (hybrid_astar only guarantees landing
    within its goal tolerance of it)."""
    pose = start
    for command in commands:
        pose = _apply_command(pose, command, command.distance_cm)
    return pose


def plan_mission(robot: Robot, obstacles: list[Obstacle]) -> MissionPlan:
    """Plan a mission visiting obstacles in the order exhaustive_search picks.

    If hybrid_astar can't find a collision-free path from wherever the robot
    currently is to the next obstacle in that order, that obstacle is
    skipped (left unvisited) rather than failing the whole mission - the
    robot just continues on toward the next obstacle in the order, still
    from its last successfully-reached position. Skipped obstacles are
    reported back in MissionPlan.skipped_ids rather than silently dropped,
    since not visiting one means lost points in the real run and whoever
    calls this needs to know.

    Note: the visiting order itself is decided once up front (by Dubins
    distance, ignoring obstacles) and is NOT re-optimised after a skip - the
    remaining stops are visited in their original relative order, not
    necessarily the shortest order for what's left.

    Raises PlanningError only if NOT ONE obstacle in the mission is
    reachable at all (nothing to drive).
    """
    graph = Graph.build(robot, obstacles)
    order = exhaustive_search(graph)

    id_pose_map = {node.id: node.viewing_pose for node in graph.nodes}
    footprints = [obstacle.footprint_corners_cm() for obstacle in obstacles]

    legs: list[Leg] = []
    skipped_ids: list[int] = []
    current_id = order[0]  # "S"
    current_pose = id_pose_map[current_id]  # updated to the actual pose reached after each leg

    for target_id in order[1:]:
        result = hybrid_astar(current_pose, id_pose_map[target_id], footprints)
        if result is None:
            skipped_ids.append(target_id)
            continue  # stay at current_id/current_pose, try the next obstacle in the order instead

        commands = _primitives_to_commands(result.primitives)
        legs.append(Leg(from_id=current_id, to_id=target_id, commands=commands))
        current_id = target_id
        # Not id_pose_map[target_id]: hybrid_astar only guarantees landing
        # within its goal tolerance of the viewing pose, so the next leg
        # must plan from where the robot will actually be.
        current_pose = _commands_end_pose(current_pose, commands)

    if not legs:
        raise PlanningError("no obstacle in the mission is reachable from the start")

    return MissionPlan(legs=legs, skipped_ids=skipped_ids)

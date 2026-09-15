from dataclasses import dataclass

import math

from algorithms.graph import Graph
from algorithms.hamiltonian import exhaustive_search
from algorithms.hybrid_astar import (
    DEFAULT_GOAL_ANGLE_TOLERANCE_RAD,
    DEFAULT_GOAL_POS_TOLERANCE_CM,
    HybridAstarResult,
    _normalize_angle,
    hybrid_astar,
)
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
    """Where the robot will end up after driving every Command in order, via
    each command's true straight line / circular arc - general-purpose
    reconstruction for callers that only have Commands to work with (e.g.
    testing/visualize_map.py, testing/gui_simulator.py). NOT used to chain
    plan_mission()'s own legs - see plan_mission for why it uses
    hybrid_astar's own validated result.path[-1] instead."""
    pose = start
    for command in commands:
        pose = _apply_command(pose, command, command.distance_cm)
    return pose


# Nudges (cm, world-frame) tried on the PREVIOUS leg's own goal when the
# CURRENT leg can't be planned - see _retry_previous_leg_for_escape.
#
# Two things were tried and abandoned before this one (see this file's git
# history for the full attempts): a forward "escape hop" a few cm away from
# the committed landing pose, then a real, collision-checked chain of
# hybrid_astar's own motion primitives up to 5 steps deep from that same
# landing pose. Neither worked on the real dead zone this was built against
# - every pose reachable via a short real drive FROM that exact landing was
# ALSO stuck, because the whole locally-reachable pocket around it can be
# cut off in hybrid_astar's discretized state graph, not just the one exact
# point. What does work: going back one step and re-landing the PREVIOUS
# leg somewhere else entirely (confirmed - see the same history) - hybrid_
# astar's own search for that leg had other pose it would have accepted as
# "arrived", just not the one it happened to pop off the heap first.
_BACKTRACK_NUDGES_CM = [
    (dx, dy)
    for dx in (-6, -4, -2, 0, 2, 4, 6)
    for dy in (-6, -4, -2, 0, 2, 4, 6)
    if (dx, dy) != (0, 0)
]


def _retry_previous_leg_for_escape(
    prev_start: Robot,
    prev_ideal_goal: Robot,
    next_goal: Robot,
    footprints: list,
) -> tuple[HybridAstarResult, HybridAstarResult] | None:
    """Re-plans the leg that landed on prev_ideal_goal toward small variants
    of that same goal, keeping only variants whose landing is still within
    the ordinary DEFAULT_GOAL_POS_TOLERANCE_CM/ANGLE of the REAL viewing
    pose (never trading image-visibility for a working next leg - a variant
    landing 9cm off the true pose is rejected even if it would otherwise
    work), and returns the first variant whose landing can also reach
    next_goal - or None if no variant can do both.

    Only called when going straight from prev_start to next_goal already
    failed AND landing exactly on prev_ideal_goal doesn't lead anywhere
    (see plan_mission) - the direct, no-detour case is always tried first
    and is unaffected by any of this.
    """
    for dx, dy in _BACKTRACK_NUDGES_CM:
        nudged_goal = Robot(prev_ideal_goal.x_cm + dx, prev_ideal_goal.y_cm + dy, prev_ideal_goal.theta_rad)
        prev_result = hybrid_astar(prev_start, nudged_goal, footprints)
        if prev_result is None:
            continue

        landing = prev_result.path[-1]
        pos_error_cm = math.hypot(landing.x_cm - prev_ideal_goal.x_cm, landing.y_cm - prev_ideal_goal.y_cm)
        angle_error_rad = abs(_normalize_angle(landing.theta_rad - prev_ideal_goal.theta_rad))
        if pos_error_cm > DEFAULT_GOAL_POS_TOLERANCE_CM or angle_error_rad > DEFAULT_GOAL_ANGLE_TOLERANCE_RAD:
            continue  # would visit the previous obstacle "worse" than normal - not worth it

        continuation = hybrid_astar(landing, next_goal, footprints)
        if continuation is not None:
            return prev_result, continuation

    return None


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

    An obstacle whose ideal viewing pose is blocked - a neighbour sitting in
    it, or the obstacle being close enough to a wall that the ideal standoff
    is out of bounds - is retargeted to a nearby unobstructed pose before
    being skipped (Graph.build via graph._resolve_viewing_pose); id_pose_map
    below already carries that resolved pose.

    A leg that can't be planned directly from wherever the previous leg
    actually left the robot gets one more try before being skipped: re-land
    the PREVIOUS leg on a small variant of its own goal (see
    _retry_previous_leg_for_escape), still within its normal goal tolerance
    of the real viewing pose, in case the exact pose that search happened
    to land on falls in a dead zone that a different, equally valid landing
    of that same leg is not actually stuck in.

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
    prev_leg_start_pose = current_pose  # the pose legs[-1] (once there is one) was planned from

    for target_id in order[1:]:
        target_pose = id_pose_map[target_id]
        leg_start_pose = current_pose
        result = hybrid_astar(leg_start_pose, target_pose, footprints)

        if result is None and legs:
            # Direct attempt failed and there's a previous leg to back up
            # into - see _retry_previous_leg_for_escape and plan_mission's
            # own docstring for why this is tried before giving up.
            retry = _retry_previous_leg_for_escape(
                prev_leg_start_pose, id_pose_map[legs[-1].to_id], target_pose, footprints
            )
            if retry is not None:
                new_prev_result, result = retry
                legs[-1] = Leg(
                    from_id=legs[-1].from_id,
                    to_id=legs[-1].to_id,
                    commands=_primitives_to_commands(new_prev_result.primitives),
                )
                # legs[-1] now lands somewhere else - THIS leg (about to be
                # appended below) actually starts from there, not from the
                # original leg_start_pose captured above.
                leg_start_pose = new_prev_result.path[-1]

        if result is None:
            skipped_ids.append(target_id)
            continue  # stay at current_id/current_pose, try the next obstacle in the order instead

        commands = _primitives_to_commands(result.primitives)
        legs.append(Leg(from_id=current_id, to_id=target_id, commands=commands))
        current_id = target_id
        prev_leg_start_pose = leg_start_pose
        # result.path[-1], not id_pose_map[target_id] or a Command
        # reconstruction: hybrid_astar only guarantees landing within its
        # goal tolerance of the viewing pose, so the next leg must plan from
        # where the robot will actually be - and result.path[-1] is the
        # exact state hybrid_astar itself already validated as collision-free
        # on the way to accepting this leg (every expansion step is checked
        # via _segment_collision_free). Reconstructing that pose from this
        # leg's rounded Commands instead (via _commands_end_pose) lands a
        # hair off from the raw search state - usually harmless, but just
        # enough, near a tight corner, to occasionally tip the reconstructed
        # pose into collision. hybrid_astar() refuses to even start a search
        # from a start pose already in collision, so that one bad
        # reconstruction silently failed every leg after it for the rest of
        # the mission, not just the one leg it actually affected.
        current_pose = result.path[-1]

    if not legs:
        raise PlanningError("no obstacle in the mission is reachable from the start")

    return MissionPlan(legs=legs, skipped_ids=skipped_ids)

import heapq
import math
from dataclasses import dataclass

from algorithms.dubins import TURNING_RADIUS_CM, dubins_length
from collision import footprint_in_collision
from model import Corners, Robot, MotionPrimitive

STEP_CM = 10
SEGMENT_SAMPLES = 2
NUM_HEADING_BUCKETS = 72
POS_RESOLUTION_CM = 5
REVERSE_COST_MULTIPLIER = 1

# Two-tier goal tolerance: try DEFAULT first, and only if that genuinely
# finds nothing, retry once with LOOSE as a last resort.
#
# There used to be a third, tighter TIGHT tier (5cm/5deg) ahead of this one.
# It was dropped: trying TIGHT first before DEFAULT never won anything -
# every leg it couldn't solve still had to pay for TIGHT's own full
# exhaustive-failure search (~20-60s) before falling through, and DEFAULT
# alone was shown to solve every leg TIGHT could plus the hard ones TIGHT
# couldn't (see the 3-tier timing comparison: adding a step-down tier
# between TIGHT and LOOSE only cut total time from 404.5s to 400.6s across
# a 5-trial/30-leg sample - TIGHT's own failures were the entire cost, not
# which fallback came after). Going straight to DEFAULT is expected to
# remove most of that ~400s.
#
# Re-validate against a fresh sample if TURNING_RADIUS_CM changes again -
# these numbers are tuned to 30cm, not derived from first principles.
DEFAULT_GOAL_POS_TOLERANCE_CM = 7
DEFAULT_GOAL_ANGLE_TOLERANCE_RAD = math.radians(10)

LOOSE_GOAL_POS_TOLERANCE_CM = 6
LOOSE_GOAL_ANGLE_TOLERANCE_RAD = math.radians(14)

# Extra cost (in cm-equivalent) charged when a primitive's type differs from
# the one immediately before it (e.g. forward_left -> forward_right). With
# no penalty, alternating primitives cost exactly the same as a long run of
# one type - the search has no reason to prefer either, so it can produce
# stuttery FL/FR/FL/FR... paths where a real (if only approximately as
# short) run would do. This biases A* toward committing to a maneuver
# instead, without smoothing anything after the fact - every pose in the
# result still maps to an exact primitive, so the command list stays exact.
#
# Swept 0/3/5/8/10/15 against the same 5-trial/30-leg sample used to
# validate the goal tolerance: transitions/commands drop fast from 0->3
# (226->150, -34%) then flatten hard (140 at 5, 133 at 10, 123 at 15) while
# time keeps climbing roughly linearly with the penalty (46s -> 66s -> 76s
# -> 106s -> 112s -> 171s). 5 sits just past the steep part of the curve -
# most of the available smoothing for a fraction of the cost of going
# further. Re-validate reachability and re-sweep if TURNING_RADIUS_CM or
# STEP_CM change.
TURN_CHANGE_PENALTY_CM = 5


def _normalize_angle(theta: float) -> float:
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta


def _motion_primitives(step: int) -> list[MotionPrimitive]:
    """MotionPrimitive for each available action."""
    dtheta = step / TURNING_RADIUS_CM
    return [
        MotionPrimitive("forward_straight", 1, 0.0, step),
        MotionPrimitive("forward_left", 1, dtheta, step),
        MotionPrimitive("forward_right", 1, -dtheta, step),
        MotionPrimitive("reverse_straight", -1, 0.0, step),
        MotionPrimitive("reverse_left", -1, dtheta, step),
        MotionPrimitive("reverse_right", -1, -dtheta, step),
    ]


@dataclass
class HybridAstarResult:
    path: list[Robot]
    primitives: list[MotionPrimitive]
    length: float


def _segment_collision_free(
    start: Robot,
    end: Robot,
    obstacles: list[Corners],
    samples: int = SEGMENT_SAMPLES,
) -> bool:
    dx = end.x_cm - start.x_cm
    dy = end.y_cm - start.y_cm
    dtheta = end.theta_rad - start.theta_rad

    for i in range(samples + 1):
        t = i / samples
        sample = Robot(
            start.x_cm + dx * t,
            start.y_cm + dy * t,
            start.theta_rad + dtheta * t,
        )
        if footprint_in_collision(sample, obstacles):
            return False
    return True


def hybrid_astar(
    start: Robot,
    goal: Robot,
    obstacles: list[Corners],
) -> HybridAstarResult | None:
    """Try DEFAULT goal tolerance first; only if that finds nothing, retry
    once with LOOSE as a last resort so a hard-to-reach obstacle still gets
    visited, just less precisely. See the DEFAULT_/LOOSE_ constants above
    for why there's no tighter tier ahead of DEFAULT."""
    for pos_tolerance_cm, angle_tolerance_rad in (
        (DEFAULT_GOAL_POS_TOLERANCE_CM, DEFAULT_GOAL_ANGLE_TOLERANCE_RAD),
        (LOOSE_GOAL_POS_TOLERANCE_CM, LOOSE_GOAL_ANGLE_TOLERANCE_RAD),
    ):
        result = _search(start, goal, obstacles, pos_tolerance_cm, angle_tolerance_rad)
        if result is not None:
            return result
    return None


def _search(
    start: Robot,
    goal: Robot,
    obstacles: list[Corners],
    pos_tolerance_cm: float,
    angle_tolerance_rad: float,
) -> HybridAstarResult | None:
    if footprint_in_collision(start, obstacles) or footprint_in_collision(
        goal, obstacles
    ):
        print(
            "Start or goal viewing pose is in collision with an obstacle or out of bounds."
        )
        return None

    actions = _motion_primitives(STEP_CM)
    heading_res = 2 * math.pi / NUM_HEADING_BUCKETS

    def state_key(x, y, theta, last_primitive_name):
        # last_primitive_name is part of the state's identity, not just
        # bookkeeping: two arrivals at the same (x, y, theta) via different
        # primitive types are genuinely different states for this search,
        # since they lead to different transition costs going forward. If
        # last_primitive_name were left out here, whichever arrival happened
        # to have lower g would silently win the dedup and the other type's
        # "committed to this maneuver" history would be lost, undermining
        # the whole point of the penalty below.
        return (
            round(x / POS_RESOLUTION_CM),
            round(y / POS_RESOLUTION_CM),
            round(_normalize_angle(theta) / heading_res) % NUM_HEADING_BUCKETS,
            last_primitive_name,
        )

    def heuristic(x, y, theta):
        return dubins_length(Robot(x, y, theta), goal)

    start_state = (start.x_cm, start.y_cm, start.theta_rad)
    start_key = state_key(*start_state, None)

    open_heap = [(heuristic(*start_state), 0.0, start_state, None, None)]
    came_from = {start_key: None}
    g_scores = {start_key: 0.0}
    visited = set()

    while open_heap:
        _, g, state, _, last_primitive_name = heapq.heappop(open_heap)
        x, y, theta = state
        key = state_key(x, y, theta, last_primitive_name)

        if key in visited:
            continue
        visited.add(key)

        reached_goal = (
            math.hypot(goal.x_cm - x, goal.y_cm - y) < pos_tolerance_cm
            and abs(_normalize_angle(theta - goal.theta_rad)) < angle_tolerance_rad
        )
        if reached_goal:
            return _reconstruct(came_from, start_state, key)

        current = Robot(x, y, theta)
        for primitive in actions:
            new_theta = theta + primitive.dtheta
            new_x = x + primitive.direction * primitive.distance * math.cos(theta)
            new_y = y + primitive.direction * primitive.distance * math.sin(theta)

            if not _segment_collision_free(
                current, Robot(new_x, new_y, new_theta), obstacles
            ):
                continue

            step_cost = primitive.distance * (
                REVERSE_COST_MULTIPLIER if primitive.direction == -1 else 1
            )
            if last_primitive_name is not None and primitive.name != last_primitive_name:
                step_cost += TURN_CHANGE_PENALTY_CM
            new_g = g + step_cost
            new_key = state_key(new_x, new_y, new_theta, primitive.name)

            if new_key in visited:
                continue
            if new_key in g_scores and g_scores[new_key] <= new_g:
                continue

            g_scores[new_key] = new_g
            came_from[new_key] = (key, (new_x, new_y, new_theta), primitive)
            new_f = new_g + heuristic(new_x, new_y, new_theta)
            heapq.heappush(open_heap, (new_f, new_g, (new_x, new_y, new_theta), key, primitive.name))

    return None


def _reconstruct(came_from, start_state, goal_key) -> HybridAstarResult:
    path = []
    primitives = []
    key = goal_key
    while came_from.get(key) is not None:
        prev_key, state, primitive = came_from[key]
        path.append(Robot(*state))
        primitives.append(primitive)
        key = prev_key
    path.append(Robot(*start_state))
    path.reverse()
    primitives.reverse()
    # Real physical distance driven - NOT the search's internal g-score,
    # which also carries TURN_CHANGE_PENALTY_CM (an artificial cost that
    # biases the search toward smoother paths, not a real distance driven).
    # Conflating the two would overstate reported/displayed path lengths by
    # however many primitive-type transitions the path has.
    length = sum(p.distance for p in primitives)
    return HybridAstarResult(path=path, primitives=primitives, length=length)

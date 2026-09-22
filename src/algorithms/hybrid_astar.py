import heapq
import math
from dataclasses import dataclass

from algorithms.dubins import dubins_length
from collision import footprint_in_collision
from model import Corners, Robot, MotionPrimitive

STEP_CM = 10
SEGMENT_SAMPLES = 3
COLLISION_SAMPLE_RESOLUTION_CM = 3  # see _segment_collision_free
NUM_HEADING_BUCKETS = 72
POS_RESOLUTION_CM = 5
REVERSE_COST_MULTIPLIER = 1
LEFT_TURNING_RADIUS_CM = 20
RIGHT_TURNING_RADIUS_CM = 35
QUARTER_TURN_RAD = math.pi / 2  # every turn is a fixed 90deg arc - see _motion_primitives

DEFAULT_GOAL_POS_TOLERANCE_CM = 3
DEFAULT_GOAL_ANGLE_TOLERANCE_RAD = math.radians(5)

LOOSE_GOAL_POS_TOLERANCE_CM = 5
LOOSE_GOAL_ANGLE_TOLERANCE_RAD = math.radians(10)

TURN_CHANGE_PENALTY_CM = 5


def _normalize_angle(theta: float) -> float:
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta


def _motion_primitives(step: int) -> list[MotionPrimitive]:
    """MotionPrimitive for each available action. Turns are always a full
    90-degree arc, not proportional to `step` - the team standardised on
    exactly-90-degree turns only (better calibrated, more predictable, than
    letting the search request arbitrary angles - see turn_tuning.h's
    TURN_LEFT_TARGET_SCALE/TURN_RIGHT_TARGET_SCALE, tuned specifically
    against A=90 commands). Only the straight primitives still scale with
    `step`; a turn's arc LENGTH is whatever a quarter-circle at that side's
    radius comes out to, rounded to the nearest cm to match the Command
    model's integer distance_cm."""
    left_arc_distance = round(LEFT_TURNING_RADIUS_CM * QUARTER_TURN_RAD)
    right_arc_distance = round(RIGHT_TURNING_RADIUS_CM * QUARTER_TURN_RAD)
    return [
        MotionPrimitive("forward_straight", 1, 0.0, step),
        MotionPrimitive("forward_left", 1, QUARTER_TURN_RAD, left_arc_distance),
        MotionPrimitive("forward_right", 1, -QUARTER_TURN_RAD, right_arc_distance),
        MotionPrimitive("reverse_straight", -1, 0.0, step),
        MotionPrimitive("reverse_left", -1, -QUARTER_TURN_RAD, left_arc_distance),
        MotionPrimitive("reverse_right", -1, QUARTER_TURN_RAD, right_arc_distance),
    ]


@dataclass
class HybridAstarResult:
    path: list[Robot]
    primitives: list[MotionPrimitive]
    length: float


def _advance(
    x: float, y: float, theta: float, primitive: MotionPrimitive, fraction: float = 1.0
) -> tuple[float, float, float]:
    new_theta = theta + primitive.dtheta * fraction
    if primitive.dtheta == 0.0:
        step = primitive.direction * primitive.distance * fraction
        return x + step * math.cos(theta), y + step * math.sin(theta), new_theta

    radius = primitive.direction * primitive.distance / primitive.dtheta
    return (
        x + radius * (math.sin(new_theta) - math.sin(theta)),
        y - radius * (math.cos(new_theta) - math.cos(theta)),
        new_theta,
    )


def _segment_collision_free(
    x: float,
    y: float,
    theta: float,
    primitive: MotionPrimitive,
    obstacles: list[Corners],
) -> bool:
    # SEGMENT_SAMPLES alone was tuned for ~STEP_CM-long primitives; a fixed
    # 90-degree turn's arc is 3-5x longer (up to ~55cm at RIGHT_TURNING_RADIUS_CM),
    # so a flat sample count would leave gaps along it wide enough to miss a
    # real collision. Scaling with the primitive's own length keeps the gap
    # between samples roughly constant (~COLLISION_SAMPLE_RESOLUTION_CM)
    # regardless of how long the primitive actually is.
    samples = max(SEGMENT_SAMPLES, round(primitive.distance / COLLISION_SAMPLE_RESOLUTION_CM))
    for i in range(samples + 1):
        sample = _advance(x, y, theta, primitive, i / samples)
        if footprint_in_collision(Robot(*sample), obstacles):
            return False
    return True


def hybrid_astar(
    start: Robot,
    goal: Robot,
    obstacles: list[Corners],
) -> HybridAstarResult | None:
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

        for primitive in actions:
            new_x, new_y, new_theta = _advance(x, y, theta, primitive)

            if not _segment_collision_free(x, y, theta, primitive, obstacles):
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
    length = sum(p.distance for p in primitives)
    return HybridAstarResult(path=path, primitives=primitives, length=length)

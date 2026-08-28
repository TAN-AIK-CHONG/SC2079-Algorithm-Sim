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
GOAL_POS_TOLERANCE_CM = 5
GOAL_ANGLE_TOLERANCE_RAD = math.radians(5)
REVERSE_COST_MULTIPLIER = 1


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
    if footprint_in_collision(start, obstacles) or footprint_in_collision(
        goal, obstacles
    ):
        print(
            "Start or goal viewing pose is in collision with an obstacle or out of bounds."
        )
        return None

    actions = _motion_primitives(STEP_CM)
    heading_res = 2 * math.pi / NUM_HEADING_BUCKETS

    def state_key(x, y, theta):
        return (
            round(x / POS_RESOLUTION_CM),
            round(y / POS_RESOLUTION_CM),
            round(_normalize_angle(theta) / heading_res) % NUM_HEADING_BUCKETS,
        )

    def heuristic(x, y, theta):
        return dubins_length(Robot(x, y, theta), goal)

    start_state = (start.x_cm, start.y_cm, start.theta_rad)
    start_key = state_key(*start_state)

    open_heap = [(heuristic(*start_state), 0.0, start_state, None)]
    came_from = {start_key: None}
    g_scores = {start_key: 0.0}
    visited = set()

    while open_heap:
        _, g, state, _ = heapq.heappop(open_heap)
        x, y, theta = state
        key = state_key(x, y, theta)

        if key in visited:
            continue
        visited.add(key)

        reached_goal = (
            math.hypot(goal.x_cm - x, goal.y_cm - y) < GOAL_POS_TOLERANCE_CM
            and abs(_normalize_angle(theta - goal.theta_rad)) < GOAL_ANGLE_TOLERANCE_RAD
        )
        if reached_goal:
            return _reconstruct(came_from, start_state, key, g)

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
            new_g = g + step_cost
            new_key = state_key(new_x, new_y, new_theta)

            if new_key in visited:
                continue
            if new_key in g_scores and g_scores[new_key] <= new_g:
                continue

            g_scores[new_key] = new_g
            came_from[new_key] = (key, (new_x, new_y, new_theta), primitive)
            new_f = new_g + heuristic(new_x, new_y, new_theta)
            heapq.heappush(open_heap, (new_f, new_g, (new_x, new_y, new_theta), key))

    return None


def _reconstruct(came_from, start_state, goal_key, total_length) -> HybridAstarResult:
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
    return HybridAstarResult(path=path, primitives=primitives, length=total_length)

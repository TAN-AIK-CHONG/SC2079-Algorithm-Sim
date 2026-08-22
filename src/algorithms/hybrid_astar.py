import heapq
import math
from dataclasses import dataclass

from algorithms.dubins import TURNING_RADIUS_CM, dubins_length
from model import ARENA_LENGTH_CM, Pose

STEP_CM = 5.0
NUM_HEADING_BUCKETS = 72 # 22.5 degrees
POS_RESOLUTION_CM = 5.0
GOAL_POS_TOLERANCE_CM = 5.0
GOAL_ANGLE_TOLERANCE_RAD = math.radians(5)
REVERSE_COST_MULTIPLIER = 1.0 # Initial assumption that forward and backward driving will have the same cost


def _normalize_angle(theta: float) -> float:
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta


def _motion_primitives(step: float):
    """(name, direction, dtheta, distance) for each available action."""
    dtheta = step / TURNING_RADIUS_CM
    return [
        ("forward_straight", 1, 0.0, step),
        ("forward_left", 1, dtheta, step),
        ("forward_right", 1, -dtheta, step),
        ("reverse_straight", -1, 0.0, step),
        ("reverse_left", -1, dtheta, step),
        ("reverse_right", -1, -dtheta, step),
    ]


@dataclass
class HybridAstarResult:
    poses: list[Pose]  # start -> ... -> goal
    length: float


def _in_collision(x: float, y: float, obstacle_boxes) -> bool:
    if not (0 <= x <= ARENA_LENGTH_CM and 0 <= y <= ARENA_LENGTH_CM):
        return True
    for (x0, y0, x1, y1) in obstacle_boxes:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _segment_collision_free(x0, y0, x1, y1, obstacle_boxes, samples=2) -> bool:
    for i in range(samples + 1):
        t = i / samples
        if _in_collision(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, obstacle_boxes):
            return False
    return True


def hybrid_astar(
    start: Pose,
    goal: Pose,
    obstacle_boxes: list[tuple[float, float, float, float]],
) -> HybridAstarResult | None:
    if _in_collision(start.x_cm, start.y_cm, obstacle_boxes) or _in_collision(goal.x_cm, goal.y_cm, obstacle_boxes):
        print("Start or goal viewing pose is in collision with an obstacle or out of bounds.")
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
        return dubins_length(Pose(x, y, theta), goal)

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

        for name, direction, dtheta, distance in actions:
            new_theta = theta + dtheta
            new_x = x + direction * distance * math.cos(theta)
            new_y = y + direction * distance * math.sin(theta)

            if not _segment_collision_free(x, y, new_x, new_y, obstacle_boxes):
                continue

            step_cost = distance * (REVERSE_COST_MULTIPLIER if direction == -1 else 1.0)
            new_g = g + step_cost
            new_key = state_key(new_x, new_y, new_theta)

            if new_key in visited:
                continue
            if new_key in g_scores and g_scores[new_key] <= new_g:
                continue

            g_scores[new_key] = new_g
            came_from[new_key] = (key, (new_x, new_y, new_theta))
            new_f = new_g + heuristic(new_x, new_y, new_theta)
            heapq.heappush(open_heap, (new_f, new_g, (new_x, new_y, new_theta), key))

    return None


def _reconstruct(came_from, start_state, goal_key, total_length) -> HybridAstarResult:
    poses = []
    key = goal_key
    while came_from.get(key) is not None:
        prev_key, state = came_from[key]
        poses.append(Pose(*state))
        key = prev_key
    poses.append(Pose(*start_state))
    poses.reverse()
    return HybridAstarResult(poses=poses, length=total_length)

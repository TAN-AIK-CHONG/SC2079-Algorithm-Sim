import heapq
import math
from collections.abc import Iterator
from dataclasses import dataclass

from algorithms.dubins import dubins_length
from collision import footprint_in_collision
from model import MAX_HEADING_ERROR_RAD, Corners, Obstacle, Robot, MotionPrimitive

STEP_CM = 10
SEGMENT_SAMPLES = 3
NUM_HEADING_BUCKETS = 72
POS_RESOLUTION_CM = 5
REVERSE_COST_MULTIPLIER = 1
LEFT_TURNING_RADIUS_CM = 20
RIGHT_TURNING_RADIUS_CM = 35

# The real car drifts more on arcs and on every change of manoeuvre than on
# straights, so these steer the search towards long straight runs with few,
# deliberate turns.
TURN_COST_MULTIPLIER = 8.0
STEERING_CHANGE_PENALTY_CM = 25
DIRECTION_SWITCH_PENALTY_CM = 35

# NOTE: Arbitrarily set. Optimization: Don't compute collision against obstacle if the obstacle is further than this distance from the robot.
MOVE_COLLISION_RADIUS_CM = 55


def _normalize_angle(theta: float) -> float:
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta


def _motion_primitives(step: int) -> list[MotionPrimitive]:
    dtheta_left = step / LEFT_TURNING_RADIUS_CM
    dtheta_right = step / RIGHT_TURNING_RADIUS_CM
    return [
        MotionPrimitive("forward_straight", 1, 0.0, step),
        MotionPrimitive("forward_left", 1, dtheta_left, step),
        MotionPrimitive("forward_right", 1, -dtheta_right, step),
        MotionPrimitive("reverse_straight", -1, 0.0, step),
        MotionPrimitive("reverse_left", -1, -dtheta_left, step),
        MotionPrimitive("reverse_right", -1, dtheta_right, step),
    ]


def _steering(primitive: MotionPrimitive) -> str:
    return primitive.name.split("_")[1]


def _transition_cost(previous: MotionPrimitive | None, primitive: MotionPrimitive) -> float:
    cost = primitive.distance
    if primitive.direction == -1:
        cost *= REVERSE_COST_MULTIPLIER
    if primitive.dtheta != 0.0:
        cost *= TURN_COST_MULTIPLIER

    if previous is None:
        return cost
    if _steering(primitive) != _steering(previous):
        cost += STEERING_CHANGE_PENALTY_CM
    if primitive.direction != previous.direction:
        cost += DIRECTION_SWITCH_PENALTY_CM
    return cost


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
    samples: int = SEGMENT_SAMPLES,
) -> bool:
    nearby_obstacles = _obstacles_within_move_reach(Robot(x, y, theta), obstacles)
    for i in range(samples + 1):
        sample = _advance(x, y, theta, primitive, i / samples)
        if footprint_in_collision(Robot(*sample), nearby_obstacles):
            return False
    return True


def _obstacles_within_move_reach(robot: Robot, obstacles: list[Corners]) -> list[Corners]:
    car_centre = _centre(robot.footprint_corners_cm())
    return [
        obstacle
        for obstacle in obstacles
        if math.dist(car_centre, _centre(obstacle)) <= MOVE_COLLISION_RADIUS_CM
    ]


def _centre(corners: Corners) -> tuple[float, float]:
    return (
        sum(x for x, _ in corners) / len(corners),
        sum(y for _, y in corners) / len(corners),
    )


def hybrid_astar(
    start: Robot,
    target: Obstacle,
    obstacles: list[Corners],
) -> HybridAstarResult | None:
    """Path to the first pose the search reaches from which the camera can
    photograph target's image (Obstacle.is_viewed_from), or None if no such
    pose is reachable."""
    return next(viewing_arrivals(start, target, obstacles), None)


def viewing_arrivals(
    start: Robot,
    target: Obstacle,
    obstacles: list[Corners],
) -> Iterator[HybridAstarResult]:
    """Paths to every pose that views target's image, in the order the
    search reaches them, one per (x, y, heading) cell. The first is what
    hybrid_astar returns; the rest are other landings of the same leg."""
    if footprint_in_collision(start, obstacles):
        print("Start pose is in collision with an obstacle or out of bounds.")
        return

    # Aimed at the ideal pose, but the search stops at the first pose that
    # already views the image, so it never manoeuvres closer than it must.
    heuristic_goal = target.cm_viewing_position()
    actions = _motion_primitives(STEP_CM)
    actions_by_name = {action.name: action for action in actions}
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
        return 2.5 * dubins_length(Robot(x, y, theta), heuristic_goal)

    start_state = (start.x_cm, start.y_cm, start.theta_rad)
    start_key = state_key(*start_state, None)

    open_heap = [(heuristic(*start_state), 0.0, start_state, None, None)]
    came_from = {start_key: None}
    g_scores = {start_key: 0.0}
    visited = set()
    arrived_cells = set()

    while open_heap:
        _, g, state, _, last_primitive_name = heapq.heappop(open_heap)
        x, y, theta = state
        key = state_key(x, y, theta, last_primitive_name)

        if key in visited:
            continue
        visited.add(key)

        cell = key[:3]  # without last_primitive_name
        if cell not in arrived_cells and target.is_viewed_from(Robot(x, y, theta)):
            arrived_cells.add(cell)
            yield _reconstruct(came_from, start_state, key)

        last_primitive = actions_by_name.get(last_primitive_name)
        for primitive in actions:
            new_x, new_y, new_theta = _advance(x, y, theta, primitive)

            if not _segment_collision_free(x, y, theta, primitive, obstacles):
                continue

            new_g = g + _transition_cost(last_primitive, primitive)
            new_key = state_key(new_x, new_y, new_theta, primitive.name)

            if new_key in visited:
                continue
            if new_key in g_scores and g_scores[new_key] <= new_g:
                continue

            g_scores[new_key] = new_g
            came_from[new_key] = (key, (new_x, new_y, new_theta), primitive)
            new_f = new_g + heuristic(new_x, new_y, new_theta)
            heapq.heappush(open_heap, (new_f, new_g, (new_x, new_y, new_theta), key, primitive.name))


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

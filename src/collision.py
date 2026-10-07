import math

from model import ARENA_LENGTH_CM, Corners, Point, Robot

# A turn swings the rear overhang out by up to ~3mm. Without that much slack,
# a car parked flush against a wall - as it is at the start - could never
# turn away from it.
WALL_TOLERANCE_CM = 0.3

WORLD_AXES = ((1.0, 0.0), (0.0, 1.0))
# cos(pi/2) is ~6e-17, not 0, so a car facing along y still has edges off
# the axis by float noise.
AXIS_ALIGNED_TOLERANCE_CM = 1e-9


def footprint_in_collision(robot: Robot, obstacles: list[Corners]) -> bool:
    corners = robot.footprint_corners_cm()
    if not _inside_arena(corners):
        return True

    return any(_overlaps(corners, obstacle) for obstacle in obstacles)


def _inside_arena(corners: Corners) -> bool:
    low = -WALL_TOLERANCE_CM
    high = ARENA_LENGTH_CM + WALL_TOLERANCE_CM
    return all(low <= cx <= high and low <= cy <= high for cx, cy in corners)


def _overlaps(robot_corners: Corners, obstacle_corners: Corners) -> bool:
    axes = WORLD_AXES
    if not _is_axis_aligned(robot_corners):
        # A car facing along x or y has edges on the world axes already, so
        # only a rotated car (mid-arc) needs its own edges tested too.
        axes += (
            _edge_axis(robot_corners[0], robot_corners[1]),
            _edge_axis(robot_corners[1], robot_corners[2]),
        )

    for axis in axes:
        robot_min, robot_max = _project(robot_corners, axis)
        obstacle_min, obstacle_max = _project(obstacle_corners, axis)
        if robot_max <= obstacle_min or obstacle_max <= robot_min:
            return False
    return True


def _is_axis_aligned(corners: Corners) -> bool:
    (start_x, start_y), (end_x, end_y) = corners[0], corners[1]
    return abs(end_x - start_x) < AXIS_ALIGNED_TOLERANCE_CM or abs(end_y - start_y) < AXIS_ALIGNED_TOLERANCE_CM


def _edge_axis(start: Point, end: Point) -> Point:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    return dx / length, dy / length


def _project(corners: Corners, axis: Point) -> tuple[float, float]:
    axis_x, axis_y = axis
    values = [axis_x * cx + axis_y * cy for cx, cy in corners]
    return min(values), max(values)

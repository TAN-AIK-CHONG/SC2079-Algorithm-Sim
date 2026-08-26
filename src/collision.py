import math

from model import ARENA_LENGTH_CM, Corners, Point, Robot

WALL_TOLERANCE_CM = 1e-9


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
    axes = (
        (1.0, 0.0),
        (0.0, 1.0),
        _edge_axis(robot_corners[0], robot_corners[1]),
        _edge_axis(robot_corners[1], robot_corners[2]),
    )

    for axis in axes:
        robot_min, robot_max = _project(robot_corners, axis)
        obstacle_min, obstacle_max = _project(obstacle_corners, axis)
        if robot_max <= obstacle_min or obstacle_max <= robot_min:
            return False
    return True


def _edge_axis(start: Point, end: Point) -> Point:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    return dx / length, dy / length


def _project(corners: Corners, axis: Point) -> tuple[float, float]:
    axis_x, axis_y = axis
    values = [axis_x * cx + axis_y * cy for cx, cy in corners]
    return min(values), max(values)

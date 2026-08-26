import math

import rsplan

from model import Robot

TURNING_RADIUS_CM = 25


def dubins_path(
    start: Robot,
    end: Robot,
    radius: float = TURNING_RADIUS_CM,
):

    return rsplan.path(
        (start.x_cm, start.y_cm, start.theta_rad),
        (end.x_cm, end.y_cm, end.theta_rad),
        radius,
        0,
        0.5,
    )


def dubins_length(
    start: Robot,
    end: Robot,
    radius: float = TURNING_RADIUS_CM,
) -> float:
    return dubins_path(start, end, radius).total_length


def path_commands(path) -> list[dict]:
    """Break an rsplan Path into a list of drive commands: each one is
    either a straight run or a constant-radius turn, in order. A Segment's
    `length` is arc length for turns (turn_radius * angle_radians) and
    straight-line distance for straight segments; `direction` is 1 (forward)
    or -1 (backward, i.e. reverse gear) regardless of segment type - Reeds-
    Shepp paths (what rsplan computes) allow reversing, unlike plain Dubins.
    """
    commands = []
    for segment in path.segments:
        distance_cm = abs(segment.length)
        if distance_cm < 1e-6:
            continue  # zero-length segments happen in degenerate cases; nothing to drive

        command = {
            "direction": "forward" if segment.direction == 1 else "backward",
            "turn": segment.type,  # "left", "right", or "straight"
            "distance_cm": round(distance_cm, 2),
            "degrees": 0.0,
        }
        if not segment.is_straight:
            command["degrees"] = round(math.degrees(distance_cm / segment.turn_radius), 2)
        commands.append(command)

    return commands

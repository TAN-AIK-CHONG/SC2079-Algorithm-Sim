import math

import rsplan

from model import Pose

TURNING_RADIUS_CM = 25


def dubins_path(
    start_pose: Pose,
    end_pose: Pose,
    radius: float = TURNING_RADIUS_CM,
):

    start = (start_pose.x_cm, start_pose.y_cm, start_pose.theta_rad)
    end = (end_pose.x_cm, end_pose.y_cm, end_pose.theta_rad)
    return rsplan.path(start, end, radius, 0, 0.5)


def dubins_length(
    start_pose: Pose,
    end_pose: Pose,
    radius: float = TURNING_RADIUS_CM,
) -> float:
    return dubins_path(start_pose, end_pose, radius).total_length


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

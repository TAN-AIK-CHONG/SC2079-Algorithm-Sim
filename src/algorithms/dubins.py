import math

import rsplan

from model import Robot

TURNING_RADIUS_CM = 35


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

import math

import rsplan

from model import Robot


# 35cm, not the physically-measured 30cm from earlier real-life testing:
# protocol v1.8's TURN,LEFT/RIGHT enforces its own minimum radius per
# direction (turn_control.c: CMD_LEFT_MIN_RADIUS_CM=18, right=35), and we
# plan every turn - left or right - at one shared radius, so it has to
# clear the higher of the two or every TURN,RIGHT command gets rejected
# with ERR,BAD_VALUE. 35 is exactly that minimum (turn_control.c's check
# is `radius_cm < min_radius_cm`, so equal to the bound is accepted).
# TIGHT_/DEFAULT_/LOOSE_ goal tolerance and TURN_CHANGE_PENALTY_CM in
# hybrid_astar.py were tuned against 30cm and have not been re-validated
# against this value - see this file's history for how that tuning was
# done if it needs redoing.
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

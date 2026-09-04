import math

import rsplan

from model import Robot

# rsplan only supports one shared radius (true asymmetric-radius Dubins
# paths aren't something it - or most Dubins libraries - solve), but
# hybrid_astar.py's real search turns tighter on the left (18cm) than the
# right (35cm; see its LEFT_TURNING_RADIUS_CM/RIGHT_TURNING_RADIUS_CM). This
# is used as an A* heuristic (hybrid_astar.py) and as TSP edge weights
# (graph.py) - both need a value that never OVERESTIMATES the true remaining
# distance, or they can bias the search toward a worse-than-optimal route.
# Since turning tighter never lengthens a Dubins path, estimating every turn
# at the tighter 18cm radius always gives a length <= what the asymmetric
# real search can achieve (which still can't turn tighter than 18cm on
# either side) - a valid, if slightly loose, lower bound. Estimating at the
# wider 35cm radius, as this used to, is NOT: a route with a real 18cm left
# turn can be genuinely shorter than any Dubins path this function would
# have offered as an estimate for it, which is exactly what "overestimate"
# means for a heuristic.
HEURISTIC_RADIUS_CM = 20


def dubins_path(
    start: Robot,
    end: Robot,
    radius: float = HEURISTIC_RADIUS_CM,
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
    radius: float = HEURISTIC_RADIUS_CM,
) -> float:
    return dubins_path(start, end, radius).total_length

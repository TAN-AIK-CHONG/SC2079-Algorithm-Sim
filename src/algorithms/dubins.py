import rsplan

from model import Pose

TURNING_RADIUS_CM = 25


def dubins_length(
    start_pose: Pose,
    end_pose: Pose,
    radius: float = TURNING_RADIUS_CM,
) -> float:
    start = (start_pose.x_cm, start_pose.y_cm, start_pose.theta_rad)
    end = (end_pose.x_cm, end_pose.y_cm, end_pose.theta_rad)
    path = rsplan.path(start, end, radius, 0, 0.5)
    return path.total_length
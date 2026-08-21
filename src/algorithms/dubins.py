import math
from model import Pose

TURNING_RADIUS_CM = 25


def dubins_length(
    start_pose: Pose,
    end_pose: Pose,
    radius: float = TURNING_RADIUS_CM,
) -> float:
    """Returns shortest Dubins path length."""

    candidates = [
        _lsl(start_pose, end_pose, radius),
        _rsr(start_pose, end_pose, radius),
        _lsr(start_pose, end_pose, radius),
        _rsl(start_pose, end_pose, radius),
        _rlr(start_pose, end_pose, radius),
        _lrl(start_pose, end_pose, radius),
    ]

    return min(c for c in candidates if c is not None)


def _lsl(start_pose: Pose, end_pose: Pose, radius: float):
    pass


def _rsr(start_pose: Pose, end_pose: Pose, radius: float):
    pass


def _lsr(start_pose: Pose, end_pose: Pose, radius: float):
    pass


def _rsl(start_pose: Pose, end_pose: Pose, radius: float):
    pass


def _rlr(start_pose: Pose, end_pose: Pose, radius: float):
    pass


def _lrl(start_pose: Pose, end_pose: Pose, radius: float):
    pass

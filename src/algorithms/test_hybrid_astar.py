"""
Tests _motion_primitives()'s dtheta signs directly - the bicycle-model
kinematics (dtheta/dt = (v/L) * tan(steering angle)) mean reversing with a
given steering side produces the OPPOSITE heading-change sign from driving
forward with that same side (the "reversing swings the car the opposite
way" behaviour familiar from parallel parking). Getting this backward
silently corrupts every pose the search chains after a reverse turn, with
no test anywhere else catching it (planner.py's and motor_controller.py's
own tests all mock around this layer). Run with:
pytest src/algorithms/test_hybrid_astar.py (from the repo root) or
pytest test_hybrid_astar.py (from inside src/algorithms/).
"""

from algorithms.hybrid_astar import _motion_primitives


def _by_name(primitives, name):
    return next(p for p in primitives if p.name == name)


def test_forward_left_and_reverse_right_share_dtheta_sign():
    # Same physical steering side (left) produces opposite dtheta for
    # forward vs reverse - so forward_left's sign matches reverse_RIGHT's,
    # not reverse_left's.
    primitives = _motion_primitives(10)
    forward_left = _by_name(primitives, "forward_left")
    reverse_left = _by_name(primitives, "reverse_left")
    reverse_right = _by_name(primitives, "reverse_right")

    assert forward_left.dtheta > 0
    assert reverse_left.dtheta < 0
    assert reverse_right.dtheta > 0
    assert forward_left.dtheta == reverse_right.dtheta
    assert reverse_left.dtheta == -forward_left.dtheta


def test_forward_right_and_reverse_left_share_dtheta_sign():
    primitives = _motion_primitives(10)
    forward_right = _by_name(primitives, "forward_right")
    reverse_left = _by_name(primitives, "reverse_left")

    assert forward_right.dtheta < 0
    assert forward_right.dtheta == reverse_left.dtheta


def test_straight_primitives_have_zero_dtheta_both_directions():
    primitives = _motion_primitives(10)
    assert _by_name(primitives, "forward_straight").dtheta == 0.0
    assert _by_name(primitives, "reverse_straight").dtheta == 0.0


def test_all_primitives_carry_the_requested_step_distance():
    primitives = _motion_primitives(10)
    assert all(p.distance == 10 for p in primitives)


def test_direction_sign_matches_forward_reverse_naming():
    primitives = _motion_primitives(10)
    for p in primitives:
        expected = 1 if p.name.startswith("forward") else -1
        assert p.direction == expected, p.name

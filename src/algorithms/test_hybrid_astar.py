"""
Tests _motion_primitives() directly: turns are always a fixed 90-degree arc
(the team's decision - see hybrid_astar.py's QUARTER_TURN_RAD comment),
regardless of the `step` argument, and the bicycle-model kinematics
(dtheta/dt = (v/L) * tan(steering angle)) still mean reversing with a given
steering side produces the OPPOSITE heading-change sign from driving
forward with that same side (the "reversing swings the car the opposite
way" behaviour familiar from parallel parking). Getting that sign backward
silently corrupts every pose the search chains after a reverse turn, with
no test anywhere else catching it (planner.py's and motor_controller.py's
own tests all mock around this layer). Run with:
pytest algorithms/test_hybrid_astar.py (from inside src/) - the plain
"algorithms.hybrid_astar" import below only resolves once src/ itself is on
sys.path, which needs src/ as the cwd; pointing pytest at this file from
anywhere else (e.g. the repo root) raises ModuleNotFoundError: No module
named 'algorithms'.
"""

from algorithms.hybrid_astar import (
    LEFT_TURNING_RADIUS_CM,
    QUARTER_TURN_RAD,
    RIGHT_TURNING_RADIUS_CM,
    _motion_primitives,
)


def _by_name(primitives, name):
    return next(p for p in primitives if p.name == name)


def test_turn_primitives_are_always_exactly_a_quarter_turn():
    # Regardless of `step` - only the straight primitives scale with it.
    for step in (10, 25, 1):
        primitives = _motion_primitives(step)
        for name in ("forward_left", "forward_right", "reverse_left", "reverse_right"):
            assert abs(_by_name(primitives, name).dtheta) == QUARTER_TURN_RAD, (step, name)


def test_turn_arc_length_matches_a_quarter_circle_at_that_sides_radius():
    primitives = _motion_primitives(10)
    assert _by_name(primitives, "forward_left").distance == round(LEFT_TURNING_RADIUS_CM * QUARTER_TURN_RAD)
    assert _by_name(primitives, "forward_right").distance == round(RIGHT_TURNING_RADIUS_CM * QUARTER_TURN_RAD)
    # same side -> same arc length regardless of forward/reverse
    assert _by_name(primitives, "reverse_left").distance == _by_name(primitives, "forward_left").distance
    assert _by_name(primitives, "reverse_right").distance == _by_name(primitives, "forward_right").distance


# NOTE: LEFT_TURNING_RADIUS_CM != RIGHT_TURNING_RADIUS_CM (each direction has
# its own protocol-enforced minimum - see hybrid_astar.py), so forward_left's
# arc LENGTH no longer equals forward_right's even though both sweep exactly
# the same 90 degrees - a left turn and a right turn are now genuinely
# different-sized arcs. The kinematic invariant this file actually exists to
# guard - reversing with a given steering side flips the sign of dtheta but
# not its magnitude, since forward and reverse on the same side share that
# side's one radius - only ever held SAME-side (forward_left vs reverse_left,
# forward_right vs reverse_right), never cross-side.
def test_forward_left_and_reverse_left_are_mirror_images():
    primitives = _motion_primitives(10)
    forward_left = _by_name(primitives, "forward_left")
    reverse_left = _by_name(primitives, "reverse_left")

    assert forward_left.dtheta == QUARTER_TURN_RAD
    assert reverse_left.dtheta == -forward_left.dtheta


def test_forward_right_and_reverse_right_are_mirror_images():
    primitives = _motion_primitives(10)
    forward_right = _by_name(primitives, "forward_right")
    reverse_right = _by_name(primitives, "reverse_right")

    assert forward_right.dtheta == -QUARTER_TURN_RAD
    assert reverse_right.dtheta == -forward_right.dtheta


def test_straight_primitives_have_zero_dtheta_both_directions():
    primitives = _motion_primitives(10)
    assert _by_name(primitives, "forward_straight").dtheta == 0.0
    assert _by_name(primitives, "reverse_straight").dtheta == 0.0


def test_straight_primitives_carry_the_requested_step_distance():
    # Only straight primitives scale with `step` - turns don't (see above).
    primitives = _motion_primitives(10)
    assert _by_name(primitives, "forward_straight").distance == 10
    assert _by_name(primitives, "reverse_straight").distance == 10


def test_direction_sign_matches_forward_reverse_naming():
    primitives = _motion_primitives(10)
    for p in primitives:
        expected = 1 if p.name.startswith("forward") else -1
        assert p.direction == expected, p.name

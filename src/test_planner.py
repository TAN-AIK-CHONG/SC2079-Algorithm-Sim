"""
Tests plan_mission()'s skip-unreachable-obstacle behavior (see planner.py)
without needing a real unreachable scenario - hybrid_astar is monkeypatched
per-test to fail for chosen legs, so the routing logic can be checked in
isolation. Run with: pytest src/test_planner.py (from the repo root) or
pytest test_planner.py (from inside src/).
"""

import planner
import pytest
from algorithms.hybrid_astar import HybridAstarResult, _motion_primitives
from model import Direction, MotionPrimitive, Obstacle, Robot


def make_scenario(num_obstacles: int) -> tuple[Robot, list[Obstacle]]:
    robot = Robot.from_grid(0, 0, Direction.NORTH)
    obstacles = [
        Obstacle(id=i, x_coord=5 + i * 3, y_coord=5, image_side=Direction.SOUTH) for i in range(num_obstacles)
    ]
    return robot, obstacles


def fake_result() -> HybridAstarResult:
    """A minimal, valid-looking hybrid_astar success result - one straight
    primitive is enough for _primitives_to_commands() to fold into a Command."""
    primitive = MotionPrimitive("forward_straight", 1, 0.0, 10)
    return HybridAstarResult(path=[], primitives=[primitive], length=10.0)


def test_all_reachable_visits_every_obstacle_in_order(monkeypatch):
    monkeypatch.setattr(planner, "hybrid_astar", lambda start, goal, obstacles: fake_result())

    robot, obstacles = make_scenario(3)
    plan = planner.plan_mission(robot, obstacles)

    assert plan.skipped_ids == []
    assert [leg.to_id for leg in plan.legs] == [obs.id for obs in obstacles] or len(plan.legs) == 3
    # every leg's from_id chains from the previous leg's to_id (or "S" for the first)
    assert plan.legs[0].from_id == "S"
    for prev_leg, leg in zip(plan.legs, plan.legs[1:]):
        assert leg.from_id == prev_leg.to_id


def test_unreachable_obstacle_is_skipped_not_fatal(monkeypatch):
    robot, obstacles = make_scenario(3)
    graph = planner.Graph.build(robot, obstacles)
    graph_order = planner.exhaustive_search(graph)
    unreachable_id = graph_order[1]  # whichever obstacle would be visited first
    unreachable_pose = next(n.viewing_pose for n in graph.nodes if n.id == unreachable_id)

    def fake_hybrid_astar(start, goal, footprints):
        # Fail only the leg landing on the unreachable obstacle; succeed otherwise.
        if goal == unreachable_pose:
            return None
        return fake_result()

    monkeypatch.setattr(planner, "hybrid_astar", fake_hybrid_astar)

    plan = planner.plan_mission(robot, obstacles)

    assert plan.skipped_ids == [unreachable_id]
    assert unreachable_id not in [leg.to_id for leg in plan.legs]
    # the two other obstacles still got visited - mission wasn't abandoned
    assert len(plan.legs) == 2


def test_current_position_carries_forward_across_a_skip(monkeypatch):
    """After skipping obstacle B, the next attempted leg must start from
    wherever the robot last successfully arrived (A, or "S" if nothing was
    reached yet) - not reset to "S" or silently jump from B."""
    robot, obstacles = make_scenario(3)
    graph = planner.Graph.build(robot, obstacles)
    order = planner.exhaustive_search(graph)
    skip_id = order[1]
    skip_pose = next(n.viewing_pose for n in graph.nodes if n.id == skip_id)

    def fake_hybrid_astar(start, goal, footprints):
        if goal == skip_pose:
            return None
        return fake_result()

    monkeypatch.setattr(planner, "hybrid_astar", fake_hybrid_astar)
    plan = planner.plan_mission(robot, obstacles)

    # first leg starts from "S"; nothing ever claims to start from the
    # skipped obstacle, since the robot never actually reached it
    assert plan.legs[0].from_id == "S"
    assert all(leg.from_id != skip_id for leg in plan.legs)


def test_planning_error_only_when_nothing_at_all_is_reachable(monkeypatch):
    monkeypatch.setattr(planner, "hybrid_astar", lambda start, goal, obstacles: None)

    robot, obstacles = make_scenario(3)
    with pytest.raises(planner.PlanningError):
        planner.plan_mission(robot, obstacles)


def _named(name: str) -> MotionPrimitive:
    return next(p for p in _motion_primitives(10) if p.name == name)


def test_combine_arcs_labels_by_steering_side_not_dtheta_sign():
    """_combine_arcs must read LEFT/RIGHT off the primitive's own name, not
    off dtheta's sign - reversing flips which dtheta sign a given steering
    side produces (see test_hybrid_astar.py), so a run of "reverse_left"
    primitives (steer=LEFT, dtheta<0) must still come out labelled LEFT."""
    reverse_left_run = [_named("reverse_left")] * 3
    command = planner._combine_arcs(reverse_left_run)
    assert command.turn == "LEFT"
    assert command.direction == "REVERSE"
    assert _named("reverse_left").dtheta < 0  # sanity: dtheta really is negative here

    reverse_right_run = [_named("reverse_right")] * 3
    command = planner._combine_arcs(reverse_right_run)
    assert command.turn == "RIGHT"
    assert command.direction == "REVERSE"
    assert _named("reverse_right").dtheta > 0  # sanity: dtheta really is positive here


def test_combine_arcs_labels_forward_turns_correctly_too():
    command = planner._combine_arcs([_named("forward_left")] * 2)
    assert command.turn == "LEFT" and command.direction == "FORWARD"

    command = planner._combine_arcs([_named("forward_right")] * 2)
    assert command.turn == "RIGHT" and command.direction == "FORWARD"


def test_combine_arcs_swept_angle_is_always_positive_magnitude():
    for name in ("forward_left", "forward_right", "reverse_left", "reverse_right"):
        command = planner._combine_arcs([_named(name)] * 4)
        assert command.swept_angle_deg > 0, name

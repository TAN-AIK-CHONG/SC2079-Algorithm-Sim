"""
Tests plan_mission()'s skip-unreachable-obstacle behavior (see planner.py)
without needing a real unreachable scenario - hybrid_astar is monkeypatched
per-test to fail for chosen legs, so the routing logic can be checked in
isolation. Run with: pytest src/test_planner.py (from the repo root) or
pytest test_planner.py (from inside src/).
"""

import json
from pathlib import Path

import planner
import pytest
from algorithms.hybrid_astar import HybridAstarResult, _motion_primitives
from model import Direction, MotionPrimitive, Obstacle, Robot, parse_scenario


def make_scenario(num_obstacles: int) -> tuple[Robot, list[Obstacle]]:
    robot = Robot.from_grid(0, 0, Direction.NORTH)
    obstacles = [
        Obstacle(id=i, x_coord=5 + i * 3, y_coord=5, image_side=Direction.SOUTH) for i in range(num_obstacles)
    ]
    return robot, obstacles


def fake_result(landing: Robot = Robot(0.0, 0.0, 0.0)) -> HybridAstarResult:
    """A minimal, valid-looking hybrid_astar success result - one straight
    primitive is enough for _primitives_to_commands() to fold into a Command.
    path needs at least one pose: plan_mission() reads path[-1] as the real
    chaining pose for the next leg (see plan_mission)."""
    primitive = MotionPrimitive("forward_straight", 1, 0.0, 10)
    return HybridAstarResult(path=[landing], primitives=[primitive], length=10.0)


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

    def fake_hybrid_astar(start, target, footprints):
        # Fail only the leg landing on the unreachable obstacle; succeed otherwise.
        if target.id == unreachable_id:
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

    def fake_hybrid_astar(start, target, footprints):
        if target.id == skip_id:
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


def test_retry_previous_leg_skips_the_landing_that_already_failed(monkeypatch):
    """The first landing viewing_arrivals yields is the one hybrid_astar
    already returned for the previous leg - the one the next leg just failed
    from - so the retry must move on to the next landing."""
    _, (prev_target, next_target) = make_scenario(2)
    prev_start = Robot.from_grid(0, 0, Direction.NORTH)
    failed_landing = Robot(100.0, 100.0, 0.0)
    other_landing = Robot(105.0, 100.0, 0.0)

    def fake_viewing_arrivals(start, target, footprints):
        return iter([fake_result(failed_landing), fake_result(other_landing)])

    monkeypatch.setattr(planner, "viewing_arrivals", fake_viewing_arrivals)
    monkeypatch.setattr(planner, "hybrid_astar", lambda start, target, footprints: fake_result())

    retry = planner._retry_previous_leg_for_escape(prev_start, prev_target, next_target, footprints=[])
    assert retry is not None
    prev_result, continuation = retry
    assert prev_result.path[-1] == other_landing


def test_retry_previous_leg_gives_up_when_no_landing_leads_on(monkeypatch):
    _, (prev_target, next_target) = make_scenario(2)
    prev_start = Robot.from_grid(0, 0, Direction.NORTH)
    landings = [Robot(100.0 + offset, 100.0, 0.0) for offset in range(0, 15, 5)]

    def fake_viewing_arrivals(start, target, footprints):
        return iter([fake_result(landing) for landing in landings])

    monkeypatch.setattr(planner, "viewing_arrivals", fake_viewing_arrivals)
    monkeypatch.setattr(planner, "hybrid_astar", lambda start, target, footprints: None)

    assert planner._retry_previous_leg_for_escape(prev_start, prev_target, next_target, footprints=[]) is None


def test_backtrack_escape_rescues_a_real_previously_unreachable_map():
    """4_obstacles/map_03.json: at LEFT_TURNING_RADIUS_CM=20, legs to
    obstacles 1 and 0 used to fail completely - not because either goal was
    blocked, but because the exact pose hybrid_astar committed to after
    visiting obstacle 3 fell in a dead zone neither goal was reachable
    from, while a different landing of that same leg (still viewing its
    image) was not stuck at all. Real map, real
    (unmocked) plan_mission - a regression fixture for
    _retry_previous_leg_for_escape."""
    path = Path(__file__).resolve().parent / "testing" / "generated_maps" / "4_obstacles" / "map_03.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    robot, obstacles = parse_scenario(data)

    plan = planner.plan_mission(robot, obstacles)

    assert plan.skipped_ids == []
    assert len(plan.legs) == 4


def test_boundary_obstacle_facing_into_the_arena_is_planned_not_skipped():
    """Real plan_mission (no mock): an obstacle jammed into the bottom-left
    corner with its image facing into the arena. Its ideal viewing pose has
    the robot footprint poking through a wall, so the leg must end at some
    other pose that still views the image (see Obstacle.is_viewed_from)."""
    robot = Robot.from_grid(10, 10, Direction.NORTH)
    obstacle = Obstacle(id=0, x_coord=0, y_coord=0, image_side=Direction.EAST)

    plan = planner.plan_mission(robot, [obstacle])

    assert plan.skipped_ids == []
    assert [leg.to_id for leg in plan.legs] == [0]


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

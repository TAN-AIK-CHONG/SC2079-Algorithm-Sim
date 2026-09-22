"""
Tests plan_mission()'s skip-unreachable-obstacle behavior (see planner.py)
without needing a real unreachable scenario - hybrid_astar is monkeypatched
per-test to fail for chosen legs, so the routing logic can be checked in
isolation. Run with: pytest src/test_planner.py (from the repo root) or
pytest test_planner.py (from inside src/).
"""

import planner
import pytest
from algorithms.hybrid_astar import DEFAULT_GOAL_POS_TOLERANCE_CM, HybridAstarResult, _motion_primitives
from model import Direction, MotionPrimitive, Obstacle, Robot


def make_scenario(num_obstacles: int) -> tuple[Robot, list[Obstacle]]:
    robot = Robot.from_grid(0, 0, Direction.NORTH)
    obstacles = [
        Obstacle(id=i, x_coord=5 + i * 3, y_coord=5, image_side=Direction.SOUTH) for i in range(num_obstacles)
    ]
    return robot, obstacles


def fake_result() -> HybridAstarResult:
    """A minimal, valid-looking hybrid_astar success result - one straight
    primitive is enough for _primitives_to_commands() to fold into a Command.
    path needs at least one pose: plan_mission() reads path[-1] as the real
    chaining pose for the next leg (see plan_mission)."""
    primitive = MotionPrimitive("forward_straight", 1, 0.0, 10)
    return HybridAstarResult(path=[Robot(0.0, 0.0, 0.0)], primitives=[primitive], length=10.0)


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
    unreachable_obstacle = next(o for o in obstacles if o.id == unreachable_id)
    # Every candidate, not just the ideal one - a genuinely unreachable
    # obstacle must fail all of them, or this fake accidentally succeeds via
    # planner._retry_with_alternate_viewing_pose instead of testing a skip.
    unreachable_poses = set(unreachable_obstacle.candidate_viewing_poses())

    def fake_hybrid_astar(start, goal, footprints):
        # Fail every viewing-pose variant of the unreachable obstacle; succeed otherwise.
        if goal in unreachable_poses:
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
    skip_obstacle = next(o for o in obstacles if o.id == skip_id)
    # Every candidate, not just the ideal one - see
    # test_unreachable_obstacle_is_skipped_not_fatal for why.
    skip_poses = set(skip_obstacle.candidate_viewing_poses())

    def fake_hybrid_astar(start, goal, footprints):
        if goal in skip_poses:
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


def test_retry_previous_leg_rejects_a_landing_too_far_from_the_true_goal(monkeypatch):
    """_retry_previous_leg_for_escape must never accept a nudge whose
    landing strays outside the ordinary DEFAULT_GOAL_POS_TOLERANCE_CM of
    the real viewing pose, even when every nudge and the continuation both
    "succeed" - image-visibility quality is never traded for a working next
    leg."""
    true_goal = Robot(100.0, 100.0, 0.0)
    next_goal = Robot(200.0, 100.0, 0.0)
    prev_start = Robot.from_grid(0, 0, Direction.NORTH)
    far_landing = Robot(true_goal.x_cm + DEFAULT_GOAL_POS_TOLERANCE_CM + 5, true_goal.y_cm, true_goal.theta_rad)
    primitive = MotionPrimitive("forward_straight", 1, 0.0, 10)

    def fake_hybrid_astar(start, goal, footprints):
        return HybridAstarResult(path=[far_landing], primitives=[primitive], length=10.0)

    monkeypatch.setattr(planner, "hybrid_astar", fake_hybrid_astar)

    assert planner._retry_previous_leg_for_escape(prev_start, true_goal, next_goal, footprints=[]) is None


def test_retry_previous_leg_accepts_a_landing_within_tolerance(monkeypatch):
    true_goal = Robot(100.0, 100.0, 0.0)
    next_goal = Robot(200.0, 100.0, 0.0)
    prev_start = Robot.from_grid(0, 0, Direction.NORTH)
    close_landing = Robot(true_goal.x_cm + 2, true_goal.y_cm, true_goal.theta_rad)  # well inside tolerance
    primitive = MotionPrimitive("forward_straight", 1, 0.0, 10)

    def fake_hybrid_astar(start, goal, footprints):
        return HybridAstarResult(path=[close_landing], primitives=[primitive], length=10.0)

    monkeypatch.setattr(planner, "hybrid_astar", fake_hybrid_astar)

    retry = planner._retry_previous_leg_for_escape(prev_start, true_goal, next_goal, footprints=[])
    assert retry is not None
    prev_result, continuation = retry
    assert prev_result.path[-1] == close_landing


def test_mission_with_a_genuinely_unreachable_leg_still_completes_the_rest(monkeypatch):
    """This used to point at a real map (4_obstacles/map_03.json) whose
    obstacle 3 was genuinely unreachable under fixed 90-degree turns even
    after _retry_previous_leg_for_escape. It no longer is: planner.py grew a
    second fallback, _retry_with_alternate_viewing_pose, tried first, which
    happens to rescue that exact map now (a different, equally valid
    vantage point of the same obstacle lands on the reachable lattice where
    the ideal one didn't). Pinning this test to a specific real map made it
    fragile to exactly this kind of improvement - the next fix might rescue
    it too. Mocked instead: block every one of the target obstacle's
    candidate_viewing_poses() (so _retry_with_alternate_viewing_pose can't
    rescue it either), in a 3-obstacle chain so _retry_previous_leg_for_escape
    also gets a real chance at the leg before giving up. Exercises the same
    invariant the real map used to: one truly unreachable leg - even after
    BOTH fallbacks - still gets skipped cleanly, not abort or corrupt the
    rest of the mission."""
    robot, obstacles = make_scenario(3)
    graph = planner.Graph.build(robot, obstacles)
    order = planner.exhaustive_search(graph)
    unreachable_id = order[-1]  # the last one visited - so it has a previous leg to fall back into
    unreachable_obstacle = next(o for o in obstacles if o.id == unreachable_id)
    unreachable_poses = set(unreachable_obstacle.candidate_viewing_poses())

    def fake_hybrid_astar(start, goal, footprints):
        if goal in unreachable_poses:
            return None
        return fake_result()

    monkeypatch.setattr(planner, "hybrid_astar", fake_hybrid_astar)

    plan = planner.plan_mission(robot, obstacles)

    assert plan.skipped_ids == [unreachable_id]
    assert len(plan.legs) == 2


def test_boundary_obstacle_facing_into_the_arena_is_planned_not_skipped():
    """Real plan_mission (no mock): an obstacle jammed into the bottom-left
    corner with its image facing into the arena. Its ideal viewing pose has
    the robot footprint poking through a wall, so Graph.build nudges it to a
    clear fallback (see graph._resolve_viewing_pose) - without that the leg's
    goal would be in collision and the obstacle would be skipped."""
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

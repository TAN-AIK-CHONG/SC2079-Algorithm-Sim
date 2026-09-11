"""
Tests Graph.build's viewing-pose resolution: when an obstacle's ideal
viewing pose is blocked (a neighbour standing in it, or the obstacle close
enough to a wall that the ideal standoff is out of bounds), the node should
carry a nearby unobstructed fallback pose instead of the blocked ideal - so
plan_mission aims hybrid_astar somewhere reachable rather than skipping the
obstacle outright. See model.Obstacle.candidate_viewing_poses and
graph._resolve_viewing_pose.

Run with: pytest algorithms/test_graph.py (from inside src/) - the plain
"algorithms.graph" / "model" / "collision" imports only resolve once src/
is on sys.path, which needs src/ as the cwd.
"""

import pytest

from algorithms.graph import Graph, _resolve_viewing_pose
from collision import footprint_in_collision
from model import Direction, Obstacle, Robot

START = Robot.from_grid(0, 0, Direction.NORTH)


def _footprints(*obstacles):
    return [obs.footprint_corners_cm() for obs in obstacles]


@pytest.mark.parametrize("image_side", list(Direction))
def test_first_candidate_is_the_ideal_viewing_pose(image_side):
    obstacle = Obstacle(0, 10, 10, image_side)
    assert obstacle.candidate_viewing_poses()[0] == obstacle.cm_viewing_position()


@pytest.mark.parametrize("image_side", list(Direction))
def test_every_candidate_keeps_the_ideal_facing(image_side):
    obstacle = Obstacle(0, 10, 10, image_side)
    ideal_theta = obstacle.cm_viewing_position().theta_rad
    assert {pose.theta_rad for pose in obstacle.candidate_viewing_poses()} == {ideal_theta}


def test_resolve_returns_the_ideal_pose_when_nothing_blocks_it():
    obstacle = Obstacle(0, 10, 10, Direction.SOUTH)
    resolved = _resolve_viewing_pose(obstacle, _footprints(obstacle))
    assert resolved == obstacle.cm_viewing_position()


def test_resolve_picks_a_clear_fallback_when_a_neighbour_blocks_the_ideal():
    target = Obstacle(0, 10, 10, Direction.SOUTH)  # ideal viewing robot spans x 90-120, y 50-80
    blocker = Obstacle(1, 11, 6, Direction.NORTH)  # footprint (110,60)-(120,70), sits in that
    footprints = _footprints(target, blocker)

    assert footprint_in_collision(target.cm_viewing_position(), footprints)  # precondition

    resolved = _resolve_viewing_pose(target, footprints)
    assert resolved != target.cm_viewing_position()
    assert resolved in target.candidate_viewing_poses()
    assert not footprint_in_collision(resolved, footprints)


def test_resolve_falls_back_to_the_ideal_when_every_candidate_is_blocked():
    # Image faces the west wall from one cell away - no candidate (all just
    # small nudges) can fit the robot between obstacle and wall.
    obstacle = Obstacle(0, 0, 10, Direction.WEST)
    footprints = _footprints(obstacle)
    assert all(
        footprint_in_collision(pose, footprints)
        for pose in obstacle.candidate_viewing_poses()
    )

    resolved = _resolve_viewing_pose(obstacle, footprints)
    assert resolved == obstacle.cm_viewing_position()  # the ideal, unchanged - obstacle still gets skipped downstream


# --- obstacle sitting on the arena boundary, image facing INTO the arena ---
# (the common real case: obstacle pushed against a wall, photo taken from the
# open side). The ideal standoff is well clear of that wall; resolution only
# has to step in when the robot's own 30cm-wide footprint clips a
# PERPENDICULAR wall - i.e. near a corner.

@pytest.mark.parametrize(
    "grid_x, grid_y, image_side",
    [
        (0, 10, Direction.EAST),   # flush on the west wall, looking east
        (19, 10, Direction.WEST),  # flush on the east wall, looking west
        (10, 0, Direction.NORTH),  # flush on the south wall, looking north
        (10, 19, Direction.SOUTH), # flush on the north wall, looking north-to-south
    ],
)
def test_mid_edge_boundary_obstacle_facing_inward_keeps_its_ideal_pose(grid_x, grid_y, image_side):
    obstacle = Obstacle(0, grid_x, grid_y, image_side)
    footprints = _footprints(obstacle)
    assert not footprint_in_collision(obstacle.cm_viewing_position(), footprints)  # ideal already fits

    resolved = _resolve_viewing_pose(obstacle, footprints)
    assert resolved == obstacle.cm_viewing_position()


@pytest.mark.parametrize(
    "grid_x, grid_y, image_side",
    [
        (0, 0, Direction.EAST),     # bottom-left corner, looking east - footprint clips the south wall
        (0, 0, Direction.NORTH),    # bottom-left corner, looking north - footprint clips the west wall
        (19, 19, Direction.SOUTH),  # top-right corner, looking south - footprint clips the east wall
        (19, 19, Direction.WEST),   # top-right corner, looking west - footprint clips the north wall
    ],
)
def test_corner_obstacle_facing_inward_is_nudged_along_the_face_into_bounds(grid_x, grid_y, image_side):
    obstacle = Obstacle(0, grid_x, grid_y, image_side)
    footprints = _footprints(obstacle)
    assert footprint_in_collision(obstacle.cm_viewing_position(), footprints)  # ideal pokes through a wall

    resolved = _resolve_viewing_pose(obstacle, footprints)
    assert resolved != obstacle.cm_viewing_position()
    assert resolved in obstacle.candidate_viewing_poses()
    assert not footprint_in_collision(resolved, footprints)
    assert resolved.theta_rad == obstacle.cm_viewing_position().theta_rad  # still faces the image


def test_obstacle_one_cell_from_the_wall_is_pulled_back_in_bounds():
    # Ideal standoff puts the robot just outside the west wall; nudging one
    # cell toward the obstacle (10cm into the camera clearance) fits it.
    obstacle = Obstacle(0, 4, 10, Direction.WEST)
    footprints = _footprints(obstacle)
    assert footprint_in_collision(obstacle.cm_viewing_position(), footprints)  # ideal is out of bounds

    resolved = _resolve_viewing_pose(obstacle, footprints)
    assert not footprint_in_collision(resolved, footprints)
    assert resolved != obstacle.cm_viewing_position()


def test_build_nodes_carry_the_resolved_pose_not_the_blocked_ideal():
    target = Obstacle(0, 10, 10, Direction.SOUTH)
    blocker = Obstacle(1, 11, 6, Direction.NORTH)

    graph = Graph.build(START, [target, blocker])
    target_node = next(node for node in graph.nodes if node.id == 0)

    assert target_node.viewing_pose == _resolve_viewing_pose(target, _footprints(target, blocker))
    assert not footprint_in_collision(target_node.viewing_pose, _footprints(target, blocker))
    assert target_node.viewing_pose != target.cm_viewing_position()


def test_build_leaves_an_unobstructed_obstacle_at_its_ideal_pose():
    obstacle = Obstacle(0, 10, 10, Direction.SOUTH)
    graph = Graph.build(START, [obstacle])
    obstacle_node = next(node for node in graph.nodes if node.id == 0)
    assert obstacle_node.viewing_pose == obstacle.cm_viewing_position()

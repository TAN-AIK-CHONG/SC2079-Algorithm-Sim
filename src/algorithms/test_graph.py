"""
Tests Graph.build's nodes: each obstacle node carries its ideal viewing
pose, even when that pose is blocked - it only estimates edge weights for
the visiting order, since a leg may end at any pose that views the image
(see model.Obstacle.is_viewed_from).

Run with: pytest algorithms/test_graph.py (from inside src/) - the plain
"algorithms.graph" / "model" imports only resolve once src/ is on sys.path,
which needs src/ as the cwd.
"""

from algorithms.graph import Graph
from model import Direction, Obstacle, Robot

START = Robot.from_grid(0, 0, Direction.NORTH)


def test_build_nodes_carry_the_ideal_viewing_pose():
    obstacle = Obstacle(0, 10, 10, Direction.SOUTH)
    graph = Graph.build(START, [obstacle])
    obstacle_node = next(node for node in graph.nodes if node.id == 0)
    assert obstacle_node.viewing_pose == obstacle.cm_viewing_position()


def test_build_keeps_the_ideal_pose_even_when_a_neighbour_blocks_it():
    target = Obstacle(0, 10, 10, Direction.SOUTH)
    blocker = Obstacle(1, 11, 6, Direction.NORTH)

    graph = Graph.build(START, [target, blocker])
    target_node = next(node for node in graph.nodes if node.id == 0)

    assert target_node.viewing_pose == target.cm_viewing_position()

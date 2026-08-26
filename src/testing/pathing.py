from algorithms.hybrid_astar import hybrid_astar
from algorithms.graph import Graph
from model import Obstacle


def calculate_final_path(
    graph: Graph, obstacles: list[Obstacle], visit_order: list[str]
):
    """Stitch hybrid A* legs between consecutive nodes in visit_order into one path.

    Returns (path, total_length_cm, completed_legs, failed_leg). failed_leg is
    None on full success, else the (start_id, goal_id) of the first leg that
    could not be planned.
    """
    robots = {node.id: node.robot for node in graph.nodes}
    obstacle_footprints = [obstacle.footprint_corners_cm() for obstacle in obstacles]
    final_path = []
    total_length = 0.0
    completed_legs = 0

    for start_id, goal_id in zip(visit_order, visit_order[1:]):
        result = hybrid_astar(
            robots[start_id],
            robots[goal_id],
            obstacle_footprints,
        )
        if result is None:
            return final_path, total_length, completed_legs, (start_id, goal_id)

        final_path.extend(result.path if not final_path else result.path[1:])
        total_length += result.length
        completed_legs += 1

    return final_path, total_length, completed_legs, None

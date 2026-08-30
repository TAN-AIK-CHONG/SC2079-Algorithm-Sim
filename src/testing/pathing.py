from algorithms.hybrid_astar import hybrid_astar
from algorithms.graph import Graph
from model import Obstacle


def calculate_final_path(
    graph: Graph, obstacles: list[Obstacle], visit_order: list[str]
):
    """Stitch hybrid A* legs between consecutive nodes in visit_order into one path.

    If a leg can't be planned (obstacle unreachable from wherever the robot
    currently is), that obstacle is skipped and stitching continues toward
    the next one in visit_order, still from the last position actually
    reached - mirrors planner.plan_mission()'s skip behaviour, so this
    reachability count isn't an undercount from stopping at the first miss.

    Returns (path, total_length_cm, completed_legs, skipped_legs).
    skipped_legs is a list of (start_id, goal_id) pairs that could not be
    planned; empty if every obstacle in visit_order was reached.
    """
    robots = {node.id: node.viewing_pose for node in graph.nodes}
    obstacle_footprints = [obstacle.footprint_corners_cm() for obstacle in obstacles]
    final_path = []
    total_length = 0.0
    completed_legs = 0
    skipped_legs = []
    current_id = visit_order[0]

    for goal_id in visit_order[1:]:
        result = hybrid_astar(
            robots[current_id],
            robots[goal_id],
            obstacle_footprints,
        )
        if result is None:
            skipped_legs.append((current_id, goal_id))
            continue  # stay at current_id, try the next stop in visit_order instead

        final_path.extend(result.path if not final_path else result.path[1:])
        total_length += result.length
        completed_legs += 1
        current_id = goal_id

    return final_path, total_length, completed_legs, skipped_legs

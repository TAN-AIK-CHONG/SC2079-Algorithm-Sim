import json
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
TEST_MAPS_DIR = ROOT_DIR / "test_maps"
sys.path.insert(0, str(ROOT_DIR / "src"))

from algorithms.hamiltonian import run_all
from algorithms.hybrid_astar import hybrid_astar
from graph import Graph
from model import Direction, Obstacle, Robot


def load_map(path: Path) -> tuple[Robot, list[Obstacle]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    robot = Robot(
        data["robot"]["x_coord"],
        data["robot"]["y_coord"],
        Direction[data["robot"]["facing"]],
    )
    obstacles = [
        Obstacle(
            obstacle["id"],
            obstacle["x_coord"],
            obstacle["y_coord"],
            Direction[obstacle["image_side"]],
        )
        for obstacle in data["obstacles"]
    ]
    return robot, obstacles


def calculate_final_path(graph, obstacles, visit_order):
    nodes = {node.id: node.pose for node in graph.nodes}
    obstacle_boxes = [obstacle.inflated_bounds() for obstacle in obstacles]
    final_path = []
    total_length = 0.0
    completed_legs = 0

    for start_id, goal_id in zip(visit_order, visit_order[1:]):
        result = hybrid_astar(
            nodes[start_id],
            nodes[goal_id],
            obstacle_boxes,
        )
        if result is None:
            return final_path, total_length, completed_legs, (start_id, goal_id)

        final_path.extend(result.poses if not final_path else result.poses[1:])
        total_length += result.length
        completed_legs += 1

    return final_path, total_length, completed_legs, None


def main():
    map_paths = sorted(TEST_MAPS_DIR.glob("*.json"))

    for map_path in map_paths:
        robot, obstacles = load_map(map_path)
        graph = Graph.build(robot, obstacles)

        # Run every Hamiltonian algorithm, then use the optimal result below.
        hamiltonian_results = run_all(graph)
        best_order = hamiltonian_results["exhaustive_search"]["path"]

        path_started = time.perf_counter()
        final_path, final_length, completed_legs, failed_leg = calculate_final_path(
            graph, obstacles, best_order
        )
        path_time = time.perf_counter() - path_started

        print(f"Map: {map_path.name}")
        print("Hamiltonian results:")
        for name, result in hamiltonian_results.items():
            print(
                f"  {name:>18}: "
                f"time={result['time'] * 1000:.3f} ms, "
                f"accuracy={result['accuracy']:.2%}, "
                f"path={' -> '.join(result['path'])}"
            )

        total_legs = len(best_order) - 1
        completeness = completed_legs / total_legs if total_legs else 1.0
        print("Path planning (using exhaustive_search):")
        print(f"  time={path_time * 1000:.3f} ms")
        print(f"  completeness={completed_legs}/{total_legs} ({completeness:.2%})")
        print(f"  length={final_length:.2f} cm, poses={len(final_path)}")
        if failed_leg:
            print(f"  failed leg={failed_leg[0]} -> {failed_leg[1]}")
        print()


if __name__ == "__main__":
    main()

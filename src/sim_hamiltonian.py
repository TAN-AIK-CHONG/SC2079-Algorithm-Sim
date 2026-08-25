import json
from pathlib import Path

from graph import Graph
from algorithms.hamiltonian import run_all
from model import Obstacle, Robot, parse_scenario

TEST_MAPS_DIR = Path("test_maps")


def load_map(path: Path) -> tuple[Robot, list[Obstacle]]:
    with open(path) as f:
        data = json.load(f)
    return parse_scenario(data)


def main():
    map_paths = sorted(TEST_MAPS_DIR.glob("*.json"))
    if not map_paths:
        raise SystemExit(f"No JSON maps found in {TEST_MAPS_DIR.resolve()}")

    for map_path in map_paths:
        robot, obstacles = load_map(map_path)
        graph = Graph.build(robot, obstacles, radius=25)
        results = run_all(graph)

        print(f"Map: {map_path.name} ({len(obstacles)} obstacles)\n")
        for name, r in results.items():
            print(
                f"{name:>18}: path={r['path']}  "
                f"length={r['length']:.2f}cm  "
                f"time={r['time']*1000:.3f}ms  "
                f"accuracy={r['accuracy']:.2%}"
            )
        print("---")


if __name__ == "__main__":
    main()
